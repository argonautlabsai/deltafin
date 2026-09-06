//! Adaptive admission for PILOT speculative expert prefetch.
//!
//! PILOT's lookahead recall averages ~50% but collapses at layers sitting at
//! or immediately after MLA attention boundaries (measured 1.2% at layer 13),
//! where the snapshot it routes on is missing the large update the next
//! attention step is about to apply. Rather than hardcoding the measured bad
//! layers, this governor scores every prediction against the authoritative
//! routes on every pass (both ID lists already exist; the comparison is a few
//! hundred bit tests) and gates only the speculative *disk reads* per layer on
//! the trailing measured recall. A second, cheaper predictor — the layer's own
//! routing from the previous token — is scored in parallel and takes over a
//! layer when it is measurably better, so collapsed layers degrade to a
//! ~30%-recall guess instead of wasting reads on a ~1% one.
//!
//! Everything here is scheduling-only and advisory by construction: the
//! governor holds no I/O resources, the authoritative demand path never
//! consults it, out-of-range inputs are skipped rather than asserted, and a
//! wrong decision costs bandwidth or latency, never output. See
//! docs/PILOT-GATE.md for the complete design rationale.

use crate::config::PilotGateRequest;
use crate::program::K3_LAYER_COUNT;
pub(crate) use crate::provider::{PILOT_MAX_PREFETCH, ROUTE_TOP_K};

/// K3 routes over this many experts per layer; IDs at or above the bound are
/// rejected by the provider ABI and skipped defensively here.
const K3_ROUTED_EXPERTS: usize = 896;
const EXPERT_BITSET_WORDS: usize = K3_ROUTED_EXPERTS.div_ceil(64);
/// One EMA step per scored pass; ~10-sample effective window, responsive at
/// the default 16-sample warmup without flapping on single outliers.
const EMA_ALPHA: f32 = 0.2;
/// A layer logs its first few suppress/resume transitions and then goes
/// quiet, so an EMA hovering at the threshold cannot spam the run log.
const FLIP_LOG_CAP: u32 = 6;

/// A validated speculative read set for one upcoming layer: the generalization
/// of the provider's PILOT hint that `try_schedule_expert_prefetch` consumes,
/// whichever predictor produced it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertPrefetchPlan {
    target_layer: u32,
    expert_count: u8,
    expert_ids: [u16; PILOT_MAX_PREFETCH],
}

/// Speculative plans narrower than the authoritative top-16 are legitimate
/// exactly when the true-route top-up guarantees the balance of the layer's
/// experts still gets prefetched at mailbox-open time: K3_TRUE_ROUTE_TOPUP=1
/// lowers the accepted floor to one expert, otherwise the historical top-16
/// floor keeps the legacy schedule byte-identical.
pub(crate) fn prefetch_plan_floor() -> usize {
    static FLOOR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        if std::env::var("K3_TRUE_ROUTE_TOPUP").is_ok_and(|value| value == "1") {
            1
        } else {
            ROUTE_TOP_K
        }
    })
}

impl ExpertPrefetchPlan {
    /// Accepts only canonical plans: a routed target layer and an ascending,
    /// duplicate-free ID set within the prefetch bounds. Malformed input is a
    /// skipped prediction, never an error.
    pub(crate) fn new(target_layer: u32, expert_ids: &[u16]) -> Option<Self> {
        if !(1..K3_LAYER_COUNT as u32).contains(&target_layer)
            || !(prefetch_plan_floor()..=PILOT_MAX_PREFETCH).contains(&expert_ids.len())
            || expert_ids
                .iter()
                .any(|&expert| expert as usize >= K3_ROUTED_EXPERTS)
            || expert_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return None;
        }
        let mut ids = [0_u16; PILOT_MAX_PREFETCH];
        ids[..expert_ids.len()].copy_from_slice(expert_ids);
        Some(Self {
            target_layer,
            expert_count: expert_ids.len() as u8,
            expert_ids: ids,
        })
    }

    pub(crate) const fn target_layer(&self) -> u32 {
        self.target_layer
    }

    pub(crate) fn expert_ids(&self) -> &[u16] {
        &self.expert_ids[..self.expert_count as usize]
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PredictorScore {
    ema: f32,
    samples: u32,
}

impl PredictorScore {
    fn absorb(&mut self, sample: f32) {
        self.ema = if self.samples == 0 {
            sample
        } else {
            self.ema + EMA_ALPHA * (sample - self.ema)
        };
        self.samples = self.samples.saturating_add(1);
    }

    const fn warmed(&self, warmup: u32) -> bool {
        self.samples >= warmup
    }
}

/// Which predictor currently owns a layer's speculative reads, judged purely
/// from stored score state. A pass where the winning predictor has nothing to
/// offer (a provider hint miss, an unseeded previous token) can still fall
/// through to the other one; this is the standing preference, shared by
/// `admit` and the report so the two cannot drift.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PreferredPredictor {
    Warming,
    Pilot,
    PrevToken,
}

impl PreferredPredictor {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Pilot => "pilot",
            Self::PrevToken => "prev-token",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct LayerGateState {
    pilot: PredictorScore,
    prev_token: PredictorScore,
    /// Hit rate of the UNION of this pass's pilot prediction with the
    /// previous token's authoritative routes. Measure-only: it never selects
    /// a predictor and never changes a read. It exists to price a re-ranked
    /// pilot — one that uses the previous token's route as a PRIOR inside the
    /// existing 16-slot budget, rather than widening the budget (widening was
    /// measured -8.9% at topk 24 and -16.5% at 32 on 2026-08-25, because the
    /// extra speculative reads displace demand reads). `union - pilot` is the
    /// share of the pilot's ~38% misses that such a prior could recover, and
    /// therefore the ceiling on the whole idea.
    union: PredictorScore,
    /// The most recent position's authoritative routing, ascending: the
    /// "previous token's routing" predictor for this layer's next pass.
    prev_routes: Option<[u16; ROUTE_TOP_K]>,
    reads_suppressed: bool,
    flip_logs: u32,
}

impl LayerGateState {
    /// The pilot stays preferred until warmed evidence says otherwise, and
    /// wins exact ties: it is the stronger predictor globally and the
    /// incumbent. A layer with no pilot evidence stream at all (layer 1,
    /// whose hint would have to come from the dense layer 0) belongs to the
    /// prev-token predictor as soon as that predictor is warmed.
    fn preferred(&self, warmup: u32) -> PreferredPredictor {
        let prev_token_ready = self.prev_token.warmed(warmup) && self.prev_routes.is_some();
        if !self.pilot.warmed(warmup) {
            if self.pilot.samples == 0 && prev_token_ready {
                PreferredPredictor::PrevToken
            } else {
                PreferredPredictor::Warming
            }
        } else if prev_token_ready && self.prev_token.ema > self.pilot.ema {
            PreferredPredictor::PrevToken
        } else {
            PreferredPredictor::Pilot
        }
    }
}

/// One taken-but-unscored PILOT hint, held until its target layer's
/// authoritative mailbox arrives one layer later.
#[derive(Debug, Clone, Copy)]
struct OutstandingPilot {
    target_layer: u32,
    expert_count: u8,
    expert_ids: [u16; PILOT_MAX_PREFETCH],
}

/// Run-level counters surfaced by the end-of-run report.
#[derive(Debug, Clone, Copy, Default)]
pub struct PilotGateTelemetry {
    pub passes_scored: u64,
    pub experts_issued: u64,
    pub experts_suppressed: u64,
    pub prev_token_plans: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PilotGateLayerReport {
    pub layer_index: u32,
    pub pilot_ema: f32,
    pub pilot_samples: u32,
    pub prev_token_ema: f32,
    pub prev_token_samples: u32,
    /// Measure-only: hit rate of pilot ∪ previous-token. See LayerGateState.
    pub union_ema: f32,
    pub union_samples: u32,
    pub preferred: PreferredPredictor,
    pub reads_suppressed: bool,
}

#[derive(Debug, Clone)]
pub struct PilotGateReport {
    pub measure_only: bool,
    pub threshold: f32,
    pub warmup: u32,
    pub telemetry: PilotGateTelemetry,
    /// Layers with at least one scored sample, ascending.
    pub layers: Vec<PilotGateLayerReport>,
}

#[derive(Debug)]
pub(crate) struct PilotGate {
    measure_only: bool,
    threshold: f32,
    warmup: u32,
    layers: Box<[LayerGateState]>,
    outstanding_pilot: Option<OutstandingPilot>,
    telemetry: PilotGateTelemetry,
}

impl PilotGate {
    /// `Off` deliberately constructs nothing so the legacy scheduler runs
    /// byte-identically with no governor in the loop.
    pub(crate) fn new(mode: PilotGateRequest, threshold: f64, warmup: u32) -> Option<Self> {
        let measure_only = match mode {
            PilotGateRequest::Off => return None,
            PilotGateRequest::Measure => true,
            PilotGateRequest::On => false,
        };
        Some(Self {
            measure_only,
            threshold: threshold as f32,
            warmup: warmup.max(1),
            layers: vec![LayerGateState::default(); K3_LAYER_COUNT].into_boxed_slice(),
            outstanding_pilot: None,
            telemetry: PilotGateTelemetry::default(),
        })
    }

    /// Score the outstanding prediction(s) for this layer against its
    /// authoritative routes and reseed the previous-token predictor. Each
    /// pass contributes one sample per predictor: the mean over rows of
    /// per-row top-16 recall, so prefill chunks, speculative verify rows, and
    /// single-row decode produce comparable numbers.
    pub(crate) fn observe_routes<'a, I>(&mut self, layer_index: u32, rows: I)
    where
        I: IntoIterator<Item = &'a [u16; ROUTE_TOP_K]>,
    {
        // A stale outstanding prediction whose target never arrived (sequence
        // teardown, restart) is dropped unscored either way.
        let outstanding = self
            .outstanding_pilot
            .take()
            .filter(|pilot| pilot.target_layer == layer_index);
        let Some(state) = self.layers.get_mut(layer_index as usize) else {
            return;
        };
        let pilot_set = outstanding
            .as_ref()
            .map(|pilot| expert_bitset(&pilot.expert_ids[..pilot.expert_count as usize]));
        let prev_set = state.prev_routes.as_ref().map(|routes| expert_bitset(routes));
        // Union is only meaningful when BOTH predictors have something to say
        // this pass; scoring it against a half-populated union would flatter it.
        let union_set = match (&pilot_set, &prev_set) {
            (Some(pilot), Some(prev)) => {
                let mut merged = *pilot;
                for (word, prev_word) in merged.iter_mut().zip(prev.iter()) {
                    *word |= *prev_word;
                }
                Some(merged)
            }
            _ => None,
        };
        let mut row_count = 0_u32;
        let mut pilot_hits = 0_u32;
        let mut prev_hits = 0_u32;
        let mut union_hits = 0_u32;
        let mut newest_row = None;
        for row in rows {
            row_count = row_count.saturating_add(1);
            if let Some(set) = &pilot_set {
                pilot_hits += count_hits(set, row);
            }
            if let Some(set) = &prev_set {
                prev_hits += count_hits(set, row);
            }
            if let Some(set) = &union_set {
                union_hits += count_hits(set, row);
            }
            newest_row = Some(*row);
        }
        let Some(newest_row) = newest_row else {
            return;
        };
        let slots = (row_count * ROUTE_TOP_K as u32) as f32;
        if pilot_set.is_some() {
            state.pilot.absorb(pilot_hits as f32 / slots);
        }
        if prev_set.is_some() {
            state.prev_token.absorb(prev_hits as f32 / slots);
        }
        if union_set.is_some() {
            state.union.absorb(union_hits as f32 / slots);
        }
        let mut routes = newest_row;
        routes.sort_unstable();
        state.prev_routes = Some(routes);
        self.telemetry.passes_scored = self.telemetry.passes_scored.saturating_add(1);
    }

    /// Decide this pass's speculative reads for `source_layer + 1`. The PILOT
    /// hint (when present) is always recorded for scoring first — suppressed
    /// layers keep accumulating evidence, which is the recovery path — and
    /// only then does the trailing recall pick a predictor and admit or
    /// suppress its reads.
    pub(crate) fn admit(
        &mut self,
        source_layer: u32,
        pilot: Option<ExpertPrefetchPlan>,
    ) -> Option<ExpertPrefetchPlan> {
        let target_layer = source_layer.checked_add(1)?;
        let pilot = pilot.filter(|plan| plan.target_layer() == target_layer);
        if let Some(plan) = &pilot {
            self.outstanding_pilot = Some(OutstandingPilot {
                target_layer,
                expert_count: plan.expert_count,
                expert_ids: plan.expert_ids,
            });
        }
        let state = self.layers.get(target_layer as usize).copied()?;
        if self.measure_only || !state.pilot.warmed(self.warmup) {
            // Legacy behavior while measuring or while evidence accrues: the
            // provider's own prediction, ungated.
            return self.issue(target_layer, pilot, false);
        }
        let prev_plan = state
            .prev_token
            .warmed(self.warmup)
            .then_some(state.prev_routes)
            .flatten()
            .and_then(|routes| ExpertPrefetchPlan::new(target_layer, &routes));
        let (winner_ema, from_prev_token) = match (&pilot, &prev_plan) {
            (Some(_), Some(_))
                if state.preferred(self.warmup) == PreferredPredictor::PrevToken =>
            {
                (state.prev_token.ema, true)
            }
            (Some(_), _) => (state.pilot.ema, false),
            (None, Some(_)) => (state.prev_token.ema, true),
            (None, None) => return None,
        };
        let suppress = winner_ema < self.threshold;
        self.note_gate_state(target_layer, suppress);
        if suppress {
            let plan = if from_prev_token { prev_plan } else { pilot };
            let withheld = plan.map_or(0, |plan| plan.expert_ids().len() as u64);
            self.telemetry.experts_suppressed =
                self.telemetry.experts_suppressed.saturating_add(withheld);
            return None;
        }
        self.issue(
            target_layer,
            if from_prev_token { prev_plan } else { pilot },
            from_prev_token,
        )
    }

    /// The first routed layer is structurally invisible to PILOT: its hint
    /// would have to come from dense layer 0, which produces no mailbox, so
    /// layer 1's experts are demand-read cold on every pass. The prev-token
    /// predictor has no such constraint — offer its plan at sequence start,
    /// under the same warmup and threshold discipline as every other layer.
    /// Measure mode declines: it must preserve the legacy read schedule
    /// exactly to stay a clean A/B baseline.
    pub(crate) fn plan_sequence_start(&mut self) -> Option<ExpertPrefetchPlan> {
        const FIRST_ROUTED_LAYER: u32 = 1;
        if self.measure_only {
            return None;
        }
        let state = self
            .layers
            .get(FIRST_ROUTED_LAYER as usize)
            .copied()
            .filter(|state| state.prev_token.warmed(self.warmup))?;
        let plan = state
            .prev_routes
            .and_then(|routes| ExpertPrefetchPlan::new(FIRST_ROUTED_LAYER, &routes))?;
        let suppress = state.prev_token.ema < self.threshold;
        self.note_gate_state(FIRST_ROUTED_LAYER, suppress);
        if suppress {
            self.telemetry.experts_suppressed = self
                .telemetry
                .experts_suppressed
                .saturating_add(plan.expert_ids().len() as u64);
            return None;
        }
        self.issue(FIRST_ROUTED_LAYER, Some(plan), true)
    }

    pub(crate) fn report(&self) -> PilotGateReport {
        PilotGateReport {
            measure_only: self.measure_only,
            threshold: self.threshold,
            warmup: self.warmup,
            telemetry: self.telemetry,
            layers: self
                .layers
                .iter()
                .enumerate()
                .filter(|(_, state)| state.pilot.samples != 0 || state.prev_token.samples != 0)
                .map(|(layer_index, state)| PilotGateLayerReport {
                    layer_index: layer_index as u32,
                    pilot_ema: state.pilot.ema,
                    pilot_samples: state.pilot.samples,
                    prev_token_ema: state.prev_token.ema,
                    prev_token_samples: state.prev_token.samples,
                    union_ema: state.union.ema,
                    union_samples: state.union.samples,
                    preferred: state.preferred(self.warmup),
                    reads_suppressed: state.reads_suppressed,
                })
                .collect(),
        }
    }

    fn issue(
        &mut self,
        target_layer: u32,
        plan: Option<ExpertPrefetchPlan>,
        from_prev_token: bool,
    ) -> Option<ExpertPrefetchPlan> {
        let plan = plan.filter(|plan| plan.target_layer() == target_layer)?;
        self.telemetry.experts_issued = self
            .telemetry
            .experts_issued
            .saturating_add(plan.expert_ids().len() as u64);
        if from_prev_token {
            self.telemetry.prev_token_plans = self.telemetry.prev_token_plans.saturating_add(1);
        }
        Some(plan)
    }

    fn note_gate_state(&mut self, target_layer: u32, suppress: bool) {
        let Some(state) = self.layers.get_mut(target_layer as usize) else {
            return;
        };
        if state.reads_suppressed == suppress {
            return;
        }
        state.reads_suppressed = suppress;
        if state.flip_logs >= FLIP_LOG_CAP {
            return;
        }
        state.flip_logs += 1;
        let silenced = if state.flip_logs == FLIP_LOG_CAP {
            "; further flips silenced"
        } else {
            ""
        };
        eprintln!(
            "[pilot-gate] layer {target_layer}: speculative reads {} (pilot {:.1}%, prev-token {:.1}%, {} samples{silenced})",
            if suppress { "suppressed" } else { "resumed" },
            state.pilot.ema * 100.0,
            state.prev_token.ema * 100.0,
            state.pilot.samples,
        );
    }
}

fn expert_bitset(expert_ids: &[u16]) -> [u64; EXPERT_BITSET_WORDS] {
    let mut set = [0_u64; EXPERT_BITSET_WORDS];
    for &expert in expert_ids {
        if (expert as usize) < K3_ROUTED_EXPERTS {
            set[expert as usize / 64] |= 1 << (expert as usize % 64);
        }
    }
    set
}

fn count_hits(set: &[u64; EXPERT_BITSET_WORDS], row: &[u16; ROUTE_TOP_K]) -> u32 {
    row.iter()
        .filter(|&&expert| {
            (expert as usize) < K3_ROUTED_EXPERTS
                && set[expert as usize / 64] & (1 << (expert as usize % 64)) != 0
        })
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(threshold: f64, warmup: u32) -> PilotGate {
        PilotGate::new(PilotGateRequest::On, threshold, warmup).expect("gate is enabled")
    }

    fn ascending_row(first: u16) -> [u16; ROUTE_TOP_K] {
        let mut row = [0_u16; ROUTE_TOP_K];
        for (index, expert) in row.iter_mut().enumerate() {
            *expert = first + index as u16;
        }
        row
    }

    fn plan(target_layer: u32, first: u16) -> ExpertPrefetchPlan {
        ExpertPrefetchPlan::new(target_layer, &ascending_row(first)).expect("canonical plan")
    }

    /// Simulate one pass at target layer `target`, in engine order: the PILOT
    /// hint for `target` is taken (and its reads admitted or suppressed) at
    /// layer `target - 1`'s final tile, then `target`'s authoritative mailbox
    /// arrives and scores it. The returned decision therefore reflects the
    /// trailing recall of *previous* passes, exactly as in the hot path.
    fn pass_for_target(
        gate: &mut PilotGate,
        target: u32,
        pilot_first: u16,
        actual_first: u16,
    ) -> Option<ExpertPrefetchPlan> {
        let decision = gate.admit(target - 1, Some(plan(target, pilot_first)));
        gate.observe_routes(target, [&ascending_row(actual_first)]);
        decision
    }

    #[test]
    fn off_mode_constructs_no_governor() {
        assert!(PilotGate::new(PilotGateRequest::Off, 0.10, 16).is_none());
        assert!(PilotGate::new(PilotGateRequest::Measure, 0.10, 16).is_some());
        assert!(PilotGate::new(PilotGateRequest::On, 0.10, 16).is_some());
    }

    #[test]
    fn plans_admit_only_canonical_ascending_expert_sets() {
        let ids = ascending_row(10);
        assert!(ExpertPrefetchPlan::new(5, &ids).is_some());
        assert!(ExpertPrefetchPlan::new(0, &ids).is_none());
        assert!(ExpertPrefetchPlan::new(K3_LAYER_COUNT as u32, &ids).is_none());
        assert!(ExpertPrefetchPlan::new(5, &ids[..ROUTE_TOP_K - 1]).is_none());
        let mut repeated = ids;
        repeated[1] = repeated[0];
        assert!(ExpertPrefetchPlan::new(5, &repeated).is_none());
        let mut descending = ids;
        descending.reverse();
        assert!(ExpertPrefetchPlan::new(5, &descending).is_none());
        let mut out_of_range = ids;
        out_of_range[ROUTE_TOP_K - 1] = 896;
        assert!(ExpertPrefetchPlan::new(5, &out_of_range).is_none());
        let wide: Vec<u16> = (0..PILOT_MAX_PREFETCH as u16 + 1).collect();
        assert!(ExpertPrefetchPlan::new(5, &wide[..PILOT_MAX_PREFETCH]).is_some());
        assert!(ExpertPrefetchPlan::new(5, &wide).is_none());
    }

    #[test]
    fn warmup_preserves_legacy_pilot_scheduling() {
        let mut gate = gate(0.10, 4);
        // Predictions never overlap reality, but until warmup completes every
        // pilot plan is still issued unchanged.
        for _ in 0..4 {
            let issued =
                pass_for_target(&mut gate, 6, 500, 100).expect("legacy plan during warmup");
            assert_eq!(issued.expert_ids(), &ascending_row(500));
        }
        assert_eq!(gate.telemetry.experts_suppressed, 0);
    }

    #[test]
    fn collapsed_pilot_recall_suppresses_reads_and_recovers() {
        let mut gate = gate(0.10, 3);
        // Warm up with zero-recall pilot predictions and unstable routes (the
        // prev-token predictor scores ~0 too): the layer must go dark.
        let mut actual = 100;
        for _ in 0..3 {
            pass_for_target(&mut gate, 6, 700, actual);
            actual += 40;
        }
        assert!(pass_for_target(&mut gate, 6, 700, actual).is_none());
        assert!(gate.telemetry.experts_suppressed > 0);
        let suppressed = gate.report();
        assert_eq!(suppressed.layers.len(), 1);
        let layer = suppressed.layers[0];
        assert!(layer.layer_index == 6 && layer.reads_suppressed && layer.pilot_ema < 0.10);

        // Recovery: the pilot keeps being scored while suppressed, so
        // accurate predictions climb the EMA back over the threshold and the
        // very next admission resumes reads.
        assert!(pass_for_target(&mut gate, 6, 700, 700).is_none());
        let resumed = pass_for_target(&mut gate, 6, 700, 700).expect("reads resumed");
        assert_eq!(resumed.expert_ids(), &ascending_row(700));
        assert!(!gate.report().layers[0].reads_suppressed);
    }

    #[test]
    fn prev_token_predictor_takes_over_a_bad_pilot_layer() {
        let mut gate = gate(0.10, 3);
        // Routing is perfectly stable across tokens while the pilot guesses a
        // disjoint set every pass: prev-token recall 1.0 vs pilot 0.0. The
        // prev-token predictor warms one pass behind the pilot (its first
        // mailbox only seeds it), then takes the layer over.
        for _ in 0..4 {
            pass_for_target(&mut gate, 6, 700, 100);
        }
        let issued = pass_for_target(&mut gate, 6, 700, 100).expect("fallback plan");
        assert_eq!(issued.expert_ids(), &ascending_row(100));
        assert_eq!(issued.target_layer(), 6);
        assert!(gate.telemetry.prev_token_plans > 0);
    }

    #[test]
    fn measure_mode_scores_without_ever_suppressing() {
        let mut gate =
            PilotGate::new(PilotGateRequest::Measure, 0.10, 2).expect("measure gate");
        for _ in 0..6 {
            let issued =
                pass_for_target(&mut gate, 6, 700, 100).expect("measure never suppresses");
            assert_eq!(issued.expert_ids(), &ascending_row(700));
        }
        assert_eq!(gate.telemetry.experts_suppressed, 0);
        let report = gate.report();
        assert!(report.measure_only);
        // Both predictors were still scored the whole time.
        let layer = report
            .layers
            .iter()
            .find(|layer| layer.layer_index == 6)
            .expect("scored layer");
        assert!(layer.pilot_samples >= 4 && layer.prev_token_samples >= 4);
        assert!(layer.pilot_ema < 0.10 && layer.prev_token_ema > 0.90);
    }

    #[test]
    fn outstanding_predictions_score_once_and_drop_on_mismatch() {
        let mut gate = gate(0.10, 2);
        gate.admit(5, Some(plan(6, 100)));
        gate.observe_routes(6, [&ascending_row(100)]);
        let scored = gate.report();
        let layer = &scored.layers[0];
        assert_eq!((layer.layer_index, layer.pilot_samples), (6, 1));
        assert_eq!(layer.pilot_ema, 1.0);

        // Without a fresh admission the same mailbox contributes no second
        // pilot sample, and a hint whose target never arrives is dropped.
        gate.observe_routes(6, [&ascending_row(100)]);
        gate.admit(5, Some(plan(6, 100)));
        gate.observe_routes(1, [&ascending_row(100)]);
        gate.observe_routes(6, [&ascending_row(100)]);
        let report = gate.report();
        let layer = report
            .layers
            .iter()
            .find(|layer| layer.layer_index == 6)
            .expect("scored layer");
        assert_eq!(layer.pilot_samples, 1);
        // The dropped layer-6 hint must not have been scored against layer
        // 1's routes: a mid-pass teardown may never contaminate another
        // layer's EMA, so layer 6 stays the only scored layer.
        assert!(report.layers.iter().all(|layer| layer.layer_index == 6));
    }

    #[test]
    fn pilot_hint_miss_falls_back_to_the_warmed_prev_token_predictor() {
        let mut gate = gate(0.10, 2);
        // Warm both predictors on a stable layer where the pilot is useless:
        // stable routes give prev-token recall 1.0, disjoint hints give the
        // pilot 0.0.
        for _ in 0..3 {
            pass_for_target(&mut gate, 6, 700, 100);
        }
        // A provider fail-soft miss (no hint this pass) must not turn a
        // warmed layer dark: the prev-token predictor still issues its plan.
        let fallback = gate.admit(5, None).expect("prev-token fallback on a hint miss");
        assert_eq!(fallback.expert_ids(), &ascending_row(100));
        assert_eq!(fallback.target_layer(), 6);
    }

    #[test]
    fn prev_token_predictor_seeds_from_the_newest_row() {
        let mut gate = gate(0.10, 1);
        // A multi-row pass must seed the predictor from its newest row (400s,
        // the latest position), not its first (100s).
        gate.admit(5, Some(plan(6, 700)));
        gate.observe_routes(6, [&ascending_row(100), &ascending_row(400)]);
        // Score the seeded routes against a pass that repeats the newest row:
        // recall is 1.0 only under newest-row seeding.
        assert!(gate.admit(5, None).is_none());
        gate.observe_routes(6, [&ascending_row(400)]);
        let issued = gate.admit(5, None).expect("newest-row prev-token plan");
        assert_eq!(issued.expert_ids(), &ascending_row(400));
        let report = gate.report();
        assert!(report.layers[0].prev_token_ema > 0.90);
    }

    #[test]
    fn exact_ties_prefer_the_pilot_predictor() {
        let mut gate = gate(0.10, 1);
        // A 32-expert pilot union covering the actual 16 scores exactly 1.0,
        // and so does the stable prev-token predictor: both EMAs sit at the
        // 1.0 fixed point, and the documented tie-break must keep the pilot's
        // wider union rather than swap in the 16-expert prev-token plan.
        let wide: Vec<u16> = (100..100 + PILOT_MAX_PREFETCH as u16).collect();
        let wide_plan = || ExpertPrefetchPlan::new(6, &wide).expect("canonical wide plan");
        for _ in 0..2 {
            gate.admit(5, Some(wide_plan()));
            gate.observe_routes(6, [&ascending_row(100)]);
        }
        let report = gate.report();
        assert_eq!(report.layers[0].pilot_ema, 1.0);
        assert_eq!(report.layers[0].prev_token_ema, 1.0);
        assert_eq!(report.layers[0].preferred, PreferredPredictor::Pilot);
        let issued = gate.admit(5, Some(wide_plan())).expect("tie keeps the pilot");
        assert_eq!(issued.expert_ids(), &wide[..]);
    }

    #[test]
    fn per_row_scoring_matches_single_row_decode_recall() {
        let mut gate = gate(0.10, 32);
        // 8 of 16 predicted experts land: one decode pass must score 0.5.
        gate.admit(5, Some(plan(6, 100)));
        gate.observe_routes(6, [&ascending_row(108)]);
        assert_eq!(gate.report().layers[0].pilot_ema, 0.5);

        // Multi-row passes average per-row recall over rows: a second pass
        // with rows at recall 1.0 and 0.0 blends 0.5 into the EMA unchanged.
        gate.admit(5, Some(plan(6, 108)));
        gate.observe_routes(6, [&ascending_row(108), &ascending_row(400)]);
        assert_eq!(gate.report().layers[0].pilot_ema, 0.5);
        assert_eq!(gate.report().layers[0].pilot_samples, 2);
    }

    #[test]
    fn sequence_start_covers_layer_one_with_the_warmed_prev_token_predictor() {
        let mut gate = gate(0.10, 2);
        // Layer 1 never receives a pilot hint (its source would be the dense
        // layer 0), so admission at sequence start waits on the prev-token
        // predictor alone: seed, warm, then serve.
        assert!(gate.plan_sequence_start().is_none());
        gate.observe_routes(1, [&ascending_row(100)]);
        assert!(gate.plan_sequence_start().is_none());
        gate.observe_routes(1, [&ascending_row(100)]);
        assert!(gate.plan_sequence_start().is_none());
        gate.observe_routes(1, [&ascending_row(100)]);
        let plan = gate.plan_sequence_start().expect("warmed layer-1 plan");
        assert_eq!(plan.target_layer(), 1);
        assert_eq!(plan.expert_ids(), &ascending_row(100));
        assert!(gate.telemetry.prev_token_plans > 0);
        let report = gate.report();
        assert_eq!(report.layers[0].preferred, PreferredPredictor::PrevToken);
    }

    #[test]
    fn sequence_start_respects_threshold_and_measure_mode() {
        // Unstable layer-1 routing scores 0.0 and stays suppressed.
        let mut gate = gate(0.5, 1);
        gate.observe_routes(1, [&ascending_row(100)]);
        gate.observe_routes(1, [&ascending_row(400)]);
        assert!(gate.plan_sequence_start().is_none());
        assert!(gate.telemetry.experts_suppressed >= ROUTE_TOP_K as u64);
        assert!(gate.report().layers[0].reads_suppressed);

        // Measure mode must keep the legacy read schedule byte-identical, so
        // it never issues the extra layer-1 prefetch even when warmed.
        let mut measure =
            PilotGate::new(PilotGateRequest::Measure, 0.10, 1).expect("measure gate");
        measure.observe_routes(1, [&ascending_row(100)]);
        measure.observe_routes(1, [&ascending_row(100)]);
        assert!(measure.plan_sequence_start().is_none());
        assert_eq!(measure.telemetry.experts_suppressed, 0);
    }

    #[test]
    fn admissions_at_the_layer_roster_edge_are_skipped() {
        let mut gate = gate(0.10, 1);
        // Layer 92 is the final routed layer; its lookahead target does not
        // exist and admission must decline without panicking.
        assert!(gate.admit(92, None).is_none());
        assert!(gate.admit(u32::MAX, None).is_none());
        gate.observe_routes(u32::MAX - 1, [&ascending_row(0)]);
        assert!(gate.report().layers.is_empty());
    }
}
