//! K3_ROUTE_ORACLE=<jsonl> + K3_ORACLE_DEPTH=N — the lead-depth oracle.
//!
//! Replays a RouterTrace recording (K3_TRACE_PATH JSONL: {step, layer, ids})
//! of a DETERMINISTIC rung as perfect-accuracy prefetch hints submitted N
//! extra layer-walls ahead of the pilot's one-wall lead. This prices the
//! lead-depth program (2026-08-25 second opinion Q3) with zero predictor
//! work: whatever depth-N shows is the ceiling any real predictor could
//! reach. Scheduling-only by construction — recorded routes equal the
//! authoritative routes (tripwire-counted), speculative hits are validated
//! per expert, and any failure falls back to the byte-identical demand
//! union, so token output cannot change. Default-off.
//!
//! Semantics: at layer L's mailbox-open the engine submits the recorded
//! routes of layer L+1+N, so a target layer's tickets are in flight for
//! N+1 layer-walls before consumption (pilot baseline: 1 wall).
//! Steps whose records are not exactly one 16-expert row (prefill, ngram
//! verify passes) are skipped whole — oracle inactive for those passes.
//!
//! Requirements: K3_PILOT_EARLY=1 and the prefetch reader (otherwise the
//! injection block never runs and the arm reports submitted=0 = invalid);
//! exactly ONE generation per process (the pass ordinal is process-global
//! and must stay aligned with the recording's step numbering). Arms with
//! dropped_full>0, pilot_suppressed>0, or route_mismatches>0 are invalid.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct RouteOracle {
    routes: HashMap<(u64, u32), Vec<u16>>,
    skip_steps: HashSet<u64>,
    depth: u32,
}

impl RouteOracle {
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Sorted unique recorded expert ids for (pass step, layer); None when
    /// the step is skipped (multi-row pass) or the pair was never recorded.
    pub fn routes_for(&self, step: u64, layer: u32) -> Option<&[u16]> {
        if self.skip_steps.contains(&step) {
            return None;
        }
        self.routes.get(&(step, layer)).map(Vec::as_slice)
    }
}

fn load(path: &str, depth: u32) -> Option<RouteOracle> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| eprintln!("[oracle] cannot read {path}: {error}"))
        .ok()?;
    let mut routes: HashMap<(u64, u32), Vec<u16>> = HashMap::new();
    let mut skip_steps = HashSet::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(step), Some(layer)) = (value["step"].as_u64(), value["layer"].as_u64()) else {
            continue;
        };
        let layer = layer as u32;
        let Some(ids) = value["ids"].as_array() else {
            continue;
        };
        // Only clean decode passes qualify: exactly one 16-expert row per
        // (step, layer), ids valid and unique after sorting. Anything else
        // (prefill rows, verify multi-row mailboxes, duplicate records)
        // poisons the whole step.
        let mut experts: Vec<u16> = ids
            .iter()
            .filter_map(|id| id.as_u64())
            .filter(|&id| id < 896)
            .map(|id| id as u16)
            .collect();
        experts.sort_unstable();
        experts.dedup();
        if ids.len() != 16 || experts.len() != 16 || routes.contains_key(&(step, layer)) {
            skip_steps.insert(step);
            routes.remove(&(step, layer));
            continue;
        }
        routes.insert((step, layer), experts);
    }
    if routes.is_empty() {
        eprintln!("[oracle] recording {path} contained no usable decode passes");
        return None;
    }
    eprintln!(
        "[oracle] loaded {} (step,layer) routes, {} skipped steps, depth={}",
        routes.len(),
        skip_steps.len(),
        depth,
    );
    Some(RouteOracle {
        routes,
        skip_steps,
        depth,
    })
}

pub fn route_oracle() -> Option<&'static RouteOracle> {
    static ORACLE: OnceLock<Option<RouteOracle>> = OnceLock::new();
    ORACLE
        .get_or_init(|| {
            let path = std::env::var("K3_ROUTE_ORACLE").ok()?;
            let depth = std::env::var("K3_ORACLE_DEPTH")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .map_or(1, |value| value.clamp(0, 6));
            load(&path, depth)
        })
        .as_ref()
}

static SUBMITTED_SETS: AtomicU64 = AtomicU64::new(0);
static DROPPED_SETS: AtomicU64 = AtomicU64::new(0);
static MERGED_TICKETS: AtomicU64 = AtomicU64::new(0);
static DUPLICATE_TICKETS: AtomicU64 = AtomicU64::new(0);
static ROUTE_MISMATCHES: AtomicU64 = AtomicU64::new(0);
static STALE_SETS: AtomicU64 = AtomicU64::new(0);
static PILOT_SUPPRESSED: AtomicU64 = AtomicU64::new(0);

pub fn note_submitted() {
    SUBMITTED_SETS.fetch_add(1, Ordering::Relaxed);
}
pub fn note_dropped() {
    DROPPED_SETS.fetch_add(1, Ordering::Relaxed);
}
pub fn note_merged(tickets: u64) {
    MERGED_TICKETS.fetch_add(tickets, Ordering::Relaxed);
}
pub fn note_duplicates(tickets: u64) {
    DUPLICATE_TICKETS.fetch_add(tickets, Ordering::Relaxed);
}
pub fn note_mismatch() {
    ROUTE_MISMATCHES.fetch_add(1, Ordering::Relaxed);
}
pub fn note_stale() {
    STALE_SETS.fetch_add(1, Ordering::Relaxed);
}
pub fn note_pilot_suppressed() {
    PILOT_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
}

/// One summary line for print_run_stats. mismatches!=0 or submitted==0
/// invalidates a ladder arm (misalignment / silent no-op — the two
/// pre-registered failure modes).
pub fn oracle_report() -> Option<String> {
    route_oracle()?;
    Some(format!(
        "[oracle] depth={} sets: submitted={} dropped_full={} stale={} | tickets merged={} duplicate={} | route_mismatches={} pilot_suppressed={}",
        route_oracle().map_or(0, RouteOracle::depth),
        SUBMITTED_SETS.load(Ordering::Relaxed),
        DROPPED_SETS.load(Ordering::Relaxed),
        STALE_SETS.load(Ordering::Relaxed),
        MERGED_TICKETS.load(Ordering::Relaxed),
        DUPLICATE_TICKETS.load(Ordering::Relaxed),
        ROUTE_MISMATCHES.load(Ordering::Relaxed),
        PILOT_SUPPRESSED.load(Ordering::Relaxed),
    ))
}
