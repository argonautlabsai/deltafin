//! Best-effort foreground disk-I/O policy for latency-sensitive model reads.
//!
//! The proven Python reader applies Darwin's process-wide `IOPOL_IMPORTANT`
//! policy before inference and marks every read worker as
//! `QOS_CLASS_USER_INITIATED`. Native `Reader` threads do the same here. The
//! policy changes scheduling only: read contents, ordering, and validation are
//! unaffected. Unsupported hosts deliberately compile to no-ops.

use std::ffi::OsStr;

const IOPOL_ENV: &str = "K3_SPINE_IOPOL";

/// Native inference defaults to the foreground policy on macOS. An explicit
/// `K3_SPINE_IOPOL=0` is the benchmark/debug escape hatch; `=1` forces it on.
/// Unknown explicit values fail closed rather than silently changing policy.
fn resolve_enabled(value: Option<&OsStr>, is_macos: bool) -> bool {
    if !is_macos {
        return false;
    }
    match value {
        None => true,
        Some(value) if value == OsStr::new("1") => true,
        Some(_) => false,
    }
}

fn enabled() -> bool {
    resolve_enabled(
        std::env::var_os(IOPOL_ENV).as_deref(),
        cfg!(target_os = "macos"),
    )
}

/// Apply the process-wide disk-I/O policy before persistent reader workers are
/// created. Darwin may reject this request; matching the Python path, policy
/// setup is best-effort and must never make inference fail.
pub(crate) fn configure_process_for_model_io() {
    if enabled() {
        platform::set_process_io_important();
    }
}

/// Apply latency-sensitive QoS once, at the start of each persistent reader
/// worker. This is intentionally not applied to compute/provider threads.
pub(crate) fn configure_model_io_thread() {
    if enabled() {
        platform::set_reader_thread_user_initiated();
    }
}

const PREFETCH_IOPOL_ENV: &str = "K3_PREFETCH_IOPOL";

/// K3_PREFETCH_IOPOL=utility|throttle — deprioritize PREFETCH reader threads
/// at the kernel's per-device I/O queues, so demand reads are served ahead of
/// speculation on every drive. Motivation (2026-09-02): suppression frees
/// prefetch capacity that elastically backfills with deeper speculation
/// (PS200: 288 GB suppressed, net −102 GB; S1: 98 GB served, K3C −19 GB),
/// and the champion's own barrier signature — Tmax/Tmedian = 2.02, top half
/// of every barrier slow together — is plausibly demand-vs-prefetch queue
/// contention. `utility` is the measured default candidate; `throttle` is
/// Darwin's background class and may starve prefetch outright — bracket it,
/// never assume it. Unset or any other value: no-op (champion unchanged).
fn prefetch_policy(value: Option<&OsStr>) -> Option<PrefetchPolicy> {
    match value {
        Some(value) if value == OsStr::new("utility") => Some(PrefetchPolicy::Utility),
        Some(value) if value == OsStr::new("throttle") => Some(PrefetchPolicy::Throttle),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefetchPolicy {
    Utility,
    Throttle,
}

/// Worker start hook for PREFETCH readers. With K3_PREFETCH_IOPOL set, the
/// thread gets a deprioritized disk policy + matching QoS; otherwise it takes
/// the ordinary foreground path, byte-identical to before this knob existed.
pub(crate) fn configure_prefetch_io_thread() {
    match prefetch_policy(std::env::var_os(PREFETCH_IOPOL_ENV).as_deref()) {
        Some(PrefetchPolicy::Utility) if cfg!(target_os = "macos") => {
            platform::set_prefetch_thread_utility();
        }
        Some(PrefetchPolicy::Throttle) if cfg!(target_os = "macos") => {
            platform::set_prefetch_thread_throttled();
        }
        _ => configure_model_io_thread(),
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use libc::qos_class_t;

    // Values from macOS <sys/resource.h>. IOPOL_IMPORTANT is the foreground
    // disk policy (also exposed by the compatibility name IOPOL_NORMAL).
    const IOPOL_TYPE_DISK: libc::c_int = 0;
    const IOPOL_SCOPE_PROCESS: libc::c_int = 0;
    const IOPOL_IMPORTANT: libc::c_int = 1;

    unsafe extern "C" {
        fn setiopolicy_np(
            policy_type: libc::c_int,
            scope: libc::c_int,
            policy: libc::c_int,
        ) -> libc::c_int;
    }

    pub(super) fn set_process_io_important() {
        // SAFETY: setiopolicy_np takes three scalar values. These constants are
        // the public Darwin values and PROCESS scope needs no pointer argument.
        let _ = unsafe { setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_IMPORTANT) };
    }

    pub(super) fn set_reader_thread_user_initiated() {
        // SAFETY: this changes only the calling thread's QoS and takes no
        // borrowed pointers. Zero is the documented relative priority.
        let _ = unsafe {
            libc::pthread_set_qos_class_self_np(qos_class_t::QOS_CLASS_USER_INITIATED, 0)
        };
    }

    // Thread-scope disk policies for prefetch deprioritization. Values from
    // <sys/resource.h>: SCOPE_THREAD=1, IOPOL_THROTTLE=3, IOPOL_UTILITY=4.
    const IOPOL_SCOPE_THREAD: libc::c_int = 1;
    const IOPOL_THROTTLE: libc::c_int = 3;
    const IOPOL_UTILITY: libc::c_int = 4;

    pub(super) fn set_prefetch_thread_utility() {
        // SAFETY: scalar args only; THREAD scope affects the calling thread.
        let _ = unsafe { setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_UTILITY) };
        let _ = unsafe { libc::pthread_set_qos_class_self_np(qos_class_t::QOS_CLASS_UTILITY, 0) };
    }

    pub(super) fn set_prefetch_thread_throttled() {
        // SAFETY: as above. THROTTLE is Darwin's background class and can be
        // aggressive on external buses — bracket before trusting.
        let _ = unsafe { setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_THROTTLE) };
        let _ = unsafe { libc::pthread_set_qos_class_self_np(qos_class_t::QOS_CLASS_BACKGROUND, 0) };
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn darwin_policy_constants_match_public_headers() {
            assert_eq!(IOPOL_TYPE_DISK, 0);
            assert_eq!(IOPOL_SCOPE_PROCESS, 0);
            assert_eq!(IOPOL_IMPORTANT, 1);
            assert_eq!(qos_class_t::QOS_CLASS_USER_INITIATED as u32, 0x19);
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    pub(super) fn set_process_io_important() {}

    pub(super) fn set_reader_thread_user_initiated() {}

    pub(super) fn set_prefetch_thread_utility() {}

    pub(super) fn set_prefetch_thread_throttled() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_toggle_has_explicit_off_and_on_with_macos_auto_default() {
        assert!(resolve_enabled(None, true));
        assert!(!resolve_enabled(Some(OsStr::new("0")), true));
        assert!(resolve_enabled(Some(OsStr::new("1")), true));
        assert!(!resolve_enabled(Some(OsStr::new("invalid")), true));
    }

    #[test]
    fn policy_is_always_disabled_on_other_hosts() {
        assert!(!resolve_enabled(None, false));
        assert!(!resolve_enabled(Some(OsStr::new("1")), false));
    }
}
