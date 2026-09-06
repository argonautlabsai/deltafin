//! Persistent, Python-free positional I/O for streamed model weights.
//!
//! A `ReadPlan` opens every immutable source once, validates complete source
//! and destination coverage, and splits large extents before the hot path. A
//! `Reader` owns fixed worker threads and a bounded reusable buffer arena.
//! Submitted reads return tickets immediately after admission; workers pull
//! short quanta from demand/prefetch queues instead of constructing a Future,
//! dictionary, or memoryview per chunk.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::any::Any;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::FileExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::error::{DeltafinError, Result};
use crate::packfile::DigestState;

#[cfg(target_os = "macos")]
pub(crate) const BUFFER_ALIGNMENT: usize = 16 * 1024;
#[cfg(not(target_os = "macos"))]
// Linux/aarch64 deployments can use 64 KiB base pages. A conservative 64 KiB
// alignment also remains valid on 4 KiB x86-64 hosts and avoids rebuilding
// the arena when direct/pinned provider uploads are enabled later.
pub(crate) const BUFFER_ALIGNMENT: usize = 64 * 1024;
const MAX_PLAN_JOBS: usize = 1_000_000;
const DEFAULT_ARENA_SLOTS: usize = 2;
const WORK_QUANTUM: usize = 4;
const MAX_DEMAND_STREAK: usize = 8;
// Keep storage from consuming the process-wide descriptor table. Packed
// layers retain their descriptors across reads, but only inside this budget.
// A complete loose K3 int8 spine has several thousand immutable components.
// The high ceiling is not an allocation target: every admission is still
// proven against the live process soft limit and leaves explicit headroom.
const MAX_PERSISTENT_STORAGE_DESCRIPTORS: usize = 8_192;
const MIN_DESCRIPTOR_HEADROOM: usize = 32;
const MAX_DESCRIPTOR_HEADROOM: usize = 1_024;
const FALLBACK_SOFT_DESCRIPTOR_LIMIT: usize = 256;
pub const LOOSE_SPINE_DESCRIPTOR_RESERVE: usize = 1_024;
// Routed decode always selects exactly 16 experts. Keeping those whole-file
// jobs inside the already-required Batch allocation removes a second jobs
// allocation and its Arc without inflating every generic spine batch.
const MAX_INLINE_DEFERRED_FILES: usize = 16;
// A loose K3 layer currently has at most 42 source components. Keep modest
// format headroom, while proving a batch can never recreate the old unbounded
// descriptor cache. With the default two arena slots, at most 128 manifest
// descriptors can coexist even if both reads reach their widest point.
const MAX_DEFERRED_MANIFEST_SOURCES: usize = 64;
// The exact full-commit Scale4 verifier is bounded to nine rows × 16 routed
// experts. Its widest union therefore names 144 raw files plus one shared
// layer sidecar and at most two authenticated ranges per expert. Keep exact
// headroom for that compile-time ceiling without admitting unbounded
// path/range metadata. The independent loose-spine manifest limit above stays
// at 64 sources.
const MAX_DEFERRED_AUTHENTICATED_SOURCES: usize = 145;
const MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS: usize = 288;
// Keep vectored positional reads bounded on the stack and comfortably below
// the minimum POSIX IOV_MAX. Scale4 uses four destinations: its header and
// three exact compressed-scale planes.
const MAX_VECTORED_DESTINATIONS: usize = 16;
// K3's longest canonical expert name is 12 bytes. Thirty-one leaves ample
// format headroom while keeping all 82,432 catalog entries near 2.7 MiB rather
// than doubling that cold but permanently resident metadata.
const MAX_DEFERRED_SOURCE_NAME_BYTES: usize = 31;

#[derive(Debug)]
struct DescriptorBudget {
    capacity: usize,
    in_use: Mutex<usize>,
    soft_limit: usize,
    observed_open: Option<usize>,
    headroom: usize,
}

impl DescriptorBudget {
    fn for_process() -> Arc<Self> {
        let soft_limit = soft_descriptor_limit().unwrap_or(FALLBACK_SOFT_DESCRIPTOR_LIMIT);
        let observed_open = count_open_descriptors();
        let proportional_headroom = (soft_limit / 8).clamp(
            MIN_DESCRIPTOR_HEADROOM.min(soft_limit),
            MAX_DESCRIPTOR_HEADROOM.min(soft_limit),
        );
        let (capacity, headroom) = if let Some(open) = observed_open {
            (
                soft_limit
                    .saturating_sub(open)
                    .saturating_sub(proportional_headroom)
                    .min(MAX_PERSISTENT_STORAGE_DESCRIPTORS),
                proportional_headroom,
            )
        } else {
            // Both supported hosts expose /dev/fd or /proc/self/fd. If that
            // inspection is unavailable, consume at most one quarter of the
            // soft limit and leave the remainder to the runtime and libraries.
            let capacity = (soft_limit / 4)
                .min(MAX_PERSISTENT_STORAGE_DESCRIPTORS)
                .min(64);
            (capacity, soft_limit.saturating_sub(capacity))
        };
        Arc::new(Self {
            capacity,
            in_use: Mutex::new(0),
            soft_limit,
            observed_open,
            headroom,
        })
    }

    #[cfg(test)]
    fn fixed(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            in_use: Mutex::new(0),
            soft_limit: capacity,
            observed_open: Some(0),
            headroom: 0,
        })
    }

    fn reserve(self: &Arc<Self>, count: usize) -> Result<DescriptorReservation> {
        let mut in_use = self.in_use.lock().unwrap();
        let available = self.capacity.saturating_sub(*in_use);
        if count > available {
            return Err(DeltafinError::new(format!(
                "read plan needs {count} persistent source descriptors, but only {available} remain in Deltafin's {}-descriptor storage budget (soft limit {}, observed open {}, reserved headroom {}); use packed layer files, release inactive read plans, or raise the descriptor limit",
                self.capacity,
                self.soft_limit,
                self.observed_open
                    .map_or_else(|| "unknown".to_owned(), |value| value.to_string()),
                self.headroom,
            )));
        }
        *in_use += count;
        drop(in_use);
        Ok(DescriptorReservation {
            budget: Arc::clone(self),
            count,
        })
    }
}

#[derive(Debug)]
struct DescriptorReservation {
    budget: Arc<DescriptorBudget>,
    count: usize,
}

impl Drop for DescriptorReservation {
    fn drop(&mut self) {
        let mut in_use = self.budget.in_use.lock().unwrap();
        debug_assert!(*in_use >= self.count);
        *in_use -= self.count;
    }
}

fn process_descriptor_budget() -> Arc<DescriptorBudget> {
    static BUDGET: OnceLock<Arc<DescriptorBudget>> = OnceLock::new();
    Arc::clone(BUDGET.get_or_init(DescriptorBudget::for_process))
}

/// Prove that an all-or-nothing persistent loose-source roster can coexist
/// with the process and provider descriptor working set.
///
/// The soft limit is raised only for this process, only when the hard limit
/// already permits it, and before the singleton storage budget is frozen.
/// Failure never admits a partial cache; automatic callers can fall back to
/// ordinary per-batch descriptors, while explicit callers surface the error.
pub fn prepare_persistent_descriptor_capacity(required: usize, reserve: usize) -> Result<()> {
    if required == 0 {
        return Ok(());
    }
    if required > MAX_PERSISTENT_STORAGE_DESCRIPTORS {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine roster needs {required} descriptors, exceeding Deltafin's {MAX_PERSISTENT_STORAGE_DESCRIPTORS}-descriptor safety ceiling"
        )));
    }
    let open = count_open_descriptors().ok_or_else(|| {
        DeltafinError::new(
            "cannot inspect the live process descriptor count; persistent loose-spine cache is not safe",
        )
    })?;
    let needed_soft = open
        .checked_add(required)
        .and_then(|value| value.checked_add(reserve))
        .ok_or_else(|| DeltafinError::new("persistent descriptor requirement overflows usize"))?;
    let (mut soft, hard) = descriptor_limits().ok_or_else(|| {
        DeltafinError::new(
            "cannot inspect the process descriptor limits; persistent loose-spine cache is not safe",
        )
    })?;
    if soft < needed_soft {
        let target = needed_soft
            .checked_next_power_of_two()
            .unwrap_or(needed_soft)
            .max(4_096)
            .min(hard);
        if target < needed_soft || !set_soft_descriptor_limit(target) {
            return Err(DeltafinError::new(format!(
                "persistent loose-spine cache needs {required} descriptors plus {reserve} reserved (currently {open} open), but the process soft/hard limits are {soft}/{hard}"
            )));
        }
        soft = descriptor_limits().map_or(soft, |limits| limits.0);
    }
    if soft.saturating_sub(open).saturating_sub(reserve) < required {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine cache needs {required} descriptors plus {reserve} reserved (currently {open} open), but only {soft} are admitted"
        )));
    }
    let budget = process_descriptor_budget();
    let available = budget
        .capacity
        .saturating_sub(*budget.in_use.lock().unwrap());
    if available < required {
        return Err(DeltafinError::new(format!(
            "persistent loose-spine cache needs {required} descriptors, but the initialized storage budget has only {available} available"
        )));
    }
    Ok(())
}

#[repr(C)]
struct NativeRlimit {
    current: u64,
    maximum: u64,
}

fn soft_descriptor_limit() -> Option<usize> {
    descriptor_limits().map(|limits| limits.0)
}

fn descriptor_limits() -> Option<(usize, usize)> {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return None;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    unsafe extern "C" {
        fn getrlimit(resource: i32, limits: *mut NativeRlimit) -> i32;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut limits = NativeRlimit {
            current: 0,
            maximum: 0,
        };
        // SAFETY: `limits` points to writable storage with the platform's
        // two-rlim_t layout; Linux and Darwin use 64-bit rlim_t on supported
        // x86-64/aarch64 targets.
        if unsafe { getrlimit(RLIMIT_NOFILE, &mut limits) } != 0 {
            return None;
        }
        Some((
            usize::try_from(limits.current).ok()?,
            usize::try_from(limits.maximum).ok()?,
        ))
    }
}

fn set_soft_descriptor_limit(soft: usize) -> bool {
    #[cfg(target_os = "linux")]
    const RLIMIT_NOFILE: i32 = 7;
    #[cfg(target_os = "macos")]
    const RLIMIT_NOFILE: i32 = 8;
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    return false;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    unsafe extern "C" {
        fn setrlimit(resource: i32, limits: *const NativeRlimit) -> i32;
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let Some((_, hard)) = descriptor_limits() else {
            return false;
        };
        if soft > hard {
            return false;
        }
        let limits = NativeRlimit {
            current: soft as u64,
            maximum: hard as u64,
        };
        // SAFETY: `limits` has the supported platform's two-rlim_t layout and
        // remains live for the duration of this process-local syscall.
        unsafe { setrlimit(RLIMIT_NOFILE, &limits) == 0 }
    }
}

fn count_open_descriptors() -> Option<usize> {
    #[cfg(target_os = "linux")]
    const FD_DIRECTORIES: [&str; 2] = ["/proc/self/fd", "/dev/fd"];
    #[cfg(not(target_os = "linux"))]
    const FD_DIRECTORIES: [&str; 2] = ["/dev/fd", "/proc/self/fd"];

    for directory in FD_DIRECTORIES {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        let mut count = 0_usize;
        let mut complete = true;
        for entry in entries {
            if entry.is_err() {
                complete = false;
                break;
            }
            let Some(next) = count.checked_add(1) else {
                complete = false;
                break;
            };
            count = next;
        }
        if complete {
            return Some(count);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BufferKind {
    Quantized,
    Scales,
    Other,
}

impl BufferKind {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Quantized, Self::Scales, Self::Other];

    const fn index(self) -> usize {
        match self {
            Self::Quantized => 0,
            Self::Scales => 1,
            Self::Other => 2,
        }
    }
}

/// Exact logical sizes of the three packed destination buffers.
///
/// These are supplied by the audited tensor manifest rather than inferred
/// from the last extent. That makes a missing trailing extent an error instead
/// of silently producing a shorter tensor.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct BufferLengths {
    pub quantized: usize,
    pub scales: usize,
    pub other: usize,
}

impl BufferLengths {
    pub const fn new(quantized: usize, scales: usize, other: usize) -> Self {
        Self {
            quantized,
            scales,
            other,
        }
    }

    const fn get(self, kind: BufferKind) -> usize {
        match kind {
            BufferKind::Quantized => self.quantized,
            BufferKind::Scales => self.scales,
            BufferKind::Other => self.other,
        }
    }

    const fn as_array(self) -> [usize; BufferKind::COUNT] {
        [self.quantized, self.scales, self.other]
    }

    fn max(self, other: Self) -> Self {
        Self::new(
            self.quantized.max(other.quantized),
            self.scales.max(other.scales),
            self.other.max(other.other),
        )
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CachePolicy {
    Resident,
    Streaming,
}

/// One direct child of a deferred-read directory, stored inline and already
/// NUL terminated for `openat(2)`. Catalog construction is cold-path work;
/// selecting a source later copies only its integer index and never creates a
/// `PathBuf`, `String`, `CString`, or hash-table entry.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct DeferredSourceName {
    bytes: [u8; MAX_DEFERRED_SOURCE_NAME_BYTES + 1],
    len: u8,
}

impl DeferredSourceName {
    pub fn new(name: &str) -> Result<Self> {
        let source = name.as_bytes();
        if source.is_empty() {
            return Err(DeltafinError::new("a deferred source name cannot be empty"));
        }
        if source.len() > MAX_DEFERRED_SOURCE_NAME_BYTES {
            return Err(DeltafinError::new(format!(
                "deferred source name is {} bytes; maximum is {MAX_DEFERRED_SOURCE_NAME_BYTES}",
                source.len()
            )));
        }
        if source == b"." || source == b".." || source.contains(&b'/') || source.contains(&0) {
            return Err(DeltafinError::new(
                "deferred source must be one non-special directory entry",
            ));
        }
        let mut bytes = [0_u8; MAX_DEFERRED_SOURCE_NAME_BYTES + 1];
        bytes[..source.len()].copy_from_slice(source);
        Ok(Self {
            bytes,
            len: source.len() as u8,
        })
    }

    fn as_c_str(&self) -> &CStr {
        // Construction rejects interior NULs and the fixed trailing byte is
        // zero, so this prefix is always exactly one valid C string.
        CStr::from_bytes_with_nul(&self.bytes[..usize::from(self.len) + 1])
            .expect("validated deferred source name must remain NUL terminated")
    }

    pub(crate) fn as_str(&self) -> &str {
        // Construction accepts UTF-8 `str` input and never mutates the prefix.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("validated deferred source name must remain UTF-8")
    }
}

impl std::fmt::Debug for DeferredSourceName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("DeferredSourceName")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Debug)]
struct DeferredExactCatalogInner {
    directory: File,
    directory_path: PathBuf,
    // Optional fast tier holding byte-identical copies of a subset of the
    // sources. Probed first by openat; ENOENT falls back to `directory`.
    // Wrong-size or non-regular hot entries fail closed like primary ones.
    hot_directory: Option<File>,
    hot_directory_path: Option<PathBuf>,
    /// Optional second canonical expert volume (K3_EXPERT_DIR_B): probed
    /// after the hot tier and before the primary directory. Part of the
    /// canonical corpus in the de-striped two-volume layout.
    secondary_directory: Option<File>,
    secondary_directory_path: Option<PathBuf>,
    /// Optional third canonical expert volume (K3_EXPERT_DIR_C): probed
    /// BEFORE the hot tier, matching `resolve_expert_path`'s deliberate
    /// dir_c-first order so a copy placed here takes the read without any
    /// file having to be deleted from the tier it duplicates.
    ///
    /// Added 2026-08-29. Until now dir_c reached only the wide-union path
    /// (`open_raw_cache_with_cache_policy`); the top-k decode path submits
    /// from this catalog, which knew about hot and dir_b only. K3C was
    /// therefore structurally unreachable on ~80% of reads: measured 4.1% of
    /// bytes against 24.4% of routed edges, with the deficit landing on the
    /// two volumes that cover the namespace by index parity.
    tertiary_directory: Option<File>,
    tertiary_directory_path: Option<PathBuf>,
    sources: Box<[DeferredSourceName]>,
    exact_source_length: u64,
    cache_policy: CachePolicy,
}

/// Immutable directory-relative source catalog for repeated whole-file reads.
///
/// The directory itself is opened once with `O_NOFOLLOW`. Individual files
/// remain ephemeral: workers use `openat` with `O_NOFOLLOW|O_CLOEXEC`, validate
/// regular-file type and exact size on that live descriptor, read it, and
/// close it. This keeps routed experts inside a bounded FD budget while moving
/// path assembly, hashing, source deduplication, and plan validation entirely
/// out of the per-layer path.
#[derive(Clone, Debug)]
pub struct DeferredExactCatalog {
    inner: Arc<DeferredExactCatalogInner>,
}

impl DeferredExactCatalog {
    pub fn open(
        directory: &Path,
        sources: impl IntoIterator<Item = DeferredSourceName>,
        exact_source_length: u64,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_tiered(
            directory,
            None,
            None,
            None,
            sources,
            exact_source_length,
            cache_policy,
        )
    }

    pub fn open_tiered(
        directory: &Path,
        hot_directory: Option<&Path>,
        secondary_directory: Option<&Path>,
        tertiary_directory: Option<&Path>,
        sources: impl IntoIterator<Item = DeferredSourceName>,
        exact_source_length: u64,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        if exact_source_length == 0 {
            return Err(DeltafinError::new(
                "a deferred source catalog needs a positive exact length",
            ));
        }
        let sources: Vec<_> = sources.into_iter().collect();
        if sources.is_empty() {
            return Err(DeltafinError::new(
                "a deferred source catalog needs at least one source",
            ));
        }
        let directory_file = OpenOptions::new()
            .read(true)
            .custom_flags(open_cloexec_nofollow())
            .open(directory)
            .map_err(|error| {
                io_error(
                    "open deferred source directory without following symlinks",
                    directory,
                    error,
                )
            })?;
        let metadata = directory_file
            .metadata()
            .map_err(|error| io_error("stat deferred source directory", directory, error))?;
        if !metadata.is_dir() {
            return Err(DeltafinError::new(format!(
                "deferred source catalog root is not a directory: {}",
                directory.display()
            )));
        }
        let hot_directory_file = hot_directory
            .map(|hot| -> Result<File> {
                let file = OpenOptions::new()
                    .read(true)
                    .custom_flags(open_cloexec_nofollow())
                    .open(hot)
                    .map_err(|error| {
                        io_error(
                            "open deferred hot-tier directory without following symlinks",
                            hot,
                            error,
                        )
                    })?;
                let metadata = file
                    .metadata()
                    .map_err(|error| io_error("stat deferred hot-tier directory", hot, error))?;
                if !metadata.is_dir() {
                    return Err(DeltafinError::new(format!(
                        "deferred hot-tier catalog root is not a directory: {}",
                        hot.display()
                    )));
                }
                Ok(file)
            })
            .transpose()?;
        let secondary_directory_file = secondary_directory
            .map(|secondary| -> Result<File> {
                let file = OpenOptions::new()
                    .read(true)
                    .custom_flags(open_cloexec_nofollow())
                    .open(secondary)
                    .map_err(|error| {
                        io_error(
                            "open deferred secondary directory without following symlinks",
                            secondary,
                            error,
                        )
                    })?;
                let metadata = file.metadata().map_err(|error| {
                    io_error("stat deferred secondary directory", secondary, error)
                })?;
                if !metadata.is_dir() {
                    return Err(DeltafinError::new(format!(
                        "deferred secondary catalog root is not a directory: {}",
                        secondary.display()
                    )));
                }
                Ok(file)
            })
            .transpose()?;
        let tertiary_directory_file = tertiary_directory
            .map(|tertiary| -> Result<File> {
                let file = OpenOptions::new()
                    .read(true)
                    .custom_flags(open_cloexec_nofollow())
                    .open(tertiary)
                    .map_err(|error| {
                        io_error(
                            "open deferred tertiary directory without following symlinks",
                            tertiary,
                            error,
                        )
                    })?;
                let metadata = file.metadata().map_err(|error| {
                    io_error("stat deferred tertiary directory", tertiary, error)
                })?;
                if !metadata.is_dir() {
                    return Err(DeltafinError::new(format!(
                        "deferred tertiary catalog root is not a directory: {}",
                        tertiary.display()
                    )));
                }
                Ok(file)
            })
            .transpose()?;
        Ok(Self {
            inner: Arc::new(DeferredExactCatalogInner {
                directory: directory_file,
                directory_path: directory.to_path_buf(),
                hot_directory: hot_directory_file,
                hot_directory_path: hot_directory.map(Path::to_path_buf),
                secondary_directory: secondary_directory_file,
                secondary_directory_path: secondary_directory.map(Path::to_path_buf),
                tertiary_directory: tertiary_directory_file,
                tertiary_directory_path: tertiary_directory.map(Path::to_path_buf),
                sources: sources.into_boxed_slice(),
                exact_source_length,
                cache_policy,
            }),
        })
    }

    pub fn source_count(&self) -> usize {
        self.inner.sources.len()
    }

    pub fn exact_source_length(&self) -> u64 {
        self.inner.exact_source_length
    }

    #[cfg(test)]
    fn source_name(&self, index: usize) -> Option<&str> {
        self.inner
            .sources
            .get(index)
            .map(DeferredSourceName::as_str)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Extent {
    Read {
        path: PathBuf,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
        /// Optional digest of this complete extent. Verified extents are never
        /// split into smaller jobs, and their successful first-read result is
        /// cached inside the immutable `ReadPlan` that owns the opened file.
        expected_digest: Option<[u8; 32]>,
    },
    /// One contiguous source range scattered, in source order, into several
    /// disjoint destination ranges by a single positional vectored read.
    ///
    /// This is deliberately narrower than an arbitrary gather: source bytes
    /// may not contain gaps or change order. That exact contract maps to
    /// `preadv(2)` on both macOS and Linux without temporary buffers or an
    /// additional copy.
    ReadVectored {
        path: PathBuf,
        source_offset: u64,
        destinations: Box<[VectoredDestination]>,
        /// Optional digest over the contiguous source range, reconstructed by
        /// hashing the ordered destination vectors after `preadv` completes.
        expected_digest: Option<[u8; 32]>,
    },
    /// An explicit zero/padding range. Implicit holes are rejected.
    Zero {
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
    },
}

impl Extent {
    pub fn new(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
    ) -> Self {
        Self::Read {
            path: path.into(),
            source_offset,
            destination,
            destination_offset,
            length,
            expected_digest: None,
        }
    }

    /// Describe one independently authenticated file range. This is used by
    /// DFSP pack chunks: hashing the bytes already in the destination slab
    /// avoids a second disk pass, while the verification bit remains tied to
    /// this read plan's exact, already-open descriptor generation.
    pub fn verified(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destination: BufferKind,
        destination_offset: usize,
        length: usize,
        expected_digest: [u8; 32],
    ) -> Self {
        Self::Read {
            path: path.into(),
            source_offset,
            destination,
            destination_offset,
            length,
            expected_digest: Some(expected_digest),
        }
    }

    /// Describe a contiguous source range whose bytes are scattered into
    /// disjoint destination ranges in the supplied order.
    pub fn vectored(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
    ) -> Result<Self> {
        Self::vectored_with_digest(path, source_offset, destinations, None)
    }

    /// Describe a vectored positional read whose complete contiguous source
    /// range must match `expected_digest` before its private batch can publish.
    pub fn vectored_verified(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
        expected_digest: [u8; 32],
    ) -> Result<Self> {
        Self::vectored_with_digest(path, source_offset, destinations, Some(expected_digest))
    }

    fn vectored_with_digest(
        path: impl Into<PathBuf>,
        source_offset: u64,
        destinations: impl IntoIterator<Item = VectoredDestination>,
        expected_digest: Option<[u8; 32]>,
    ) -> Result<Self> {
        let destinations: Vec<_> = destinations.into_iter().collect();
        if destinations.is_empty() {
            return Err(DeltafinError::new(
                "a vectored read needs at least one destination",
            ));
        }
        if destinations.len() > MAX_VECTORED_DESTINATIONS {
            return Err(DeltafinError::new(format!(
                "a vectored read has {} destinations; bounded maximum is {MAX_VECTORED_DESTINATIONS}",
                destinations.len()
            )));
        }
        let mut total = 0_usize;
        for destination in &destinations {
            if destination.length == 0 {
                return Err(DeltafinError::new(
                    "a vectored read destination may not be empty",
                ));
            }
            total = total
                .checked_add(destination.length)
                .ok_or_else(|| DeltafinError::new("vectored read length overflows usize"))?;
        }
        source_offset
            .checked_add(total as u64)
            .ok_or_else(|| DeltafinError::new("vectored source range overflows u64"))?;
        Ok(Self::ReadVectored {
            path: path.into(),
            source_offset,
            destinations: destinations.into_boxed_slice(),
            expected_digest,
        })
    }

    pub const fn zero(destination: BufferKind, destination_offset: usize, length: usize) -> Self {
        Self::Zero {
            destination,
            destination_offset,
            length,
        }
    }
}

/// One output range in a contiguous positional vectored read.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VectoredDestination {
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
}

impl VectoredDestination {
    pub const fn new(destination: BufferKind, destination_offset: usize, length: usize) -> Self {
        Self {
            destination,
            destination_offset,
            length,
        }
    }
}

/// One exact source range that must authenticate before any gathered bytes
/// from the same live descriptor may be published.
///
/// Unlike [`Extent::verified`], this digest is over source bytes rather than
/// one contiguous destination extent. It therefore supports strict gathers
/// where an authenticated source record is scattered into disjoint output
/// ranges. The source is opened lazily by a reader worker with `O_NOFOLLOW`.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeferredSourceVerification {
    path: PathBuf,
    exact_source_length: u64,
    source_offset: u64,
    length: usize,
    expected_digest: [u8; 32],
}

/// Exact live-descriptor length for a deferred partial-range source.
///
/// This is the non-hashing counterpart to [`DeferredSourceVerification`]. It
/// exists for formats whose retained bytes are authenticated after assembly
/// (for example, a Scale4 sidecar record) while their canonical raw packed
/// planes use the same exact-length/no-follow trust boundary as raw-v1 expert
/// execution. Workers still open every source with `O_NOFOLLOW|O_CLOEXEC` and
/// validate the regular file and exact length on the descriptor they read.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeferredSourceLength {
    path: PathBuf,
    exact_source_length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
}

impl DeferredSourceLength {
    pub fn new(path: impl Into<PathBuf>, exact_source_length: u64) -> Self {
        Self {
            path: path.into(),
            exact_source_length,
            expected_identity: None,
        }
    }

    /// Pin the current no-follow descriptor identity as part of this range
    /// contract. The reader rechecks that identity on the descriptor used for
    /// every positional read and once more after the complete batch finishes.
    /// This is intended for immutable files whose successful record digests may
    /// be cached by identity across batches.
    pub fn new_with_live_identity(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
    ) -> Result<Self> {
        let path = path.into();
        let expected_identity = capture_deferred_source_identity(&path, exact_source_length)?;
        Ok(Self {
            path,
            exact_source_length,
            expected_identity: Some(expected_identity),
        })
    }

    /// Reuse an identity previously captured by
    /// [`Self::new_with_live_identity`] without synchronously reopening the
    /// source. This crate-private constructor is reserved for immutable session
    /// caches: workers still open with `O_NOFOLLOW`, require the exact length,
    /// compare this identity before reading, and compare it again after the
    /// batch completes.
    pub(crate) fn new_with_captured_identity(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
        expected_identity: DeferredSourceIdentity,
    ) -> Result<Self> {
        if expected_identity.bytes != exact_source_length {
            return Err(DeltafinError::new(format!(
                "captured deferred source identity is {} bytes; expected exact length {exact_source_length}",
                expected_identity.bytes,
            )));
        }
        Ok(Self {
            path: path.into(),
            exact_source_length,
            expected_identity: Some(expected_identity),
        })
    }

    pub const fn identity(&self) -> Option<DeferredSourceIdentity> {
        self.expected_identity
    }
}

impl DeferredSourceVerification {
    pub fn new(
        path: impl Into<PathBuf>,
        exact_source_length: u64,
        source_offset: u64,
        length: usize,
        expected_digest: [u8; 32],
    ) -> Self {
        Self {
            path: path.into(),
            exact_source_length,
            source_offset,
            length,
            expected_digest,
        }
    }
}

#[derive(Debug, Clone)]
struct SourceVerification {
    source_offset: u64,
    length: usize,
    expected_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy)]
struct SourceScatter {
    source_offset: u64,
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
    verification_index: usize,
}

#[derive(Debug, Clone)]
struct AuthenticatedSourceContract {
    exact_length: u64,
    verifications: Vec<SourceVerification>,
}

#[derive(Debug, Clone, Copy)]
struct DeferredRangeContract {
    exact_length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
}

#[derive(Debug)]
struct Source {
    path: PathBuf,
    file: Option<File>,
    /// A deferred manifest can retain its first validated descriptor for the
    /// entire immutable plan lifetime. `OnceLock<Result<_>>` also pins a
    /// first-use failure, so a path replacement can never turn a rejected
    /// source into a different inode inside the published plan.
    persistent_file: Option<OnceLock<Result<File>>>,
    length: u64,
    expected_identity: Option<DeferredSourceIdentity>,
    verifications: Box<[SourceVerification]>,
    /// Disjoint authenticated source ranges copied directly into the private
    /// arena while their enclosing verification range is hashed.  Keeping
    /// these on the source lets one worker make one physical pass over every
    /// authenticated byte instead of hashing the source and then rereading
    /// the retained ranges.
    scatter_extents: Vec<SourceScatter>,
    cache_policy: CachePolicy,
}

#[derive(Debug)]
struct Sources {
    values: Vec<Source>,
    persistent_count: usize,
    // The reservation must outlive every Batch clone of this source set, not
    // merely the ReadPlan that admitted it.
    _descriptor_reservation: DescriptorReservation,
}

#[derive(Debug, Clone, Copy)]
enum JobSource {
    File {
        source: usize,
        source_offset: u64,
    },
    Vectored {
        source: usize,
        scatter: usize,
    },
    AuthenticatedScatter {
        source: usize,
    },
    DeferredCatalog {
        source: u32,
        /// Byte offset of this chunk within the source file. 0 for legacy
        /// whole-file jobs (K3_SPLIT_READ absent/1).
        source_offset: u64,
        /// K3_SPLIT_READ home routing: Some(true) reads the hot/internal
        /// copy, Some(false) reads the enclosure copy, each falling back to
        /// the other when its home lacks the file (resolution stays total).
        /// None keeps the legacy probe order (mirror scheduler or
        /// tier-first) with a fresh open per job.
        prefer_internal_home: Option<bool>,
        /// Index of this job's source within its batch (NOT the catalog
        /// index) — keys the batch's per-source home-descriptor cache so a
        /// file's chunks share two opens instead of re-probing per chunk.
        batch_slot: u16,
    },
    Zero,
}

#[derive(Debug)]
struct VectoredRead {
    source_offset: u64,
    destinations: Box<[VectoredDestination]>,
}

#[derive(Debug, Clone, Copy)]
struct ReadJob {
    source: JobSource,
    destination: BufferKind,
    destination_offset: usize,
    length: usize,
    expected_digest: Option<[u8; 32]>,
    verification_index: Option<usize>,
}

struct InlineReadJobs {
    sources: [u32; MAX_INLINE_DEFERRED_FILES],
    len: usize,
    destination: BufferKind,
    source_length: usize,
    /// K3_SPLIT_READ: jobs per source file. 1 = legacy whole-file jobs.
    chunks_per_source: usize,
    /// BUFFER_ALIGNMENT-rounded ceil(source_length / chunks_per_source);
    /// equals source_length when chunks_per_source is 1.
    chunk_length: usize,
    /// The first `internal_chunks` chunk indices of every source prefer the
    /// hot/internal home; the rest prefer the enclosures. Sized to the
    /// measured device-rate ratio (internal ~13.73 vs enclosure ~5.58 GB/s).
    internal_chunks: usize,
}

enum BatchJobs {
    Shared(Arc<Vec<ReadJob>>),
    Inline(InlineReadJobs),
}

impl BatchJobs {
    fn get(&self, index: usize) -> Option<ReadJob> {
        match self {
            Self::Shared(jobs) => jobs.get(index).copied(),
            Self::Inline(jobs) if index < jobs.len * jobs.chunks_per_source => {
                // File-major striping: indices 0..len are chunk 0 of every
                // source, len..2*len chunk 1, and so on. For multi-file
                // batches, consecutive claims by one worker quantum therefore
                // touch DIFFERENT files. Single-source tickets (len=1,
                // prefetch) degrade to consecutive chunks of one file — the
                // claim loop interleaves claims with I/O, so free workers
                // still pick up remaining chunks concurrently; the stall
                // trace shows prefetch claims are near-instant (queue 0.1%),
                // so free workers are the common case.
                let file = index % jobs.len;
                let chunk = index / jobs.len;
                let chunk_offset = chunk * jobs.chunk_length;
                let length = (jobs.source_length - chunk_offset).min(jobs.chunk_length);
                let prefer_internal_home = (jobs.chunks_per_source > 1)
                    .then(|| chunk < jobs.internal_chunks);
                Some(ReadJob {
                    source: JobSource::DeferredCatalog {
                        source: jobs.sources[file],
                        source_offset: chunk_offset as u64,
                        prefer_internal_home,
                        batch_slot: file as u16,
                    },
                    destination: jobs.destination,
                    destination_offset: file * jobs.source_length + chunk_offset,
                    length,
                    expected_digest: None,
                    verification_index: None,
                })
            }
            Self::Inline(_) => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Shared(jobs) => jobs.len(),
            Self::Inline(jobs) => jobs.len * jobs.chunks_per_source,
        }
    }
}

enum BatchSources {
    Plan {
        sources: Arc<Sources>,
        /// One live descriptor per deferred source for the lifetime of this
        /// batch. Large whole-file tensors are split into several parallel
        /// positional jobs; sharing the validated descriptor removes repeated
        /// open/stat/cache-policy calls and guarantees every chunk observes
        /// the same inode. Persistent plans leave this absent.
        deferred_files: Option<Box<[OnceLock<Result<File>>]>>,
    },
    DeferredCatalog(Arc<DeferredExactCatalogInner>),
}

#[derive(Debug)]
pub struct ReadPlan {
    sources: Arc<Sources>,
    jobs: Arc<Vec<ReadJob>>,
    vectored_reads: Arc<Vec<VectoredRead>>,
    verified_extents: Arc<Vec<AtomicBool>>,
    buffer_lengths: BufferLengths,
    logical_bytes: u64,
}

#[derive(Debug, Clone)]
enum SourceAdmission {
    Persistent,
    DeferredExact(u64),
    /// Every read extent names one complete source file.  Its declared extent
    /// length becomes that source's exact-size contract, but the file is not
    /// opened until a reader worker executes the job.  This is the loose-spine
    /// compatibility path: thousands of immutable tensor components can be
    /// compiled once without retaining thousands of descriptors or serially
    /// opening/statting them on the inference thread.
    DeferredManifest,
    /// Same exact whole-file contract as `DeferredManifest`, with a bounded
    /// all-or-nothing descriptor reservation made at plan construction. Files
    /// are still opened lazily on reader workers and validated with
    /// O_NOFOLLOW against the descriptor actually used for every pread.
    DeferredManifestPersistent,
    /// Partial ranges from heterogeneous exact-length files. Unlike the
    /// manifest variants, extents need not cover a complete source. Unlike an
    /// authenticated gather, no source bytes are reread solely to hash gaps;
    /// callers must validate their assembled format before publication.
    DeferredRanges(Arc<HashMap<PathBuf, DeferredRangeContract>>),
    /// Sources are opened on workers; every declared verification range is
    /// authenticated on its live descriptor while its disjoint retained
    /// extents are scattered into the private destination arena.
    DeferredAuthenticated(Arc<HashMap<PathBuf, AuthenticatedSourceContract>>),
}

impl ReadPlan {
    pub fn open(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::Persistent,
        )
    }

    /// Compile a plan without opening its source files on the caller thread.
    /// Each worker opens one source with `O_NOFOLLOW`, validates the exact
    /// regular-file length on that descriptor, reads it positionally, applies
    /// cache policy, and closes it. This is intended for ephemeral routed
    /// expert unions where retaining hundreds of descriptors would exceed the
    /// process budget and serial open/stat calls would extend token latency.
    pub fn open_deferred_exact(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
        exact_source_length: u64,
    ) -> Result<Self> {
        if exact_source_length == 0 {
            return Err(DeltafinError::new(
                "deferred read sources need a positive exact length",
            ));
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredExact(exact_source_length),
        )
    }

    /// Compile a heterogeneous whole-file manifest without opening sources.
    ///
    /// Unlike [`Self::open_deferred_exact`], each source may have a different
    /// exact length. Every non-empty read must cover its complete source from
    /// offset zero. Workers later open with `O_NOFOLLOW|O_CLOEXEC`, validate a
    /// live regular-file descriptor against the recorded length, read it, and
    /// close it. This makes the unpacked model layout safe under low descriptor
    /// limits while keeping file validation on the descriptor actually read.
    pub fn open_deferred_manifest(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredManifest,
        )
    }

    pub fn open_persistent_deferred_manifest(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredManifestPersistent,
        )
    }

    /// Compile a heterogeneous partial-range gather without opening sources.
    ///
    /// Every source must have exactly one explicit length contract. The
    /// worker-owned descriptor is shared by all ranges from that source in one
    /// batch, then closed after completion. This keeps hot gathers free of
    /// redundant whole-file authentication reads while retaining the same
    /// no-follow, regular-file and exact-length boundary as deferred raw-v1.
    pub fn open_deferred_ranges(
        extents: impl IntoIterator<Item = Extent>,
        sources: impl IntoIterator<Item = DeferredSourceLength>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let mut contracts = HashMap::new();
        for source in sources {
            if source.exact_source_length == 0 {
                return Err(DeltafinError::new(
                    "deferred range source needs a positive exact length",
                ));
            }
            if contracts.len() == MAX_DEFERRED_AUTHENTICATED_SOURCES
                && !contracts.contains_key(&source.path)
            {
                return Err(DeltafinError::new(format!(
                    "deferred range plan exceeds the {MAX_DEFERRED_AUTHENTICATED_SOURCES}-source safety limit"
                )));
            }
            let contract = DeferredRangeContract {
                exact_length: source.exact_source_length,
                expected_identity: source.expected_identity,
            };
            if contracts.insert(source.path, contract).is_some() {
                return Err(DeltafinError::new(
                    "deferred range source length is declared more than once",
                ));
            }
        }
        if contracts.is_empty() {
            return Err(DeltafinError::new(
                "deferred range plan needs at least one source contract",
            ));
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredRanges(Arc::new(contracts)),
        )
    }

    /// Compile a deferred, exact-length gather whose source identity is
    /// authenticated on the same live descriptor used for every copied byte.
    ///
    /// Each source named by an extent must have at least one verification.
    /// Multiple verifications may qualify disjoint records in one larger
    /// source file. Authentication is performed once per admitted batch
    /// descriptor, and an identity recheck after the one-pass authenticated
    /// scatter fails closed if the source changes while that descriptor is in
    /// use.
    pub fn open_deferred_authenticated(
        extents: impl IntoIterator<Item = Extent>,
        verifications: impl IntoIterator<Item = DeferredSourceVerification>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<Self> {
        let mut contracts: HashMap<PathBuf, AuthenticatedSourceContract> = HashMap::new();
        let mut verification_count = 0_usize;
        for verification in verifications {
            verification_count = verification_count
                .checked_add(1)
                .ok_or_else(|| DeltafinError::new("authenticated range count overflows usize"))?;
            if verification_count > MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS {
                return Err(DeltafinError::new(format!(
                    "authenticated deferred plan exceeds the {MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS}-range safety limit"
                )));
            }
            if verification.exact_source_length == 0 || verification.length == 0 {
                return Err(DeltafinError::new(
                    "authenticated deferred sources and ranges must be non-empty",
                ));
            }
            let end = verification
                .source_offset
                .checked_add(verification.length as u64)
                .ok_or_else(|| DeltafinError::new("authenticated source range overflows u64"))?;
            if end > verification.exact_source_length {
                return Err(DeltafinError::new(format!(
                    "authenticated range {}..{} exceeds exact source length {} for {}",
                    verification.source_offset,
                    end,
                    verification.exact_source_length,
                    verification.path.display(),
                )));
            }
            if !contracts.contains_key(&verification.path)
                && contracts.len() == MAX_DEFERRED_AUTHENTICATED_SOURCES
            {
                return Err(DeltafinError::new(format!(
                    "authenticated deferred plan exceeds the {MAX_DEFERRED_AUTHENTICATED_SOURCES}-source safety limit"
                )));
            }
            let contract =
                contracts
                    .entry(verification.path)
                    .or_insert_with(|| AuthenticatedSourceContract {
                        exact_length: verification.exact_source_length,
                        verifications: Vec::new(),
                    });
            if contract.exact_length != verification.exact_source_length {
                return Err(DeltafinError::new(
                    "authenticated source has conflicting exact-length contracts",
                ));
            }
            if contract.verifications.iter().any(|existing| {
                existing.source_offset == verification.source_offset
                    && existing.length == verification.length
            }) {
                return Err(DeltafinError::new(
                    "authenticated source range is declared more than once",
                ));
            }
            contract.verifications.push(SourceVerification {
                source_offset: verification.source_offset,
                length: verification.length,
                expected_digest: verification.expected_digest,
            });
        }
        if contracts.is_empty() {
            return Err(DeltafinError::new(
                "authenticated deferred read needs at least one source contract",
            ));
        }
        for contract in contracts.values_mut() {
            contract
                .verifications
                .sort_by_key(|verification| verification.source_offset);
            for adjacent in contract.verifications.windows(2) {
                let previous = &adjacent[0];
                let next = &adjacent[1];
                let previous_end = previous
                    .source_offset
                    .checked_add(previous.length as u64)
                    .ok_or_else(|| {
                        DeltafinError::new("authenticated source range overflows u64")
                    })?;
                if next.source_offset < previous_end {
                    return Err(DeltafinError::new(
                        "authenticated source ranges overlap; every gathered byte must have one unambiguous digest contract",
                    ));
                }
            }
        }
        Self::open_with_admission(
            extents,
            buffer_lengths,
            chunk_bytes,
            cache_policy,
            SourceAdmission::DeferredAuthenticated(Arc::new(contracts)),
        )
    }

    fn open_with_admission(
        extents: impl IntoIterator<Item = Extent>,
        buffer_lengths: BufferLengths,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
        admission: SourceAdmission,
    ) -> Result<Self> {
        let extents: Vec<Extent> = extents.into_iter().collect();
        validate_destinations(&extents, buffer_lengths)?;

        let unique_source_count = extents
            .iter()
            .filter_map(|extent| match extent {
                Extent::Read { path, length, .. } if *length != 0 => Some(path),
                Extent::ReadVectored {
                    path, destinations, ..
                } if !destinations.is_empty() => Some(path),
                _ => None,
            })
            .collect::<HashSet<_>>()
            .len();
        if matches!(
            &admission,
            SourceAdmission::DeferredManifest | SourceAdmission::DeferredManifestPersistent
        ) && unique_source_count > MAX_DEFERRED_MANIFEST_SOURCES
        {
            return Err(DeltafinError::new(format!(
                "deferred manifest has {unique_source_count} sources; bounded maximum is {MAX_DEFERRED_MANIFEST_SOURCES}"
            )));
        }
        let persistent_count = match &admission {
            SourceAdmission::Persistent | SourceAdmission::DeferredManifestPersistent => {
                unique_source_count
            }
            SourceAdmission::DeferredExact(_)
            | SourceAdmission::DeferredManifest
            | SourceAdmission::DeferredRanges(_)
            | SourceAdmission::DeferredAuthenticated(_) => 0,
        };
        let descriptor_reservation = process_descriptor_budget().reserve(persistent_count)?;

        let mut sources = Vec::with_capacity(unique_source_count);
        let mut source_indices = HashMap::with_capacity(unique_source_count);
        let mut jobs = Vec::new();
        let mut vectored_reads = Vec::new();
        let mut logical_bytes = 0_u64;

        for extent in extents {
            if let Extent::ReadVectored {
                path,
                source_offset,
                destinations,
                expected_digest,
            } = extent
            {
                if destinations.is_empty()
                    || destinations
                        .iter()
                        .any(|destination| destination.length == 0)
                {
                    return Err(DeltafinError::new(
                        "a vectored read needs only non-empty destinations",
                    ));
                }
                let SourceAdmission::DeferredRanges(contracts) = &admission else {
                    return Err(DeltafinError::new(
                        "vectored reads require deferred-range source contracts",
                    ));
                };
                let extent_length =
                    destinations
                        .iter()
                        .try_fold(0_usize, |total, destination| {
                            total.checked_add(destination.length).ok_or_else(|| {
                                DeltafinError::new("vectored read length overflows usize")
                            })
                        })?;
                if extent_length == 0 {
                    continue;
                }
                let source_end = source_offset
                    .checked_add(extent_length as u64)
                    .ok_or_else(|| DeltafinError::new("vectored source range overflows u64"))?;
                let contract = contracts.get(&path).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "deferred range source {} has no exact-length contract",
                        path.display()
                    ))
                })?;
                if source_end > contract.exact_length {
                    return Err(DeltafinError::new(format!(
                        "vectored source extent {}..{} exceeds {}-byte file {}",
                        source_offset,
                        source_end,
                        contract.exact_length,
                        path.display()
                    )));
                }
                logical_bytes = logical_bytes
                    .checked_add(extent_length as u64)
                    .ok_or_else(|| DeltafinError::new("read-plan byte count overflows u64"))?;
                let source = if let Some(source) = source_indices.get(&path) {
                    *source
                } else {
                    let index = sources.len();
                    sources.push(Source {
                        path: path.clone(),
                        file: None,
                        persistent_file: None,
                        length: contract.exact_length,
                        expected_identity: contract.expected_identity,
                        verifications: Vec::new().into_boxed_slice(),
                        scatter_extents: Vec::new(),
                        cache_policy,
                    });
                    source_indices.insert(path, index);
                    index
                };
                let first = *destinations
                    .first()
                    .expect("vectored constructor and plan validation reject empty targets");
                let scatter = vectored_reads.len();
                vectored_reads.push(VectoredRead {
                    source_offset,
                    destinations,
                });
                jobs.push(ReadJob {
                    source: JobSource::Vectored { source, scatter },
                    destination: first.destination,
                    destination_offset: first.destination_offset,
                    length: extent_length,
                    expected_digest,
                    verification_index: None,
                });
                if jobs.len() > MAX_PLAN_JOBS {
                    return Err(DeltafinError::new(format!(
                        "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit"
                    )));
                }
                continue;
            }
            if matches!(
                &extent,
                Extent::Read { length: 0, .. } | Extent::Zero { length: 0, .. }
            ) {
                continue;
            }
            let (source, destination, destination_offset, extent_length, expected_digest) =
                match extent {
                    Extent::Read {
                        path,
                        source_offset,
                        destination,
                        destination_offset,
                        length,
                        expected_digest,
                    } => {
                        let source_end = source_offset
                            .checked_add(length as u64)
                            .ok_or_else(|| DeltafinError::new("source extent overflows u64"))?;
                        if let SourceAdmission::DeferredAuthenticated(contracts) = &admission {
                            if expected_digest.is_some() {
                                return Err(DeltafinError::new(
                                    "authenticated deferred gathers use source-range digests and may not also declare a per-extent digest",
                                ));
                            }
                            let contract = contracts.get(&path).ok_or_else(|| {
                                DeltafinError::new(format!(
                                    "deferred gather source {} has no authentication contract",
                                    path.display()
                                ))
                            })?;
                            let covered = contract.verifications.iter().any(|verification| {
                                let verification_end =
                                    verification.source_offset + verification.length as u64;
                                verification.source_offset <= source_offset
                                    && source_end <= verification_end
                            });
                            if !covered {
                                return Err(DeltafinError::new(format!(
                                    "deferred gather extent {}..{} from {} is outside every authenticated range",
                                    source_offset,
                                    source_end,
                                    path.display()
                                )));
                            }
                        }
                        if let SourceAdmission::DeferredRanges(contracts) = &admission
                            && !contracts.contains_key(&path)
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred range source {} has no exact-length contract",
                                path.display()
                            )));
                        }
                        if matches!(
                            &admission,
                            SourceAdmission::DeferredManifest
                                | SourceAdmission::DeferredManifestPersistent
                        ) && source_offset != 0
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred-manifest source {} must be read whole from offset zero",
                                path.display()
                            )));
                        }
                        logical_bytes =
                            logical_bytes.checked_add(length as u64).ok_or_else(|| {
                                DeltafinError::new("read-plan byte count overflows u64")
                            })?;
                        let source = if let Some(source) = source_indices.get(&path) {
                            *source
                        } else {
                            let (file, file_size, expected_identity, verifications) =
                                match &admission {
                                    SourceAdmission::Persistent => {
                                        let file = File::open(&path).map_err(|error| {
                                    if matches!(error.raw_os_error(), Some(23 | 24)) {
                                        DeltafinError::new(format!(
                                            "descriptor headroom was consumed by another subsystem while opening {}; close unrelated files or raise the descriptor limit",
                                            path.display()
                                        ))
                                    } else {
                                        io_error("open", &path, error)
                                    }
                                })?;
                                        let file_size = file
                                            .metadata()
                                            .map_err(|error| io_error("stat", &path, error))?
                                            .len();
                                        configure_cache_policy(&file, &path, cache_policy)?;
                                        (Some(file), file_size, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredExact(length) => {
                                        (None, *length, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredManifest => {
                                        (None, source_end, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredManifestPersistent => {
                                        (None, source_end, None, Vec::new().into_boxed_slice())
                                    }
                                    SourceAdmission::DeferredRanges(contracts) => {
                                        let contract = contracts.get(&path).ok_or_else(|| {
                                        DeltafinError::new(format!(
                                            "deferred range source {} has no exact-length contract",
                                            path.display()
                                        ))
                                    })?;
                                        (
                                            None,
                                            contract.exact_length,
                                            contract.expected_identity,
                                            Vec::new().into_boxed_slice(),
                                        )
                                    }
                                    SourceAdmission::DeferredAuthenticated(contracts) => {
                                        let contract = contracts.get(&path).ok_or_else(|| {
                                        DeltafinError::new(format!(
                                            "deferred gather source {} has no authentication contract",
                                            path.display()
                                        ))
                                    })?;
                                        (
                                            None,
                                            contract.exact_length,
                                            None,
                                            contract.verifications.clone().into_boxed_slice(),
                                        )
                                    }
                                };
                            let index = sources.len();
                            sources.push(Source {
                                path: path.clone(),
                                file,
                                persistent_file: matches!(
                                    &admission,
                                    SourceAdmission::DeferredManifestPersistent
                                )
                                .then(OnceLock::new),
                                length: file_size,
                                expected_identity,
                                verifications,
                                scatter_extents: Vec::new(),
                                cache_policy,
                            });
                            source_indices.insert(path.clone(), index);
                            index
                        };
                        let file_size = sources[source].length;
                        if matches!(
                            &admission,
                            SourceAdmission::DeferredManifest
                                | SourceAdmission::DeferredManifestPersistent
                        ) && file_size != source_end
                        {
                            return Err(DeltafinError::new(format!(
                                "deferred-manifest source {} has conflicting whole-file lengths {} and {}",
                                path.display(),
                                file_size,
                                source_end
                            )));
                        }
                        if source_end > file_size {
                            return Err(DeltafinError::new(format!(
                                "source extent {}..{} exceeds {}-byte file {}",
                                source_offset,
                                source_end,
                                file_size,
                                path.display()
                            )));
                        }
                        if matches!(&admission, SourceAdmission::DeferredAuthenticated(_)) {
                            let verification_index = sources[source]
                                .verifications
                                .iter()
                                .position(|verification| {
                                    let verification_end = verification.source_offset
                                        + verification.length as u64;
                                    verification.source_offset <= source_offset
                                        && source_end <= verification_end
                                })
                                .ok_or_else(|| {
                                    DeltafinError::new(format!(
                                        "deferred gather extent {}..{} from {} lost its authentication contract",
                                        source_offset,
                                        source_end,
                                        path.display()
                                    ))
                                })?;
                            sources[source].scatter_extents.push(SourceScatter {
                                source_offset,
                                destination,
                                destination_offset,
                                length,
                                verification_index,
                            });
                            // Authenticated gathers become one source-owned
                            // job below.  Emitting ordinary extent jobs here
                            // would authenticate the source and then reread
                            // every retained byte a second time.
                            continue;
                        }
                        (
                            JobSource::File {
                                source,
                                source_offset,
                            },
                            destination,
                            destination_offset,
                            length,
                            expected_digest,
                        )
                    }
                    Extent::Zero {
                        destination,
                        destination_offset,
                        length,
                    } => (
                        JobSource::Zero,
                        destination,
                        destination_offset,
                        length,
                        None,
                    ),
                    Extent::ReadVectored { .. } => {
                        unreachable!("vectored extents are compiled before ordinary read jobs")
                    }
                };

            let chunk = if expected_digest.is_some() || chunk_bytes == 0 {
                extent_length
            } else {
                chunk_bytes
            };
            let mut consumed = 0_usize;
            while consumed < extent_length {
                let length = chunk.min(extent_length - consumed);
                let source = match source {
                    JobSource::File {
                        source,
                        source_offset,
                    } => JobSource::File {
                        source,
                        source_offset: source_offset + consumed as u64,
                    },
                    JobSource::Vectored { source, scatter } => {
                        JobSource::Vectored { source, scatter }
                    }
                    JobSource::AuthenticatedScatter { source } => {
                        JobSource::AuthenticatedScatter { source }
                    }
                    JobSource::DeferredCatalog {
                        source,
                        source_offset,
                        prefer_internal_home,
                        batch_slot,
                    } => JobSource::DeferredCatalog {
                        source,
                        source_offset: source_offset + consumed as u64,
                        prefer_internal_home,
                        batch_slot,
                    },
                    JobSource::Zero => JobSource::Zero,
                };
                jobs.push(ReadJob {
                    source,
                    destination,
                    destination_offset: destination_offset + consumed,
                    length,
                    expected_digest,
                    verification_index: None,
                });
                if jobs.len() > MAX_PLAN_JOBS {
                    return Err(DeltafinError::new(format!(
                        "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit; increase chunk size"
                    )));
                }
                consumed += length;
            }
        }

        if let SourceAdmission::DeferredRanges(contracts) = &admission
            && sources.len() != contracts.len()
        {
            return Err(DeltafinError::new(
                "deferred range plan contains an unused source contract",
            ));
        }

        if let SourceAdmission::DeferredAuthenticated(contracts) = &admission {
            if sources.len() != contracts.len() {
                return Err(DeltafinError::new(
                    "authenticated deferred plan contains an unused source contract",
                ));
            }
            for (source_index, source) in sources.iter_mut().enumerate() {
                source
                    .scatter_extents
                    .sort_by_key(|extent| extent.source_offset);
                for adjacent in source.scatter_extents.windows(2) {
                    let previous_end = adjacent[0]
                        .source_offset
                        .checked_add(adjacent[0].length as u64)
                        .ok_or_else(|| {
                            DeltafinError::new("authenticated gather range overflows u64")
                        })?;
                    if adjacent[1].source_offset < previous_end {
                        return Err(DeltafinError::new(format!(
                            "authenticated gather ranges overlap in source {}",
                            source.path.display()
                        )));
                    }
                }
                let physical_bytes =
                    source
                        .verifications
                        .iter()
                        .try_fold(0_usize, |total, verification| {
                            total.checked_add(verification.length).ok_or_else(|| {
                                DeltafinError::new(
                                    "authenticated source verification byte count overflows usize",
                                )
                            })
                        })?;
                let first = source.scatter_extents.first().ok_or_else(|| {
                    DeltafinError::new(format!(
                        "authenticated source {} has no gathered extent",
                        source.path.display()
                    ))
                })?;
                jobs.push(ReadJob {
                    source: JobSource::AuthenticatedScatter {
                        source: source_index,
                    },
                    // The authenticated-scatter worker uses the source-owned
                    // destination list. These fields retain ReadJob's compact
                    // common scheduling shape and provide a deterministic
                    // diagnostic anchor only.
                    destination: first.destination,
                    destination_offset: first.destination_offset,
                    length: physical_bytes,
                    expected_digest: None,
                    verification_index: None,
                });
            }
            if jobs.len() > MAX_PLAN_JOBS {
                return Err(DeltafinError::new(format!(
                    "read plan exceeds the {MAX_PLAN_JOBS}-job safety limit"
                )));
            }
        }

        // Match the established longest-processing-time-first policy. Fixed
        // workers pull through one atomic index, so the short chunks naturally
        // collect at the tail instead of stranding one worker on a large read.
        jobs.sort_by_key(|job| Reverse(job.length));
        let mut verified_extents = Vec::new();
        for job in &mut jobs {
            if job.expected_digest.is_some() && !matches!(job.source, JobSource::Vectored { .. }) {
                job.verification_index = Some(verified_extents.len());
                verified_extents.push(AtomicBool::new(false));
            }
        }
        Ok(Self {
            sources: Arc::new(Sources {
                values: sources,
                persistent_count,
                _descriptor_reservation: descriptor_reservation,
            }),
            jobs: Arc::new(jobs),
            vectored_reads: Arc::new(vectored_reads),
            verified_extents: Arc::new(verified_extents),
            buffer_lengths,
            logical_bytes,
        })
    }

    pub fn jobs(&self) -> usize {
        self.jobs.len()
    }

    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    pub fn source_count(&self) -> usize {
        self.sources.values.len()
    }

    pub fn persistent_source_count(&self) -> usize {
        self.sources.persistent_count
    }

    /// Number of lazy persistent descriptors already opened by this plan.
    /// This is observation only; it never opens or stats a source.
    pub fn opened_persistent_source_count(&self) -> usize {
        self.sources
            .values
            .iter()
            .filter(|source| {
                source
                    .persistent_file
                    .as_ref()
                    .is_some_and(|slot| matches!(slot.get(), Some(Ok(_))))
            })
            .count()
    }

    /// Uniform OS cache-admission policy carried by this immutable plan.
    /// Plans built only from zero-fill extents have no source policy.
    pub fn cache_policy(&self) -> Option<CachePolicy> {
        let first = self.sources.values.first()?.cache_policy;
        self.sources
            .values
            .iter()
            .all(|source| source.cache_policy == first)
            .then_some(first)
    }

    pub fn buffer_len(&self, kind: BufferKind) -> usize {
        self.buffer_lengths.get(kind)
    }

    /// Require every source contract to have one canonical length.
    ///
    /// Some raw cache formats use the file itself as their versioned storage
    /// envelope. For those formats, accepting an oversized object just because
    /// all requested extents fit would weaken the format contract and differ
    /// from the established loader's exact-size admission rule.
    pub fn require_all_sources_exact_length(&self, expected: u64) -> Result<()> {
        for source in &self.sources.values {
            if source.length != expected {
                return Err(DeltafinError::new(format!(
                    "source {} is {} bytes; canonical length is {expected}",
                    source.path.display(),
                    source.length,
                )));
            }
        }
        Ok(())
    }
}

fn validate_destinations(extents: &[Extent], declared: BufferLengths) -> Result<()> {
    let mut ranges: Vec<(BufferKind, usize, usize)> = Vec::new();
    for extent in extents {
        let destinations: Box<dyn Iterator<Item = VectoredDestination> + '_> = match extent {
            Extent::Read {
                destination,
                destination_offset,
                length,
                ..
            }
            | Extent::Zero {
                destination,
                destination_offset,
                length,
            } => Box::new(std::iter::once(VectoredDestination::new(
                *destination,
                *destination_offset,
                *length,
            ))),
            Extent::ReadVectored { destinations, .. } => Box::new(destinations.iter().copied()),
        };
        for range in destinations.filter(|range| range.length != 0) {
            let end = range
                .destination_offset
                .checked_add(range.length)
                .ok_or_else(|| DeltafinError::new("destination extent overflows usize"))?;
            if end > declared.get(range.destination) {
                return Err(DeltafinError::new(format!(
                    "{:?} destination extent {}..{} exceeds declared length {}",
                    range.destination,
                    range.destination_offset,
                    end,
                    declared.get(range.destination)
                )));
            }
            ranges.push((range.destination, range.destination_offset, end));
        }
    }
    ranges.sort_unstable();
    for kind in BufferKind::ALL {
        let mut covered = 0_usize;
        for &(_, start, end) in ranges
            .iter()
            .filter(|(range_kind, _, _)| *range_kind == kind)
        {
            if start < covered {
                return Err(DeltafinError::new(format!(
                    "overlapping {:?} destination ranges at {}",
                    kind, start
                )));
            }
            if start > covered {
                return Err(DeltafinError::new(format!(
                    "uncovered {:?} destination range {}..{}; declare padding with Extent::zero",
                    kind, covered, start
                )));
            }
            covered = end;
        }
        let expected = declared.get(kind);
        if covered != expected {
            return Err(DeltafinError::new(format!(
                "uncovered {:?} destination range {}..{}; declared length is {}",
                kind, covered, expected, expected
            )));
        }
    }
    Ok(())
}

fn io_error(operation: &str, path: &Path, error: io::Error) -> DeltafinError {
    DeltafinError::new(format!("{operation} {}: {error}", path.display()))
}

struct AlignedBuffer {
    pointer: NonNull<u8>,
    capacity: usize,
    layout: Layout,
    /// false = the memory belongs to someone else (a Summer pool slot); Drop
    /// must not free it. `keepalive` pins the owner for the buffer's life.
    owned: bool,
    keepalive: Option<Arc<dyn Any + Send + Sync>>,
}

impl AlignedBuffer {
    fn new(capacity: usize) -> Result<Self> {
        let allocated_len = aligned_allocation_len(capacity)?;
        let layout = Layout::from_size_align(allocated_len, BUFFER_ALIGNMENT)
            .map_err(|_| DeltafinError::new("invalid aligned-buffer layout"))?;
        // SAFETY: `layout` is non-zero and valid. Ownership is retained by
        // this value and released exactly once in `Drop` with the same layout.
        let pointer = unsafe { alloc_zeroed(layout) };
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| DeltafinError::new("aligned-buffer allocation failed"))?;
        Ok(Self {
            pointer,
            capacity,
            layout,
            owned: true,
            keepalive: None,
        })
    }

    /// A read destination over memory this reader does NOT own — the Summer
    /// pool's registered arena slot (K3_SUMMER_READ_INTO_SLOT). `pointer`
    /// must be BUFFER_ALIGNMENT-aligned and stay valid while `keepalive`
    /// lives; the caller guarantees no other writer touches the range.
    fn borrowed(
        pointer: NonNull<u8>,
        capacity: usize,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        if capacity == 0 || (pointer.as_ptr() as usize) % BUFFER_ALIGNMENT != 0 {
            return Err(DeltafinError::new(
                "borrowed read destination must be non-empty and page-aligned",
            ));
        }
        let layout = Layout::from_size_align(capacity, BUFFER_ALIGNMENT)
            .map_err(|_| DeltafinError::new("invalid borrowed-buffer layout"))?;
        Ok(Self {
            pointer,
            capacity,
            layout,
            owned: false,
            keepalive: Some(keepalive),
        })
    }

    fn as_slice(&self, logical_len: usize) -> &[u8] {
        debug_assert!(logical_len <= self.capacity);
        // SAFETY: the allocation is live for `self`; immutable access is only
        // exposed after a batch reaches completion.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), logical_len) }
    }

    const fn allocation_len(&self) -> usize {
        self.layout.size()
    }

    fn pointer_at(&self, offset: usize, length: usize) -> *mut u8 {
        debug_assert!(offset <= self.capacity);
        debug_assert!(length <= self.capacity - offset);
        // The caller may turn this into a mutable slice only for a prevalidated
        // destination range while the batch is unpublished.
        self.pointer.as_ptr().wrapping_add(offset)
    }
}

fn aligned_allocation_len(capacity: usize) -> Result<usize> {
    capacity
        .max(1)
        .checked_add(BUFFER_ALIGNMENT - 1)
        .map(|length| length / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT)
        .ok_or_else(|| DeltafinError::new("aligned-buffer size overflows usize"))
}

fn shared_allocation_len(capacities: BufferLengths) -> Result<u64> {
    BufferKind::ALL.iter().try_fold(0_u64, |total, &kind| {
        let bytes = u64::try_from(aligned_allocation_len(capacities.get(kind))?)
            .map_err(|_| DeltafinError::new("aligned-buffer allocation exceeds u64"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| DeltafinError::new("shared-buffer allocation overflows u64"))
    })
}

// SAFETY: mutation is private to `Batch::run_job`, all mutable ranges are
// proven disjoint before publication, and readers only receive immutable views
// after every job has completed.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.owned {
            // Borrowed destination: the owner (pinned by `keepalive`) frees it.
            self.keepalive.take();
            return;
        }
        // SAFETY: `pointer` came from `alloc_zeroed(self.layout)` and has not
        // been deallocated or transferred.
        unsafe { dealloc(self.pointer.as_ptr(), self.layout) }
    }
}

struct SharedBuffers {
    values: [AlignedBuffer; BufferKind::COUNT],
}

impl SharedBuffers {
    fn new(capacities: BufferLengths) -> Result<Self> {
        let capacities = capacities.as_array();
        Ok(Self {
            values: [
                AlignedBuffer::new(capacities[0])?,
                AlignedBuffer::new(capacities[1])?,
                AlignedBuffer::new(capacities[2])?,
            ],
        })
    }

    fn get(&self, kind: BufferKind) -> &AlignedBuffer {
        &self.values[kind.index()]
    }

    /// Buffers whose `Other` destination is borrowed external memory (a
    /// Summer pool slot); the other two kinds are minimal owned stubs.
    fn external_other(
        pointer: NonNull<u8>,
        capacity: usize,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        let other = AlignedBuffer::borrowed(pointer, capacity, keepalive)?;
        let mut values = [
            AlignedBuffer::new(0)?,
            AlignedBuffer::new(0)?,
            AlignedBuffer::new(0)?,
        ];
        values[BufferKind::Other.index()] = other;
        Ok(Self { values })
    }
}

struct ArenaSlot {
    buffers: Option<Arc<SharedBuffers>>,
    capacities: BufferLengths,
    in_use: bool,
}

pub(crate) type BufferRetireHook = Arc<dyn Fn() -> Result<()> + Send + Sync>;
/// Per-blob variant for the retention graveyard: receives the retiring
/// expert-span base pointers (as usizes — raw pointers are not Send) and
/// evicts only their wrappers, leaving the rest of the cache warm.
pub(crate) type BufferDropHook = Arc<dyn Fn(&[usize]) -> Result<()> + Send + Sync>;

fn invoke_buffer_retire_hook(hook: &BufferRetireHook) -> Result<()> {
    match catch_unwind(AssertUnwindSafe(|| hook())) {
        Ok(result) => result,
        Err(_) => Err(DeltafinError::new(
            "storage arena retirement hook panicked before releasing an externally aliased allocation",
        )),
    }
}

/// Process-global donation pool (K3_EXPERT_RETAIN v1): allocations whose
/// retained generation was replaced return HERE instead of freeing, and
/// every arena's growth path draws a fitting donation before allocating
/// fresh. Addresses therefore cycle within a stable set — the address-keyed
/// no-copy Metal wrapper cache stays permanently warm (reuse needs no
/// flush; only FREEING does, and donated slabs are never freed on the
/// steady path). Bounded; overflow falls back to the flush-before-free
/// graveyard.
const DONATION_POOL_LIMIT: usize = 96;

fn expert_donation_pool() -> &'static Mutex<Vec<(Arc<SharedBuffers>, BufferLengths)>> {
    static POOL: OnceLock<Mutex<Vec<(Arc<SharedBuffers>, BufferLengths)>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(Vec::new()))
}

/// Donate a sole-owner allocation into the pool. On rejection (shared
/// ownership, full pool) the Arc comes back to the caller so custody is
/// never silently dropped.
fn donate_expert_allocation_owned(
    buffers: Arc<SharedBuffers>,
) -> std::result::Result<(), Arc<SharedBuffers>> {
    if Arc::strong_count(&buffers) != 1 {
        return Err(buffers);
    }
    let capacities = BufferLengths::new(
        buffers.get(BufferKind::Quantized).allocation_len(),
        buffers.get(BufferKind::Scales).allocation_len(),
        buffers.get(BufferKind::Other).allocation_len(),
    );
    let mut pool = expert_donation_pool().lock().unwrap();
    if pool.len() >= DONATION_POOL_LIMIT {
        return Err(buffers);
    }
    pool.push((buffers, capacities));
    Ok(())
}

/// Pop the smallest donation whose every capacity fits `target`.
fn take_fitting_donation(target: BufferLengths) -> Option<(Arc<SharedBuffers>, BufferLengths)> {
    let mut pool = expert_donation_pool().lock().unwrap();
    let mut best: Option<(usize, usize)> = None;
    for (index, (_, capacities)) in pool.iter().enumerate() {
        let fits = BufferKind::ALL
            .iter()
            .all(|&kind| capacities.get(kind) >= target.get(kind));
        if !fits {
            continue;
        }
        let size = capacities.quantized + capacities.scales + capacities.other;
        if best.is_none_or(|(_, best_size)| size < best_size) {
            best = Some((index, size));
        }
    }
    best.map(|(index, _)| pool.swap_remove(index))
}

struct BufferArena {
    inner: Mutex<Vec<ArenaSlot>>,
    available: Condvar,
    retire_hook: Option<BufferRetireHook>,
}

impl BufferArena {
    fn new(slots: usize) -> Result<Arc<Self>> {
        Self::new_with_retire_hook(slots, None)
    }

    fn new_with_retire_hook(
        slots: usize,
        retire_hook: Option<BufferRetireHook>,
    ) -> Result<Arc<Self>> {
        if slots == 0 {
            return Err(DeltafinError::new(
                "storage buffer arena needs at least one slot",
            ));
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(
                (0..slots)
                    .map(|_| ArenaSlot {
                        buffers: None,
                        capacities: BufferLengths::default(),
                        in_use: false,
                    })
                    .collect(),
            ),
            available: Condvar::new(),
            retire_hook,
        }))
    }

    fn acquire(
        self: &Arc<Self>,
        lengths: BufferLengths,
        wait: bool,
        priority: ReadPriority,
    ) -> Result<Option<Arc<BufferLeaseInner>>> {
        let (slot_index, target_capacities, needs_allocation, retired_capacities, retired_buffers) = {
            let mut slots = self.inner.lock().unwrap();
            loop {
                let free_slots = slots.iter().filter(|slot| !slot.in_use).count();
                // With more than one slot, prefetch may not consume the final
                // slot: a newly routed demand read must remain admissible even
                // if a speculative result is waiting for its CPU consumer.
                let can_admit =
                    priority == ReadPriority::Demand || slots.len() == 1 || free_slots > 1;
                if can_admit {
                    if let Some(index) = slots
                        .iter()
                        .enumerate()
                        .filter(|(_, slot)| !slot.in_use)
                        .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
                        .map(|(index, _)| index)
                    {
                        let slot = &mut slots[index];
                        slot.in_use = true;
                        let old_capacities = slot.capacities;
                        let needs_allocation = slot.buffers.is_none()
                            || BufferKind::ALL
                                .iter()
                                .any(|&kind| old_capacities.get(kind) < lengths.get(kind));
                        let target_capacities = old_capacities.max(lengths);
                        // A free slot has no live CPU lease. Retire its old slab
                        // now so a growth allocation does not temporarily hold
                        // old+new multi-gigabyte buffers at once.
                        let retired_buffers = if needs_allocation {
                            slot.capacities = BufferLengths::default();
                            slot.buffers.take()
                        } else {
                            None
                        };
                        break (
                            index,
                            target_capacities,
                            needs_allocation,
                            old_capacities,
                            retired_buffers,
                        );
                    }
                }
                if !wait {
                    return Ok(None);
                }
                slots = self.available.wait(slots).unwrap();
            }
        };

        let mut retired_buffers = retired_buffers;
        if retired_buffers.is_some()
            && let Some(retire_hook) = self.retire_hook.as_ref()
            && let Err(error) = invoke_buffer_retire_hook(retire_hook)
        {
            // The old allocation remains owned until the external cache has
            // proven that no no-copy wrapper aliases it. A failed flush must
            // therefore restore the exact free slot and reject this growth.
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            debug_assert!(slot.in_use);
            debug_assert!(slot.buffers.is_none());
            slot.buffers = retired_buffers.take();
            slot.capacities = retired_capacities;
            slot.in_use = false;
            self.available.notify_all();
            return Err(DeltafinError::new(format!(
                "storage arena refused to retire an externally aliased allocation: {error}"
            )));
        }
        // Cache retirement has completed before the allocation's final arena
        // Arc can be released. Stable, fitting slabs never invoke the hook.
        drop(retired_buffers);
        let replacement = if needs_allocation {
            // Donated allocations first: stable addresses keep the no-copy
            // wrapper cache warm and cost no page faults. Fresh allocation
            // remains the fallback.
            if let Some(donated) = take_fitting_donation(target_capacities) {
                Some(donated)
            } else {
                match SharedBuffers::new(target_capacities) {
                    Ok(buffers) => Some((Arc::new(buffers), target_capacities)),
                    Err(error) => {
                        self.release(slot_index);
                        return Err(error);
                    }
                }
            }
        } else {
            None
        };

        let buffers = {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            if let Some((buffers, capacities)) = replacement {
                slot.buffers = Some(buffers);
                slot.capacities = capacities;
            }
            Arc::clone(
                slot.buffers
                    .as_ref()
                    .expect("reserved arena slot must contain buffers"),
            )
        };

        Ok(Some(Arc::new(BufferLeaseInner {
            arena: Arc::downgrade(self),
            slot_index,
            buffers: Some(buffers),
            lengths,
        })))
    }

    fn release(&self, slot_index: usize) {
        let mut slots = self.inner.lock().unwrap();
        let slot = slots
            .get_mut(slot_index)
            .expect("buffer lease refers to an unknown arena slot");
        debug_assert!(slot.in_use);
        slot.in_use = false;
        // Demand and prefetch use different admission predicates; wake all so
        // an ineligible prefetch waiter cannot strand an eligible demand one.
        self.available.notify_all();
    }

    /// Complete allocation which a new request may add at its next arena
    /// admission boundary. A fitting free slot needs no allocation. Growth
    /// charges the complete replacement slab, not merely its delta, because a
    /// platform allocator may temporarily retain pages from the retired slab.
    /// When every slot is busy its future capacity is deliberately treated as
    /// unknown and the complete requested slab is charged conservatively.
    fn replacement_admission_bytes(&self, lengths: BufferLengths) -> Result<u64> {
        let slots = self.inner.lock().unwrap();
        if slots.iter().any(|slot| {
            !slot.in_use
                && slot.buffers.is_some()
                && BufferKind::ALL
                    .iter()
                    .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind))
        }) {
            return Ok(0);
        }
        let target = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| !slot.in_use)
            .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
            .map_or(lengths, |(_, slot)| slot.capacities.max(lengths));
        shared_allocation_len(target)
    }

    /// Grow a free arena slot without publishing a read. The retired slab is
    /// released before allocation, matching ordinary low-peak arena growth.
    /// An allocation failure leaves a valid empty slot for the caller's
    /// smaller exact fallback.
    fn reserve_capacity(&self, lengths: BufferLengths) -> Result<()> {
        let (slot_index, target, old_capacities, old_buffers) = {
            let mut slots = self.inner.lock().unwrap();
            if slots.iter().any(|slot| {
                !slot.in_use
                    && slot.buffers.is_some()
                    && BufferKind::ALL
                        .iter()
                        .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind))
            }) {
                return Ok(());
            }
            let index = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| !slot.in_use)
                .min_by_key(|(index, slot)| arena_slot_cost(slot, lengths, *index))
                .map(|(index, _)| index)
                .ok_or_else(|| DeltafinError::new("storage arena has no free slot to reserve"))?;
            let slot = &mut slots[index];
            slot.in_use = true;
            let old_capacities = slot.capacities;
            let target = old_capacities.max(lengths);
            let old_buffers = slot.buffers.take();
            slot.capacities = BufferLengths::default();
            (index, target, old_capacities, old_buffers)
        };

        if old_buffers.is_some()
            && let Some(retire_hook) = self.retire_hook.as_ref()
            && let Err(error) = invoke_buffer_retire_hook(retire_hook)
        {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            slot.buffers = old_buffers;
            slot.capacities = old_capacities;
            slot.in_use = false;
            self.available.notify_all();
            return Err(error);
        }
        drop(old_buffers);
        let replacement = match SharedBuffers::new(target) {
            Ok(buffers) => Arc::new(buffers),
            Err(error) => {
                let mut slots = self.inner.lock().unwrap();
                let slot = &mut slots[slot_index];
                slot.in_use = false;
                self.available.notify_all();
                return Err(error);
            }
        };
        {
            let mut slots = self.inner.lock().unwrap();
            let slot = &mut slots[slot_index];
            slot.buffers = Some(replacement);
            slot.capacities = target;
            slot.in_use = false;
            self.available.notify_all();
        }
        Ok(())
    }
}

impl Drop for BufferArena {
    fn drop(&mut self) {
        let Some(retire_hook) = self.retire_hook.as_ref() else {
            return;
        };
        let slots = self
            .inner
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !slots.iter().any(|slot| slot.buffers.is_some()) {
            return;
        }
        if invoke_buffer_retire_hook(retire_hook).is_err() {
            // Teardown cannot return an error. Leaking these bounded slabs is
            // safer than freeing pages still named by a no-copy device cache;
            // the hook's captured provider lease remains live through this
            // drop attempt, so ordinary successful teardown releases both.
            for slot in slots {
                if let Some(buffers) = slot.buffers.take() {
                    std::mem::forget(buffers);
                }
            }
        }
    }
}

fn arena_slot_cost(
    slot: &ArenaSlot,
    lengths: BufferLengths,
    index: usize,
) -> (u8, u128, u128, usize) {
    let fits = slot.buffers.is_some()
        && BufferKind::ALL
            .iter()
            .all(|&kind| slot.capacities.get(kind) >= lengths.get(kind));
    let mut growth = 0_u128;
    let mut waste_or_total = 0_u128;
    for kind in BufferKind::ALL {
        let old = if slot.buffers.is_some() {
            slot.capacities.get(kind)
        } else {
            0
        };
        let requested = lengths.get(kind);
        if fits {
            waste_or_total += (old - requested) as u128;
        } else {
            let target = old.max(requested);
            growth += (target - old) as u128;
            waste_or_total += target as u128;
        }
    }
    // Reuse a fitting slab first. Otherwise minimize growth in total retained
    // memory, then the resulting slab size; the index is a deterministic tie.
    (
        !fits as u8,
        if fits { waste_or_total } else { growth },
        waste_or_total,
        index,
    )
}

struct BufferLeaseInner {
    arena: Weak<BufferArena>,
    slot_index: usize,
    buffers: Option<Arc<SharedBuffers>>,
    lengths: BufferLengths,
}

impl BufferLeaseInner {
    fn buffers(&self) -> &SharedBuffers {
        self.buffers
            .as_deref()
            .expect("live buffer lease must contain its arena buffers")
    }
}

impl Drop for BufferLeaseInner {
    fn drop(&mut self) {
        // Drop the lease's buffer Arc before advertising the arena slot as
        // free. A waiter growing this slot can then retire the slot's final Arc
        // before allocating its replacement, even under concurrent teardown.
        drop(self.buffers.take());
        if let Some(arena) = self.arena.upgrade() {
            arena.release(self.slot_index);
        }
    }
}

/// A CPU-complete arena lease. Its storage returns to the bounded pool when
/// this value drops. A future asynchronous Metal/CUDA bridge must retain this
/// lease until its device completion event fires; this type deliberately does
/// not pretend that submitting a GPU command is completion.
pub struct LayerBuffers {
    lease: Arc<BufferLeaseInner>,
}

/// Cross-token keep-alive for a lease's underlying allocation WITHOUT the
/// arena slot: dropping the `LayerBuffers` still releases its slot, and a
/// waiter reusing that slot allocates a replacement allocation while these
/// bytes stay alive here (the release path was designed for exactly this
/// teardown order). Used by K3_EXPERT_RETAIN's previous-token expert tier —
/// strictly read-only, strictly zero-copy.
pub struct RetainedAllocation {
    buffers: Arc<SharedBuffers>,
    lengths: BufferLengths,
}

impl RetainedAllocation {
    pub fn other(&self) -> &[u8] {
        self.buffers
            .get(BufferKind::Other)
            .as_slice(self.lengths.other)
    }

    /// Full allocation envelope of the Other buffer — NOT the logical
    /// length. A recycled slab's Metal wrapper history spans every
    /// span-aligned address over the ENVELOPE (earlier leases packed
    /// different union sizes from the same base), so per-blob eviction
    /// before freeing must cover all of it (2026-08-26 review, CRITICAL).
    pub(crate) fn other_envelope_len(&self) -> usize {
        self.buffers.get(BufferKind::Other).allocation_len()
    }

    /// Return this allocation to the donation pool (v1 recycle path).
    /// On failure (pool full, storage unexpectedly shared) custody comes
    /// BACK to the caller — donated-or-returned, never silently dropped,
    /// because dropping here would free wrapper-aliased pages without the
    /// mandatory flush.
    pub(crate) fn donate(self) -> Option<Self> {
        let Self { buffers, lengths } = self;
        match donate_expert_allocation_owned(buffers) {
            Ok(()) => None,
            Err(buffers) => Some(Self { buffers, lengths }),
        }
    }
}

impl LayerBuffers {
    pub fn quantized(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Quantized)
            .as_slice(self.lease.lengths.quantized)
    }

    pub fn scales(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Scales)
            .as_slice(self.lease.lengths.scales)
    }

    pub fn other(&self) -> &[u8] {
        self.lease
            .buffers()
            .get(BufferKind::Other)
            .as_slice(self.lease.lengths.other)
    }

    /// Borrow the stable host pointer while this lease remains alive.
    ///
    /// This is not yet a GPU lifetime contract. Callers must not drop the
    /// `LayerBuffers` while any native CPU consumer can still access it.
    pub fn pointer(&self, kind: BufferKind) -> *const u8 {
        self.lease.buffers().get(kind).pointer.as_ptr()
    }

    /// Return the allocator's complete rounded backing lengths, not the
    /// manifest's logical tensor lengths.
    ///
    /// A device bridge that borrows these pages (Metal shared storage, CUDA
    /// host registration, or a later Windows equivalent) must receive the
    /// actual allocation envelope. Inferring it from the logical slice would
    /// make the final aligned page invisible to the lifetime contract.
    pub(crate) fn allocation_lengths(&self) -> BufferLengths {
        BufferLengths::new(
            self.lease
                .buffers()
                .get(BufferKind::Quantized)
                .allocation_len(),
            self.lease
                .buffers()
                .get(BufferKind::Scales)
                .allocation_len(),
            self.lease.buffers().get(BufferKind::Other).allocation_len(),
        )
    }

    /// Detach the underlying allocation from its arena slot for the
    /// previous-token expert tier (K3_EXPERT_RETAIN). The slot is left
    /// empty, so the NEXT acquire allocates a fresh slab through the
    /// ordinary needs-allocation path (no retire, no external-cache hook) —
    /// the arena can never recycle these bytes underneath the retained
    /// spans. This lease's own reference stays intact for its remaining
    /// lifetime; once it drops, retention is the sole owner. Returns None
    /// when the arena is gone or the slot was already detached.
    pub(crate) fn detach_allocation_for_retention(&self) -> Option<RetainedAllocation> {
        let arena = self.lease.arena.upgrade()?;
        let buffers = {
            let mut slots = arena.inner.lock().unwrap();
            let slot = slots.get_mut(self.lease.slot_index)?;
            debug_assert!(slot.in_use, "detaching lease must still hold its slot");
            slot.capacities = BufferLengths::default();
            slot.buffers.take()?
        };
        Some(RetainedAllocation {
            buffers,
            lengths: self.lease.lengths,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReadStats {
    /// Logical bytes published in the compact destination buffers. An
    /// authenticated gather may physically read more enclosing source bytes
    /// while proving their digest; `elapsed` includes that work.
    pub bytes: u64,
    pub jobs: usize,
    pub workers: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReadPriority {
    Demand,
    Prefetch,
}

// ---------------------------------------------------------------------------
// K3_STALL_TRACE=1 — measure-only decomposition of blocking read waits.
//
// Prices the per-token blocking-read bucket by classifying, for every
// ReadTicket::wait that actually blocks, where the wait window went:
//   queue-late   — the batch's critical job (the one that finished last) was
//                  still UNCLAIMED while the engine waited: workers/devices
//                  were busy elsewhere. More spindles or workers could help.
//   service-late — the critical job was CLAIMED and inside pread while the
//                  engine waited: the read itself is the pole. Faster
//                  per-file assembly (chunking/split-homing) could help.
// Waits that find the batch already complete are counted as `ready` (the
// read was fully hidden behind compute — nothing to fix). `lead` is how long
// the batch existed before the engine blocked on it (submission lead time);
// a small lead with large waits is the route/submit-late signature that no
// storage change can fix. Totals are printed by print_run_stats via
// stall_trace_report(). Default-off; when the env flag is absent the only
// cost is one branch per site and a None field per batch.
fn stall_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("K3_STALL_TRACE").is_some_and(|v| v == "1"))
}

// ---------------------------------------------------------------------------
// K3_SPLIT_READ=N — split-homed chunked expert reads (default off).
//
// Splits every deferred-catalog expert read into N page-aligned chunk jobs
// and routes the first K3_SPLIT_READ_INTERNAL of them hot-tier-first while
// the rest probe the enclosures first (tier as final fallback). A dual-homed
// expert therefore assembles from the internal SSD AND its enclosure
// simultaneously; a single-homed expert degrades to same-device chunked
// parallelism (measured neutral). Motivated by the 2026-08-25 stall trace
// (prefetch waits 99.9% service-late, demand waits 69% queue-late) and the
// k3-genbench gate (split assembly −13.7% p50 at 21.5 GB/s across 3
// devices). Byte-identity is unaffected: chunks land in the same slab bytes
// whichever home serves them.
fn split_read_chunks() -> usize {
    static CHUNKS: OnceLock<usize> = OnceLock::new();
    *CHUNKS.get_or_init(|| {
        std::env::var("K3_SPLIT_READ")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(1, |value| value.clamp(1, 8))
    })
}

/// K3_SPLIT_READ_DEMAND=N — chunk DEMAND-priority batches only (the
/// audit's surviving cell: demand blocked improved inside the negative
/// prefetch experiment). Home-PRESERVING: every chunk probes internal
/// first (the legacy tier-first order) via the per-batch descriptor
/// cache — no bytes move between devices, only job granularity changes,
/// making miss bursts preemptible at chunk size instead of whole files.
fn split_read_demand_chunks() -> usize {
    static CHUNKS: OnceLock<usize> = OnceLock::new();
    *CHUNKS.get_or_init(|| {
        std::env::var("K3_SPLIT_READ_DEMAND")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map_or_else(split_read_chunks, |value| value.clamp(1, 8))
    })
}

fn split_read_internal_chunks(chunks: usize) -> usize {
    static INTERNAL: OnceLock<Option<usize>> = OnceLock::new();
    let configured = *INTERNAL.get_or_init(|| {
        std::env::var("K3_SPLIT_READ_INTERNAL")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    });
    // Default: the measured device-rate split, internal 13.73 of an
    // aggregate 19.31 GB/s ≈ 71% of the chunks, at least one to each side.
    let default = ((chunks as f64) * 13.73 / (13.73 + 5.58)).round() as usize;
    // K3_SPLIT_READ_INTERNAL=<chunks> is allowed: every prefetch chunk then
    // probes the hot tier first, so a hot-resident expert is served whole by
    // the hot device (the split-trace test of 2026-09-05 showed the hot half
    // waiting ~5 ms for its dir_b partner). The default is unchanged.
    configured
        .unwrap_or(default.clamp(1, chunks.saturating_sub(1).max(1)))
        .clamp(1, chunks.max(1))
}

// K3_SPLIT_ETA=1 — least-expected-completion homing for split chunks
// (default off). Every chunk of a split read is homed on whichever tier
// directory (hot, dir_c, dir_b, primary) holds the file and has the smallest
// (in-flight chunks + 1) x chunk_bytes / rate, instead of the fixed
// hot-first / enclosure-chain assignment. Rates in GB/s come from
// K3_SPLIT_ETA_GBPS="hot,dir_c,dir_b,primary" (default 5.6,5.6,5.6,13.7).
// Byte-identity is unaffected: chunks land in the same slab bytes whichever
// tier serves them.
fn split_eta_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("K3_SPLIT_ETA").is_ok_and(|value| value.trim() == "1"))
}

fn split_eta_rates() -> &'static [f64; 4] {
    static RATES: OnceLock<[f64; 4]> = OnceLock::new();
    RATES.get_or_init(|| {
        let mut rates = [5.6_f64, 5.6, 5.6, 13.7];
        if let Ok(raw) = std::env::var("K3_SPLIT_ETA_GBPS") {
            for (slot, value) in rates.iter_mut().zip(raw.split(',')) {
                if let Ok(parsed) = value.trim().parse::<f64>() {
                    if parsed > 0.0 {
                        *slot = parsed;
                    }
                }
            }
        }
        rates
    })
}

/// K3_SPLIT_CAP="hot,dir_c,dir_b,primary" — per-tier in-flight chunk caps for
/// the capped router (requires K3_SPLIT_ETA=1). Default off.
fn split_caps() -> Option<&'static [u64; 4]> {
    static CAPS: OnceLock<Option<[u64; 4]>> = OnceLock::new();
    CAPS.get_or_init(|| {
        let raw = std::env::var("K3_SPLIT_CAP").ok()?;
        let mut caps = [u64::MAX; 4];
        for (slot, value) in caps.iter_mut().zip(raw.split(',')) {
            if let Ok(parsed) = value.trim().parse::<u64>() {
                *slot = parsed.max(1);
            }
        }
        Some(caps)
    })
    .as_ref()
}

static SPLIT_ETA_OUTSTANDING: [PaddedU64; 4] = [
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
];
static SPLIT_ETA_SERVED: [PaddedU64; 4] = [
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
];

/// Expected microseconds for `tier` to finish one more chunk of `bytes`.
fn split_eta_us(tier: usize, bytes: u64) -> u64 {
    let queued = SPLIT_ETA_OUTSTANDING[tier].0.load(Ordering::Relaxed);
    let unit_us = (bytes as f64) / (split_eta_rates()[tier] * 1.0e3);
    (unit_us * (queued as f64 + 1.0)).round() as u64
}

// ---------------------------------------------------------------------------
// K3_PLAN_BALANCE=1 (2026-09-06) — the plan-path reads (45% of a run's expert
// bytes: prefetch generations and every whole-file job) resolved their home
// with a static chain, dir_c -> hot -> dir_b -> primary, first hit wins. That
// is the defect behind White pinned at its ceiling (dir_b outranks the
// internal for every file it holds) and behind every failed mirror layout
// (a full dir_b takes everything). With the knob on, the plan path picks the
// tier with the lowest expected completion time among the tiers that hold
// the file — the same in-flight × bytes / rate cost the ETA router uses for
// chunked reads, rates from K3_SPLIT_ETA_GBPS (hot, dir_c, dir_b, primary) —
// charges that tier at plan time and retires it when the read lands.
// ---------------------------------------------------------------------------
pub fn plan_balance_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("K3_PLAN_BALANCE").is_ok_and(|v| v.trim() == "1"))
}

static TIER_DIRS: OnceLock<Mutex<Vec<(PathBuf, usize)>>> = OnceLock::new();

/// Register a tier's expert directory (0 hot, 1 dir_c, 2 dir_b, 3 primary) so
/// a finished plan-path read can retire its in-flight charge by path.
pub fn register_tier_dir(tier: usize, path: &Path) {
    let dirs = TIER_DIRS.get_or_init(|| Mutex::new(Vec::new()));
    let mut dirs = dirs.lock().unwrap();
    dirs.retain(|(_, t)| *t != tier);
    dirs.push((path.to_path_buf(), tier));
}

fn tier_of_path(path: &Path) -> Option<usize> {
    let parent = path.parent()?;
    let dirs = TIER_DIRS.get()?.lock().unwrap();
    dirs.iter().find(|(dir, _)| dir == parent).map(|(_, t)| *t)
}

/// Pick the tier for one plan-path read among the tiers that hold the file
/// (index = tier), charging it in flight. None when nothing holds it.
pub fn plan_pick_tier(holds: [bool; 4], bytes: u64) -> Option<usize> {
    let mut best: Option<(u64, usize)> = None;
    for tier in 0..4 {
        if !holds[tier] {
            continue;
        }
        let cost = split_eta_us(tier, bytes);
        if best.is_none_or(|(c, _)| cost < c) {
            best = Some((cost, tier));
        }
    }
    let (_, tier) = best?;
    SPLIT_ETA_OUTSTANDING[tier].0.fetch_add(1, Ordering::Relaxed);
    SPLIT_ETA_SERVED[tier].0.fetch_add(1, Ordering::Relaxed);
    Some(tier)
}

fn plan_retire_path(path: &Path) {
    if let Some(tier) = tier_of_path(path) {
        SPLIT_ETA_OUTSTANDING[tier]
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)))
            .ok();
    }
}

/// In-flight accounting for one ETA-homed chunk; retired on drop (also on
/// error paths), so a device can never be exiled by a leaked increment.
struct SplitEtaGuard(usize);

impl SplitEtaGuard {
    fn issue(tier: usize) -> Self {
        SPLIT_ETA_OUTSTANDING[tier].0.fetch_add(1, Ordering::Relaxed);
        SPLIT_ETA_SERVED[tier].0.fetch_add(1, Ordering::Relaxed);
        SplitEtaGuard(tier)
    }
}

impl Drop for SplitEtaGuard {
    fn drop(&mut self) {
        SPLIT_ETA_OUTSTANDING[self.0]
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)))
            .ok();
    }
}

/// Chunks served per tier under K3_SPLIT_ETA: (hot, dir_c, dir_b, primary).
pub fn split_eta_report() -> Option<(u64, u64, u64, u64)> {
    if !split_eta_enabled() {
        return None;
    }
    let n = |tier: usize| SPLIT_ETA_SERVED[tier].0.load(Ordering::Relaxed);
    Some((n(0), n(1), n(2), n(3)))
}

// ---------------------------------------------------------------------------
// K3_SPLIT_TRACE=1 — completion trace of split-read halves (diagnostic).
//
// A split expert is usable only when BOTH halves have landed, so the question
// "does splitting with a slow device help?" is answered by which half arrives
// last and by how much, grouped by the pair of tiers that served the halves.
// Each chunk job records (batch, slot, chunk offset, tier, completion time on
// the batch clock, transfer duration); the report pairs the chunks of every
// source and prints per (priority, tier pair) percentiles. Default off; the
// record is one mutex push per chunk (~2,500/s), invisible at run scale.
// ---------------------------------------------------------------------------
static SPLIT_TRACE_BATCH_SEQ: AtomicU64 = AtomicU64::new(1);
const SPLIT_TRACE_FD_SLOTS: usize = 65_536;
const SPLIT_TRACE_MAX_RECORDS: usize = 8_000_000;
static SPLIT_FD_TIER: [AtomicU32; SPLIT_TRACE_FD_SLOTS] =
    [const { AtomicU32::new(0) }; SPLIT_TRACE_FD_SLOTS];

#[derive(Clone, Copy)]
struct SplitTraceRecord {
    batch: u64,
    slot: u16,
    chunk_offset: u64,
    tier: u8,
    prefetch: bool,
    done_ns: u64,
    read_ns: u64,
}

static SPLIT_TRACE: Mutex<Vec<SplitTraceRecord>> = Mutex::new(Vec::new());

fn split_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("K3_SPLIT_TRACE").is_ok_and(|value| value == "1"))
}

/// Remember which tier a descriptor came from (tier + 1; 0 = unknown).
fn split_fd_tier_set(descriptor: i32, tier: u8) {
    if descriptor >= 0 && (descriptor as usize) < SPLIT_TRACE_FD_SLOTS {
        SPLIT_FD_TIER[descriptor as usize].store(u32::from(tier) + 1, Ordering::Relaxed);
    }
}

fn split_fd_tier(descriptor: i32) -> u8 {
    if descriptor >= 0 && (descriptor as usize) < SPLIT_TRACE_FD_SLOTS {
        let stored = SPLIT_FD_TIER[descriptor as usize].load(Ordering::Relaxed);
        if stored == 0 { 255 } else { (stored - 1) as u8 }
    } else {
        255
    }
}

fn split_trace_record(record: SplitTraceRecord) {
    let mut records = SPLIT_TRACE.lock().unwrap();
    if records.len() < SPLIT_TRACE_MAX_RECORDS {
        records.push(record);
    }
}

fn split_trace_tier_name(tier: u8) -> &'static str {
    match tier {
        0 => "hot",
        1 => "dir_c",
        2 => "dir_b",
        3 => "primary",
        _ => "?",
    }
}

fn split_trace_percentile_ms(values: &mut [u64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * fraction).round() as usize;
    values[index] as f64 / 1.0e6
}

/// One-line statement of the split-read configuration for the startup log, so
/// a run's telemetry shows these settings independently of the mirror,
/// retain and summer feature lines.
pub fn split_read_config_report() -> String {
    let chunks = split_read_chunks();
    let demand = split_read_demand_chunks();
    let prefetch_internal = if chunks > 1 { split_read_internal_chunks(chunks) } else { 1 };
    format!(
        "chunks={} demand_chunks={} prefetch_hot_side_chunks={} demand_home=hot-first-both-chunks tier_balance={} split_eta={} trace={}",
        chunks,
        demand,
        prefetch_internal,
        u8::from(tier_balance_enabled()),
        u8::from(split_eta_enabled()),
        u8::from(split_trace_enabled()),
    )
}

/// The K3_SPLIT_TRACE report: one line per (priority, tier of chunk 0, tier
/// of chunk 1) with completion percentiles of each half on the batch clock,
/// the share of sources whose second (enclosure-side) half landed last, the
/// gap between the halves, and each half's transfer time; then one line per
/// tier with the transfer-time percentiles of every chunk it served. None
/// when the trace is off or nothing was recorded.
pub fn split_trace_report() -> Option<Vec<String>> {
    if !split_trace_enabled() {
        return None;
    }
    let records = std::mem::take(&mut *SPLIT_TRACE.lock().unwrap());
    if records.is_empty() {
        return Some(vec!["[split] trace on, no chunked reads recorded".to_string()]);
    }
    let mut by_source: HashMap<(u64, u16), Vec<SplitTraceRecord>> = HashMap::new();
    for record in &records {
        by_source.entry((record.batch, record.slot)).or_default().push(*record);
    }
    struct PairStats {
        n: u64,
        c1_last: u64,
        c0_done: Vec<u64>,
        c1_done: Vec<u64>,
        delta: Vec<u64>,
        c0_read: Vec<u64>,
        c1_read: Vec<u64>,
    }
    let mut pairs: HashMap<(bool, u8, u8), PairStats> = HashMap::new();
    let mut unpaired = 0_u64;
    for (_, mut chunks) in by_source {
        if chunks.len() != 2 {
            unpaired += 1;
            continue;
        }
        chunks.sort_by_key(|record| record.chunk_offset);
        let (first, second) = (chunks[0], chunks[1]);
        let stats = pairs
            .entry((first.prefetch, first.tier, second.tier))
            .or_insert_with(|| PairStats {
                n: 0,
                c1_last: 0,
                c0_done: Vec::new(),
                c1_done: Vec::new(),
                delta: Vec::new(),
                c0_read: Vec::new(),
                c1_read: Vec::new(),
            });
        stats.n += 1;
        if second.done_ns > first.done_ns {
            stats.c1_last += 1;
        }
        stats.c0_done.push(first.done_ns);
        stats.c1_done.push(second.done_ns);
        stats.delta.push(first.done_ns.abs_diff(second.done_ns));
        stats.c0_read.push(first.read_ns);
        stats.c1_read.push(second.read_ns);
    }
    let mut lines = Vec::new();
    let mut keys: Vec<_> = pairs.keys().copied().collect();
    keys.sort_by_key(|key| (key.0, std::cmp::Reverse(pairs[key].n)));
    for key in keys {
        let stats = pairs.get_mut(&key).unwrap();
        lines.push(format!(
            "[split] {} c0={} c1={} n={} c0_done p50/p90={:.2}/{:.2}ms c1_done p50/p90={:.2}/{:.2}ms c1_last={:.1}% gap p50/p90={:.2}/{:.2}ms c0_read p50/p90={:.2}/{:.2}ms c1_read p50/p90={:.2}/{:.2}ms",
            if key.0 { "prefetch" } else { "demand" },
            split_trace_tier_name(key.1),
            split_trace_tier_name(key.2),
            stats.n,
            split_trace_percentile_ms(&mut stats.c0_done, 0.5),
            split_trace_percentile_ms(&mut stats.c0_done, 0.9),
            split_trace_percentile_ms(&mut stats.c1_done, 0.5),
            split_trace_percentile_ms(&mut stats.c1_done, 0.9),
            100.0 * stats.c1_last as f64 / stats.n as f64,
            split_trace_percentile_ms(&mut stats.delta, 0.5),
            split_trace_percentile_ms(&mut stats.delta, 0.9),
            split_trace_percentile_ms(&mut stats.c0_read, 0.5),
            split_trace_percentile_ms(&mut stats.c0_read, 0.9),
            split_trace_percentile_ms(&mut stats.c1_read, 0.5),
            split_trace_percentile_ms(&mut stats.c1_read, 0.9),
        ));
    }
    for tier in [0_u8, 1, 2, 3, 255] {
        let mut reads: Vec<u64> = records
            .iter()
            .filter(|record| record.tier == tier)
            .map(|record| record.read_ns)
            .collect();
        if reads.is_empty() {
            continue;
        }
        let n = reads.len();
        lines.push(format!(
            "[split] tier {} chunks={} read p50/p90/p99={:.2}/{:.2}/{:.2}ms",
            split_trace_tier_name(tier),
            n,
            split_trace_percentile_ms(&mut reads, 0.5),
            split_trace_percentile_ms(&mut reads, 0.9),
            split_trace_percentile_ms(&mut reads, 0.99),
        ));
    }
    lines.push(format!(
        "[split] records={} sources_unpaired={} (a source counts as unpaired when it had one chunk or more than two)",
        records.len(),
        unpaired
    ));
    Some(lines)
}

/// Probe exactly one tier directory (0 hot, 1 dir_c, 2 secondary/dir_b,
/// 3 primary) for `source`; Ok(None) = that tier does not hold the file.
fn open_catalog_tier(
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    tier: usize,
) -> Result<Option<File>> {
    unsafe extern "C" {
        fn openat(
            directory: libc::c_int,
            path: *const libc::c_char,
            flags: libc::c_int,
            ...
        ) -> libc::c_int;
    }
    let directory = match tier {
        0 => catalog.hot_directory.as_ref(),
        1 => catalog.tertiary_directory.as_ref(),
        2 => catalog.secondary_directory.as_ref(),
        _ => Some(&catalog.directory),
    };
    let Some(directory) = directory else {
        return Ok(None);
    };
    let open_started = std::time::Instant::now();
    // SAFETY: live directory descriptor retained by the catalog; validated
    // NUL-terminated direct child name; flags never create a file.
    let descriptor = unsafe {
        openat(
            directory.as_raw_fd(),
            source.as_c_str().as_ptr(),
            open_cloexec_nofollow(),
        )
    };
    EXPERT_OPEN_NS.fetch_add(open_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    EXPERT_OPEN_COUNT.fetch_add(1, Ordering::Relaxed);
    if descriptor >= 0 {
        // Tier tag for the read trace (ETA-mode opens bypass open_catalog_home,
        // whose tag the trace's device record relies on; without it the
        // record's source fell back to "read" and the device was lost).
        split_fd_tier_set(descriptor, tier as u8);
        return finish_catalog_open(catalog, source, descriptor).map(Some);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::NotFound {
        Ok(None)
    } else {
        Err(catalog_io_error("open split-eta catalog source", catalog, source, error))
    }
}

// K3_RDADVISE=1 — issue an F_RDADVISE advisory read for the exact byte range
// of every expert pread before the copying read starts (default off). The
// kernel then queues the whole range to the drive at once instead of feeding
// it request-by-request as the synchronous pread progresses, which is the
// difference between the ~0.7 GB/s per-stream rate the enclosures show under
// the engine and the ~5.6 GB/s they show to a deep-queue raw reader.
// Byte-identity is unaffected: the pread still copies the same bytes.
fn rdadvise_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("K3_RDADVISE").is_ok_and(|value| value.trim() == "1"))
}

fn advise_read_range(file: &File, offset: u64, length: usize) {
    if !rdadvise_enabled() || length == 0 {
        return;
    }
    let advisory = libc::radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: length.min(i32::MAX as usize) as libc::c_int,
    };
    // SAFETY: valid descriptor and a fully initialized radvisory struct; the
    // call never modifies user memory and failure is advisory-only.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_RDADVISE, &advisory as *const libc::radvisory);
    }
}

fn stall_trace_now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

struct BatchTrace {
    created_ns: u64,
    critical_done_ns: AtomicU64,
    critical_claim_ns: AtomicU64,
}

impl BatchTrace {
    fn new_if_enabled() -> Option<Box<BatchTrace>> {
        stall_trace_enabled().then(|| {
            Box::new(BatchTrace {
                created_ns: stall_trace_now_ns(),
                critical_done_ns: AtomicU64::new(0),
                critical_claim_ns: AtomicU64::new(0),
            })
        })
    }

    /// Record a finished job; keep the (claim, done) pair of whichever job
    /// finished last. The claim store races benignly with a concurrent later
    /// finisher — telemetry-grade accuracy is sufficient here.
    fn record_job(&self, claim_ns: u64, done_ns: u64) {
        let mut current = self.critical_done_ns.load(Ordering::Relaxed);
        while done_ns > current {
            match self.critical_done_ns.compare_exchange_weak(
                current,
                done_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.critical_claim_ns.store(claim_ns, Ordering::Relaxed);
                    break;
                }
                Err(observed) => current = observed,
            }
        }
    }
}

/// Lead-bucket upper bounds in ms; the last bucket is unbounded. The
/// blocked-vs-lead curve discriminates the two demand-side models: if
/// blocked ≈ max(0, backlog − lead), deeper hint lead converts to speed;
/// if blocked is lead-independent, the oracle-depth ladder is not worth
/// building (2026-08-25 second opinion, item 0).
const STALL_LEAD_BUCKETS_MS: [u64; 7] = [10, 15, 20, 25, 30, 40, 60];

#[derive(Default)]
struct StallCounters {
    waits: AtomicU64,
    ready: AtomicU64,
    wait_ns: AtomicU64,
    lead_ns: AtomicU64,
    queue_ns: AtomicU64,
    service_ns: AtomicU64,
    lead_bucket_waits: [AtomicU64; 8],
    lead_bucket_blocked_ns: [AtomicU64; 8],
}

fn stall_counters() -> &'static [StallCounters; 2] {
    static COUNTERS: OnceLock<[StallCounters; 2]> = OnceLock::new();
    COUNTERS.get_or_init(|| [StallCounters::default(), StallCounters::default()])
}

fn stall_trace_record(
    priority: ReadPriority,
    wait_start_ns: u64,
    wait_end_ns: u64,
    was_ready: bool,
    trace: &BatchTrace,
) {
    let counters = &stall_counters()[match priority {
        ReadPriority::Demand => 0,
        ReadPriority::Prefetch => 1,
    }];
    counters.waits.fetch_add(1, Ordering::Relaxed);
    counters
        .lead_ns
        .fetch_add(wait_start_ns.saturating_sub(trace.created_ns), Ordering::Relaxed);
    if was_ready {
        counters.ready.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let blocked_ns = wait_end_ns.saturating_sub(wait_start_ns);
    counters.wait_ns.fetch_add(blocked_ns, Ordering::Relaxed);
    let lead_ms = wait_start_ns.saturating_sub(trace.created_ns) / 1_000_000;
    let bucket = STALL_LEAD_BUCKETS_MS
        .iter()
        .position(|&bound| lead_ms < bound)
        .unwrap_or(STALL_LEAD_BUCKETS_MS.len());
    counters.lead_bucket_waits[bucket].fetch_add(1, Ordering::Relaxed);
    counters.lead_bucket_blocked_ns[bucket].fetch_add(blocked_ns, Ordering::Relaxed);
    let critical_done = trace.critical_done_ns.load(Ordering::Relaxed);
    if critical_done == 0 {
        return;
    }
    let critical_claim = trace.critical_claim_ns.load(Ordering::Relaxed);
    let claim_clamped = critical_claim.clamp(wait_start_ns, wait_end_ns);
    counters
        .queue_ns
        .fetch_add(claim_clamped - wait_start_ns, Ordering::Relaxed);
    counters
        .service_ns
        .fetch_add(wait_end_ns.saturating_sub(claim_clamped), Ordering::Relaxed);
}

/// Formatted totals for print_run_stats, or None when tracing is off.
pub fn stall_trace_report() -> Option<String> {
    if !stall_trace_enabled() {
        return None;
    }
    let seconds = |ns: u64| ns as f64 / 1.0e9;
    let mut lines = String::new();
    for (label, counters) in [("demand", &stall_counters()[0]), ("prefetch", &stall_counters()[1])]
    {
        let waits = counters.waits.load(Ordering::Relaxed);
        if waits == 0 {
            continue;
        }
        let ready = counters.ready.load(Ordering::Relaxed);
        let wait_ns = counters.wait_ns.load(Ordering::Relaxed);
        let queue_ns = counters.queue_ns.load(Ordering::Relaxed);
        let service_ns = counters.service_ns.load(Ordering::Relaxed);
        let share = |part: u64| {
            if wait_ns == 0 { 0.0 } else { part as f64 * 100.0 / wait_ns as f64 }
        };
        if !lines.is_empty() {
            lines.push('\n');
        }
        lines.push_str(&format!(
            "[stall-trace] {label}: waits={waits} ready={ready} blocked_wait={:.3}s queue={:.3}s ({:.1}%) service={:.3}s ({:.1}%) mean_lead={:.3}ms",
            seconds(wait_ns),
            seconds(queue_ns),
            share(queue_ns),
            seconds(service_ns),
            share(service_ns),
            seconds(counters.lead_ns.load(Ordering::Relaxed)) * 1.0e3 / waits as f64,
        ));
        // The blocked-vs-lead curve: per lead bucket, blocked-wait count and
        // MEAN blocked ms. Falling mean with rising lead = depth converts.
        let mut curve = format!("\n[stall-lead] {label}:");
        for (index, waits_in_bucket) in counters.lead_bucket_waits.iter().enumerate() {
            let bucket_waits = waits_in_bucket.load(Ordering::Relaxed);
            if bucket_waits == 0 {
                continue;
            }
            let blocked = counters.lead_bucket_blocked_ns[index].load(Ordering::Relaxed);
            let label = if index < STALL_LEAD_BUCKETS_MS.len() {
                format!("<{}ms", STALL_LEAD_BUCKETS_MS[index])
            } else {
                format!(">={}ms", STALL_LEAD_BUCKETS_MS[STALL_LEAD_BUCKETS_MS.len() - 1])
            };
            curve.push_str(&format!(
                " {label} n={bucket_waits} mean={:.2}ms |",
                blocked as f64 / bucket_waits as f64 / 1.0e6,
            ));
        }
        lines.push_str(&curve);
    }
    (!lines.is_empty()).then_some(lines)
}

// Split out from Batch so a worker can hand off its own strong reference to
// the lease-holding Batch before publishing completion. Batch::run_quantum
// used to notify from inside its own `&self` call, while the caller
// (worker_main) kept its Arc<Batch> -- and therefore the batch's arena lease
// -- alive until the *next* loop iteration. A waiter woken by that notify
// could observe the lease still held. Cloning this handle costs nothing (it
// never touches the lease) and lets worker_main drop its Arc<Batch> first.
struct BatchCompletion {
    remaining: AtomicUsize,
    cancelled: AtomicBool,
    first_error: Mutex<Option<DeltafinError>>,
    lock: Mutex<()>,
    condvar: Condvar,
}

impl BatchCompletion {
    fn new(jobs: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(jobs),
            cancelled: AtomicBool::new(false),
            first_error: Mutex::new(None),
            lock: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }
}

struct Batch {
    sources: BatchSources,
    jobs: BatchJobs,
    vectored_reads: Option<Arc<Vec<VectoredRead>>>,
    verified_extents: Option<Arc<Vec<AtomicBool>>>,
    lease: Arc<BufferLeaseInner>,
    priority: ReadPriority,
    next_job: AtomicUsize,
    completion: Arc<BatchCompletion>,
    trace: Option<Box<BatchTrace>>,
    /// K3_SPLIT_READ only: per-source [internal, enclosure] descriptor cache
    /// so a file's chunk jobs share at most two opens instead of re-probing
    /// the directory chain per chunk (the v1 regression: 310k opens at
    /// 0.205 ms mean = 63.6 s/rung of worker time). Ok(None) = that home
    /// does not hold the file.
    chunk_home_files: Option<Box<[[HomeSlot; 2]]>>,
    /// K3_SPLIT_ETA only: per-source descriptor cache with one slot per tier
    /// directory (hot, dir_c, dir_b, primary) so a chunk can be homed on the
    /// tier with the least expected completion time.
    chunk_tier_files: Option<Box<[[HomeSlot; 4]]>>,
    /// Arrival-driven compute support (inline union batches only): chunks
    /// completed per source, and how many sources have every chunk done.
    /// Waiters on `completion.condvar` are woken whenever a source completes.
    source_chunks_done: Option<Box<[AtomicUsize]>>,
    sources_ready: AtomicUsize,
    /// K3_SPLIT_TRACE=1: batch identity and start stamp so each chunk's
    /// completion can be placed on the batch's own clock.
    batch_seq: u64,
    started_ns: u64,
}

type HomeSlot = OnceLock<Result<Option<File>>>;

/// Requeue: unclaimed jobs remain, push this Arc<Batch> back onto the queue.
/// Idle: nothing left to claim right now (someone else may still be finishing
/// it). Finished: this call's last job made `remaining` hit zero; the caller
/// must drop its own Arc<Batch> before notifying the enclosed handle.
enum QuantumOutcome {
    Requeue,
    Idle,
    Finished(Arc<BatchCompletion>),
}

impl Batch {
    fn new(plan: &ReadPlan, lease: Arc<BufferLeaseInner>, priority: ReadPriority) -> Self {
        let deferred_files = plan
            .sources
            .values
            .iter()
            .any(|source| source.file.is_none() && source.persistent_file.is_none())
            .then(|| {
                (0..plan.sources.values.len())
                    .map(|_| OnceLock::new())
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            });
        Self {
            sources: BatchSources::Plan {
                sources: Arc::clone(&plan.sources),
                deferred_files,
            },
            jobs: BatchJobs::Shared(Arc::clone(&plan.jobs)),
            vectored_reads: Some(Arc::clone(&plan.vectored_reads)),
            verified_extents: Some(Arc::clone(&plan.verified_extents)),
            lease,
            priority,
            next_job: AtomicUsize::new(0),
            completion: Arc::new(BatchCompletion::new(plan.jobs.len())),
            trace: BatchTrace::new_if_enabled(),
            chunk_home_files: None,
            chunk_tier_files: None,
            source_chunks_done: None,
            sources_ready: AtomicUsize::new(0),
            batch_seq: SPLIT_TRACE_BATCH_SEQ.fetch_add(1, Ordering::Relaxed),
            started_ns: stall_trace_now_ns(),
        }
    }

    fn new_deferred_exact_validated(
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        source_length: usize,
        lease: Arc<BufferLeaseInner>,
        priority: ReadPriority,
    ) -> Self {
        debug_assert!(!source_indices.is_empty());
        debug_assert!(source_indices.len() <= MAX_INLINE_DEFERRED_FILES);
        debug_assert!(
            source_indices
                .iter()
                .all(|&source| (source as usize) < catalog.inner.sources.len())
        );
        debug_assert_eq!(source_length as u64, catalog.inner.exact_source_length);
        let mut sources = [0_u32; MAX_INLINE_DEFERRED_FILES];
        sources[..source_indices.len()].copy_from_slice(source_indices);
        // K3_SPLIT_READ chunking applies only to page-multiple sources large
        // enough that a chunk stays a long sequential run; everything else
        // keeps the legacy single whole-file job per source.
        let requested_chunks = match priority {
            ReadPriority::Demand => split_read_demand_chunks(),
            ReadPriority::Prefetch => split_read_chunks(),
        };
        let (chunks_per_source, chunk_length, internal_chunks) = if requested_chunks > 1
            && source_length % BUFFER_ALIGNMENT == 0
            && source_length >= requested_chunks * 2 * 1024 * 1024
        {
            let raw = source_length.div_ceil(requested_chunks);
            let chunk_length = raw.next_multiple_of(BUFFER_ALIGNMENT);
            let chunks = source_length.div_ceil(chunk_length);
            if chunks > 1 {
                // Demand batches stay home-preserving: every chunk keeps the
                // legacy internal-first probe order. Only prefetch batches
                // (K3_SPLIT_READ, measured negative) ever split homes.
                let internal = match priority {
                    ReadPriority::Demand => chunks,
                    ReadPriority::Prefetch => split_read_internal_chunks(chunks),
                };
                (chunks, chunk_length, internal)
            } else {
                (1, source_length, 1)
            }
        } else {
            (1, source_length, 1)
        };
        Self {
            sources: BatchSources::DeferredCatalog(Arc::clone(&catalog.inner)),
            jobs: BatchJobs::Inline(InlineReadJobs {
                sources,
                len: source_indices.len(),
                destination,
                source_length,
                chunks_per_source,
                chunk_length,
                internal_chunks,
            }),
            vectored_reads: None,
            // Raw-v1 is guarded by exact length/type/name contracts rather
            // than an authenticated pack digest. Scale4-v2 must use its
            // separate manifest/hash path and never enters this constructor.
            verified_extents: None,
            lease,
            priority,
            next_job: AtomicUsize::new(0),
            completion: Arc::new(BatchCompletion::new(
                source_indices.len() * chunks_per_source,
            )),
            trace: BatchTrace::new_if_enabled(),
            chunk_home_files: (chunks_per_source > 1).then(|| {
                (0..source_indices.len())
                    .map(|_| [HomeSlot::new(), HomeSlot::new()])
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            }),
            chunk_tier_files: (chunks_per_source > 1 && split_eta_enabled()).then(|| {
                (0..source_indices.len())
                    .map(|_| [HomeSlot::new(), HomeSlot::new(), HomeSlot::new(), HomeSlot::new()])
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            }),
            source_chunks_done: Some(
                (0..source_indices.len())
                    .map(|_| AtomicUsize::new(0))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
            sources_ready: AtomicUsize::new(0),
            batch_seq: SPLIT_TRACE_BATCH_SEQ.fetch_add(1, Ordering::Relaxed),
            started_ns: stall_trace_now_ns(),
        }
    }

    /// Run at most one scheduling quantum. `Finished` carries the completion
    /// handle the caller must notify -- but only after it has dropped its own
    /// Arc<Batch>, so the notified waiter never observes this batch's arena
    /// lease still held by the worker that just finished it.
    fn run_quantum(&self) -> QuantumOutcome {
        for _ in 0..WORK_QUANTUM {
            let index = self.next_job.fetch_add(1, Ordering::Relaxed);
            let Some(job) = self.jobs.get(index) else {
                return QuantumOutcome::Idle;
            };
            let trace_claim_ns = self.trace.as_deref().map(|_| stall_trace_now_ns());
            if let Err(error) = self.run_job(job) {
                let mut first_error = self.completion.first_error.lock().unwrap();
                if first_error.is_none() {
                    *first_error = Some(error);
                }
            }
            if let (Some(trace), Some(claim_ns)) = (self.trace.as_deref(), trace_claim_ns) {
                trace.record_job(claim_ns, stall_trace_now_ns());
            }
            self.note_source_progress(index);
            if self.completion.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                return QuantumOutcome::Finished(Arc::clone(&self.completion));
            }
        }
        if self.next_job.load(Ordering::Relaxed) < self.jobs.len() {
            QuantumOutcome::Requeue
        } else {
            QuantumOutcome::Idle
        }
    }

    /// Record one finished chunk job for its source (inline union batches:
    /// file-major striping, so job `index` belongs to source `index % len`).
    /// When a source's last chunk lands, bump `sources_ready` and wake waiters.
    fn note_source_progress(&self, index: usize) {
        let (Some(done), BatchJobs::Inline(jobs)) = (self.source_chunks_done.as_deref(), &self.jobs)
        else {
            return;
        };
        if jobs.len == 0 {
            return;
        }
        let source = index % jobs.len;
        if let Some(counter) = done.get(source)
            && counter.fetch_add(1, Ordering::AcqRel) + 1 == jobs.chunks_per_source
        {
            self.sources_ready.fetch_add(1, Ordering::AcqRel);
            let _guard = self.completion.lock.lock().unwrap();
            self.completion.condvar.notify_all();
        }
    }

    /// Indices (into the batch's source list) whose every chunk has landed.
    fn ready_sources(&self) -> Vec<usize> {
        let (Some(done), BatchJobs::Inline(jobs)) = (self.source_chunks_done.as_deref(), &self.jobs)
        else {
            // No per-source tracking on this batch kind: everything is ready
            // exactly when the whole read has completed.
            if self.completion.remaining.load(Ordering::Acquire) == 0 {
                let n = match &self.jobs {
                    BatchJobs::Inline(jobs) => jobs.len,
                    BatchJobs::Shared(jobs) => jobs.len(),
                };
                return (0..n).collect();
            }
            return Vec::new();
        };
        done.iter()
            .enumerate()
            .filter(|(_, c)| c.load(Ordering::Acquire) >= jobs.chunks_per_source)
            .map(|(i, _)| i)
            .collect()
    }

    /// Block until at least `minimum` sources are complete, every job has
    /// finished, or the batch was cancelled. Returns the ready count.
    fn wait_ready_sources(&self, minimum: usize) -> usize {
        let mut guard = self.completion.lock.lock().unwrap();
        loop {
            let ready = self.sources_ready.load(Ordering::Acquire);
            if ready >= minimum
                || self.completion.remaining.load(Ordering::Acquire) == 0
                || self.completion.cancelled.load(Ordering::Acquire)
            {
                return ready;
            }
            guard = self.completion.condvar.wait(guard).unwrap();
        }
    }

    /// Atomically claim every job that no worker has started. Jobs already in
    /// `pread` finish normally; their arena lease remains owned by this batch
    /// until the last active worker publishes completion. This is the exact
    /// cancellation primitive needed by speculative I/O: no recycled buffer
    /// can race an in-flight kernel or read.
    fn cancel_unclaimed(&self) {
        self.completion.cancelled.store(true, Ordering::Release);
        let jobs = self.jobs.len();
        loop {
            let next = self.next_job.load(Ordering::Acquire);
            if next >= jobs {
                break;
            }
            if self
                .next_job
                .compare_exchange(next, jobs, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let cancelled = jobs - next;
                if self.completion.remaining.fetch_sub(cancelled, Ordering::AcqRel) == cancelled {
                    let _guard = self.completion.lock.lock().unwrap();
                    self.completion.condvar.notify_all();
                }
                break;
            }
        }
    }

    fn run_job(&self, job: ReadJob) -> Result<()> {
        if let JobSource::Vectored { source, scatter } = job.source {
            return self.run_vectored_read(source, scatter, job.expected_digest);
        }
        if let JobSource::AuthenticatedScatter { source } = job.source {
            return self.run_authenticated_scatter(source);
        }
        // SAFETY: `ReadPlan::open` rejected every overlapping destination and
        // bounded every range. The batch does not publish immutable access
        // until `remaining` reaches zero.
        let pointer = self
            .lease
            .buffers()
            .get(job.destination)
            .pointer_at(job.destination_offset, job.length);
        // SAFETY: `ReadPlan::open` rejected overlapping destinations and
        // bounded the range. No immutable view is exposed until completion.
        let destination = unsafe { std::slice::from_raw_parts_mut(pointer, job.length) };
        // Snapshot qualification before starting I/O. If two first-use
        // batches race, both authenticate the bytes they individually read;
        // one completed batch cannot retroactively qualify the other's
        // already-in-flight read.
        let verify_after_read = match job.verification_index {
            Some(index) => {
                let verified = self
                    .verified_extents
                    .as_ref()
                    .and_then(|extents| extents.get(index))
                    .ok_or_else(|| {
                        DeltafinError::new("read job refers to an unknown verification slot")
                    })?;
                !verified.load(Ordering::Acquire)
            }
            None => false,
        };
        match job.source {
            JobSource::Zero => {
                destination.fill(0);
            }
            JobSource::File {
                source,
                source_offset,
            } => {
                let BatchSources::Plan {
                    sources,
                    deferred_files,
                } = &self.sources
                else {
                    return Err(DeltafinError::new(
                        "file read job is attached to the wrong source set",
                    ));
                };
                let source_index = source;
                let source = sources
                    .values
                    .get(source_index)
                    .ok_or_else(|| DeltafinError::new("read job refers to an unknown source"))?;
                if !source.verifications.is_empty() {
                    return Err(DeltafinError::new(
                        "authenticated source was incorrectly routed through a double-read file job",
                    ));
                }
                let file = if let Some(file) = source.file.as_ref() {
                    file
                } else if let Some(slot) = source.persistent_file.as_ref() {
                    match slot.get_or_init(|| open_deferred_source(source)) {
                        Ok(file) => file,
                        Err(error) => return Err(error.clone()),
                    }
                } else {
                    let slot = deferred_files
                        .as_ref()
                        .and_then(|files| files.get(source_index))
                        .ok_or_else(|| {
                            DeltafinError::new("deferred read source has no batch descriptor slot")
                        })?;
                    match slot.get_or_init(|| open_deferred_source(source)) {
                        Ok(file) => file,
                        Err(error) => return Err(error.clone()),
                    }
                };
                let plan_read_started = std::time::Instant::now();
                let mut completed = 0_usize;
                while completed < destination.len() {
                    let count = match file.read_at(
                        &mut destination[completed..],
                        source_offset + completed as u64,
                    ) {
                        Ok(count) => count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => return Err(io_error("pread", &source.path, error)),
                    };
                    EXPERT_READ_BYTES.fetch_add(count as u64, Ordering::Relaxed);
                    if count == 0 {
                        return Err(DeltafinError::new(format!(
                            "short pread {}/{} from {} at {}",
                            completed,
                            destination.len(),
                            source.path.display(),
                            source_offset
                        )));
                    }
                    completed += count;
                }
                if plan_balance_enabled() {
                    plan_retire_path(&source.path);
                }
                // Device record for the per-read provenance trace (2026-09-06):
                // this plan-path read carried 45% of a run's expert bytes and
                // was invisible to the trace. Device from the source's volume,
                // layer/expert from its file name, priority unknown ("-"), and
                // the whole job length is one record (bytes = full file unless
                // the plan chunked it).
                if trace_enabled() {
                    let name = source.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if let Some((layer, expert)) = parse_expert_name(name) {
                        let path_str = source.path.to_string_lossy();
                        let code = if path_str.starts_with("/Volumes/Yellow") {
                            SRC_K3A
                        } else if path_str.starts_with("/Volumes/Green") {
                            SRC_K3B
                        } else if path_str.starts_with("/Volumes/White") {
                            SRC_K3C
                        } else {
                            SRC_INTERNAL
                        };
                        trace_read_prio(
                            TRACE_BARRIER.load(Ordering::Relaxed),
                            layer,
                            expert,
                            code,
                            plan_read_started.elapsed().as_nanos() as u64,
                            Some(matches!(self.priority, ReadPriority::Prefetch)),
                            source_offset,
                            destination.len() as u64,
                        );
                    }
                }
                if let (true, Some(expected), Some(index)) = (
                    verify_after_read,
                    job.expected_digest,
                    job.verification_index,
                ) {
                    let verified = self
                        .verified_extents
                        .as_ref()
                        .and_then(|extents| extents.get(index))
                        .ok_or_else(|| {
                            DeltafinError::new("read job refers to an unknown verification slot")
                        })?;
                    let actual = crate::packfile::digest_bytes(destination);
                    if actual != expected {
                        return Err(DeltafinError::new(format!(
                            "authenticated read from {} at {} failed SHA-256 verification",
                            source.path.display(),
                            source_offset,
                        )));
                    }
                    verified.store(true, Ordering::Release);
                }
                drop_completed_cache(file, source.cache_policy, source_offset, job.length);
            }
            JobSource::AuthenticatedScatter { .. } => unreachable!(
                "authenticated scatter jobs are handled before creating one destination slice"
            ),
            JobSource::Vectored { .. } => unreachable!(
                "vectored jobs are handled before creating one contiguous destination slice"
            ),
            JobSource::DeferredCatalog {
                source,
                source_offset,
                prefer_internal_home,
                batch_slot,
            } => {
                let BatchSources::DeferredCatalog(catalog) = &self.sources else {
                    return Err(DeltafinError::new(
                        "catalog read job is attached to the wrong source set",
                    ));
                };
                let source_name = catalog.sources.get(source as usize).ok_or_else(|| {
                    DeltafinError::new("catalog read job refers to an unknown source")
                })?;
                // Chunk jobs share per-(source, home) descriptors cached on
                // the batch; legacy whole-file jobs keep a fresh open with
                // the legacy probe order.
                let legacy_file;
                let mut eta_guard: Option<SplitEtaGuard> = None;
                let file: &File = if let (Some(tiers), true) =
                    (self.chunk_tier_files.as_deref(), prefer_internal_home.is_some())
                {
                    // K3_SPLIT_ETA: home this chunk on the tier directory that
                    // can finish it first — (in-flight chunks + 1) x chunk cost
                    // at that tier's configured rate — probing tiers in ETA
                    // order and taking the first that holds the file.
                    let slots = tiers.get(batch_slot as usize).ok_or_else(|| {
                        DeltafinError::new("chunk read job refers to an unknown batch slot")
                    })?;
                    let mut order = [0_usize, 1, 2, 3];
                    let bytes = destination.len() as u64;
                    if let Some(caps) = split_caps() {
                        // K3_SPLIT_CAP: prefer the fastest tier whose in-flight
                        // depth is under its cap (primary first, then dir_b,
                        // dir_c, hot); tiers at cap fall back to ETA order.
                        // Enclosure queues stay short (bounded tails) and the
                        // internal absorbs the overflow.
                        order.sort_by_key(|&tier| {
                            let queued = SPLIT_ETA_OUTSTANDING[tier].0.load(Ordering::Relaxed);
                            let under_cap = queued < caps[tier];
                            (!under_cap, if under_cap { 3 - tier as u64 } else { split_eta_us(tier, bytes) })
                        });
                    } else {
                        order.sort_by_key(|&tier| split_eta_us(tier, bytes));
                    }
                    let mut chosen: Option<(&File, usize)> = None;
                    for tier in order {
                        match slots[tier].get_or_init(|| open_catalog_tier(catalog, source_name, tier)) {
                            Err(error) => return Err(error.clone()),
                            Ok(Some(file)) => {
                                chosen = Some((file, tier));
                                break;
                            }
                            Ok(None) => continue,
                        }
                    }
                    let Some((file, tier)) = chosen else {
                        return Err(DeltafinError::new(format!(
                            "deferred source {}/{} missing from every tier",
                            catalog.directory_path.display(),
                            source_name.as_str(),
                        )));
                    };
                    eta_guard = Some(SplitEtaGuard::issue(tier));
                    file
                } else if let (Some(cache), Some(prefer_internal)) =
                    (self.chunk_home_files.as_deref(), prefer_internal_home)
                {
                    let slots = cache.get(batch_slot as usize).ok_or_else(|| {
                        DeltafinError::new("chunk read job refers to an unknown batch slot")
                    })?;
                    let preferred_index = if prefer_internal { 0 } else { 1 };
                    let preferred = slots[preferred_index]
                        .get_or_init(|| open_catalog_home(catalog, source_name, prefer_internal));
                    match preferred {
                        Err(error) => return Err(error.clone()),
                        Ok(Some(file)) => file,
                        Ok(None) => {
                            let fallback = slots[1 - preferred_index].get_or_init(|| {
                                open_catalog_home(catalog, source_name, !prefer_internal)
                            });
                            match fallback {
                                Err(error) => return Err(error.clone()),
                                Ok(Some(file)) => file,
                                Ok(None) => {
                                    return Err(DeltafinError::new(format!(
                                        "deferred source {}/{} missing from every home",
                                        catalog.directory_path.display(),
                                        source_name.as_str(),
                                    )));
                                }
                            }
                        }
                    }
                } else {
                    legacy_file =
                        open_deferred_catalog_source(catalog, source_name, prefer_internal_home)?;
                    &legacy_file
                };
                let read_started = std::time::Instant::now();
                advise_read_range(file, source_offset, destination.len());
                let mut completed = 0_usize;
                while completed < destination.len() {
                    let count = match file.read_at(
                        &mut destination[completed..],
                        source_offset + completed as u64,
                    ) {
                        Ok(count) => count,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => {
                            return Err(catalog_io_error("pread", catalog, source_name, error));
                        }
                    };
                    EXPERT_READ_BYTES.fetch_add(count as u64, Ordering::Relaxed);
                    if count == 0 {
                        return Err(DeltafinError::new(format!(
                            "short pread {}/{} from {}/{} at {}",
                            completed,
                            destination.len(),
                            catalog.directory_path.display(),
                            source_name.as_str(),
                            source_offset,
                        )));
                    }
                    completed += count;
                }
                drop(eta_guard);
                if prefer_internal_home.is_some() && (split_trace_enabled() || trace_enabled()) {
                    let tier = split_fd_tier(file.as_raw_fd());
                    let read_ns = read_started.elapsed().as_nanos() as u64;
                    if split_trace_enabled() {
                        split_trace_record(SplitTraceRecord {
                            batch: self.batch_seq,
                            slot: batch_slot,
                            chunk_offset: source_offset,
                            tier,
                            prefetch: matches!(self.priority, ReadPriority::Prefetch),
                            done_ns: stall_trace_now_ns().saturating_sub(self.started_ns),
                            read_ns,
                        });
                    }
                    // Device record for the per-read provenance trace: chunked
                    // reads open through the per-batch descriptor cache, so the
                    // legacy open-site record never sees them. One record per
                    // chunk, tier mapped onto the legacy source codes, duration
                    // = this chunk's transfer (a replay needs file + device).
                    if trace_enabled() {
                        if let Some((layer, expert)) = parse_expert_name(source_name.as_str()) {
                            let source = match tier {
                                0 => SRC_K3B,
                                1 => SRC_K3A,
                                2 => SRC_K3C,
                                3 => SRC_INTERNAL,
                                _ => SRC_READ,
                            };
                            trace_read_prio(
                                TRACE_BARRIER.load(Ordering::Relaxed),
                                layer,
                                expert,
                                source,
                                read_ns,
                                Some(matches!(self.priority, ReadPriority::Prefetch)),
                                source_offset,
                                job.length as u64,
                            );
                        }
                    }
                }
                // The transfer itself — this is the duration that sets the
                // barrier tail. The open-site record times openat() only, which
                // is ~0.1 ms against a ~2 ms transfer, so a tail computed from
                // opens describes descriptor resolution and not the read the
                // layer is actually waiting on.
                if trace_enabled() {
                    if let Some((layer, expert)) = parse_expert_name(source_name.as_str()) {
                        trace_read(
                            TRACE_BARRIER.load(Ordering::Relaxed),
                            layer,
                            expert,
                            SRC_READ,
                            read_started.elapsed().as_nanos() as u64,
                        );
                    }
                }
                // The record has landed: this device no longer owes it. Placed
                // after the transfer rather than after the open, because the
                // open returns in ~0.1 ms while the 17.5 MB read takes ~2 ms —
                // retiring at open would make every device look idle and the
                // policy would collapse back to a static order.
                mirror_retire();
                drop_completed_cache(file, catalog.cache_policy, source_offset, job.length);
            }
        }
        Ok(())
    }

    fn run_vectored_read(
        &self,
        source_index: usize,
        scatter_index: usize,
        expected_digest: Option<[u8; 32]>,
    ) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
        } = &self.sources
        else {
            return Err(DeltafinError::new(
                "vectored read job is attached to the wrong source set",
            ));
        };
        let source = sources
            .values
            .get(source_index)
            .ok_or_else(|| DeltafinError::new("vectored job refers to an unknown source"))?;
        if !source.verifications.is_empty() {
            return Err(DeltafinError::new(
                "authenticated source was incorrectly routed through a vectored read job",
            ));
        }
        let scatter = self
            .vectored_reads
            .as_ref()
            .and_then(|reads| reads.get(scatter_index))
            .ok_or_else(|| DeltafinError::new("vectored job refers to an unknown scatter"))?;
        if scatter.destinations.is_empty() || scatter.destinations.len() > MAX_VECTORED_DESTINATIONS
        {
            return Err(DeltafinError::new(
                "vectored job has an invalid destination count",
            ));
        }
        let file = if let Some(file) = source.file.as_ref() {
            file
        } else if let Some(slot) = source.persistent_file.as_ref() {
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        } else {
            let slot = deferred_files
                .as_ref()
                .and_then(|files| files.get(source_index))
                .ok_or_else(|| {
                    DeltafinError::new("vectored read source has no batch descriptor slot")
                })?;
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        };

        // A fixed stack table avoids an allocation for every routed expert.
        // validate_destinations proved these arena regions are bounded and
        // globally disjoint; the batch remains private until every job ends.
        let mut vectors: [libc::iovec; MAX_VECTORED_DESTINATIONS] = unsafe { std::mem::zeroed() };
        let mut total = 0_usize;
        for (vector, destination) in vectors.iter_mut().zip(scatter.destinations.iter()) {
            let pointer = self
                .lease
                .buffers()
                .get(destination.destination)
                .pointer_at(destination.destination_offset, destination.length);
            vector.iov_base = pointer.cast();
            vector.iov_len = destination.length;
            total = total
                .checked_add(destination.length)
                .ok_or_else(|| DeltafinError::new("vectored read length overflows usize"))?;
        }
        if total > isize::MAX as usize {
            return Err(DeltafinError::new(
                "vectored read exceeds the platform syscall length",
            ));
        }
        let mut first = 0_usize;
        let mut remaining = total;
        let mut file_offset = i64::try_from(scatter.source_offset)
            .map_err(|_| DeltafinError::new("vectored source offset exceeds off_t"))?;
        while remaining != 0 {
            let count = i32::try_from(scatter.destinations.len() - first)
                .map_err(|_| DeltafinError::new("vectored destination count exceeds c_int"))?;
            // SAFETY: every iovec points into a live, private, prevalidated
            // arena range. `file` remains open for the syscall, `count` is
            // bounded, and preadv does not retain either pointer or descriptor.
            let read = unsafe {
                libc::preadv(
                    file.as_raw_fd(),
                    vectors[first..].as_ptr(),
                    count,
                    file_offset,
                )
            };
            if read == -1 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(io_error("preadv", &source.path, error));
            }
            if read == 0 {
                return Err(DeltafinError::new(format!(
                    "short preadv {}/{} from {} at {}",
                    total - remaining,
                    total,
                    source.path.display(),
                    scatter.source_offset,
                )));
            }
            let read = usize::try_from(read)
                .map_err(|_| DeltafinError::new("preadv returned a negative byte count"))?;
            if read > remaining {
                return Err(DeltafinError::new("preadv exceeded its destination length"));
            }
            remaining -= read;
            EXPERT_READ_BYTES.fetch_add(read as u64, Ordering::Relaxed);
            file_offset = file_offset
                .checked_add(read as i64)
                .ok_or_else(|| DeltafinError::new("vectored source offset overflows off_t"))?;

            let mut consumed = read;
            while consumed != 0 {
                let length = vectors[first].iov_len;
                if consumed >= length {
                    consumed -= length;
                    first += 1;
                } else {
                    // SAFETY: `consumed < iov_len`, so advancing the pointer
                    // remains inside the same validated destination range.
                    vectors[first].iov_base =
                        unsafe { vectors[first].iov_base.cast::<u8>().add(consumed).cast() };
                    vectors[first].iov_len -= consumed;
                    consumed = 0;
                }
            }
        }
        if let Some(expected) = expected_digest {
            // Unlike persistent ordinary extents, deferred vectored
            // sources can reopen on every batch. Hash every verified job;
            // higher-level immutable identity caches omit the digest from
            // later plans instead of storing a plan-local qualification
            // that might cross descriptor generations.
            let mut digest = DigestState::new();
            for destination in &scatter.destinations {
                let pointer = self
                    .lease
                    .buffers()
                    .get(destination.destination)
                    .pointer_at(destination.destination_offset, destination.length);
                // SAFETY: destination validation proved this complete
                // range is bounded. Its read job has finished filling it,
                // and every other in-flight job owns a disjoint range.
                let bytes =
                    unsafe { std::slice::from_raw_parts(pointer.cast_const(), destination.length) };
                digest.update(bytes);
            }
            if digest.finalize() != expected {
                return Err(DeltafinError::new(format!(
                    "authenticated vectored read from {} at {} failed SHA-256 verification",
                    source.path.display(),
                    scatter.source_offset,
                )));
            }
        }
        drop_completed_cache(file, source.cache_policy, scatter.source_offset, total);
        Ok(())
    }

    fn run_authenticated_scatter(&self, source_index: usize) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
            ..
        } = &self.sources
        else {
            return Err(DeltafinError::new(
                "authenticated scatter job is attached to the wrong source set",
            ));
        };
        let source = sources
            .values
            .get(source_index)
            .ok_or_else(|| DeltafinError::new("scatter job refers to an unknown source"))?;
        if source.verifications.is_empty() || source.scatter_extents.is_empty() {
            return Err(DeltafinError::new(
                "authenticated scatter source has an incomplete plan",
            ));
        }
        let file = if let Some(file) = source.file.as_ref() {
            file
        } else if let Some(slot) = source.persistent_file.as_ref() {
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        } else {
            let slot = deferred_files
                .as_ref()
                .and_then(|files| files.get(source_index))
                .ok_or_else(|| {
                    DeltafinError::new("authenticated scatter source has no batch descriptor slot")
                })?;
            match slot.get_or_init(|| open_deferred_source(source)) {
                Ok(file) => file,
                Err(error) => return Err(error.clone()),
            }
        };
        authenticate_and_scatter_source(file, source, &self.lease)
    }

    fn wait_for_completion(&self) {
        let mut guard = self.completion.lock.lock().unwrap();
        while self.completion.remaining.load(Ordering::Acquire) != 0 {
            guard = self.completion.condvar.wait(guard).unwrap();
        }
        drop(guard);
    }

    fn validate_deferred_source_identities(&self) -> Result<()> {
        let BatchSources::Plan {
            sources,
            deferred_files,
        } = &self.sources
        else {
            return Ok(());
        };
        for (source_index, source) in sources.values.iter().enumerate() {
            let Some(expected) = source.expected_identity else {
                continue;
            };
            let actual = if let Some(file) = source.file.as_ref() {
                descriptor_identity(file, &source.path)?
            } else if let Some(slot) = source.persistent_file.as_ref() {
                match slot.get() {
                    Some(Ok(file)) => descriptor_identity(file, &source.path)?,
                    Some(Err(error)) => return Err(error.clone()),
                    None => {
                        return Err(DeltafinError::new(
                            "identity-pinned persistent source was never opened",
                        ));
                    }
                }
            } else {
                let slot = deferred_files
                    .as_ref()
                    .and_then(|files| files.get(source_index))
                    .ok_or_else(|| {
                        DeltafinError::new("identity-pinned deferred source has no descriptor slot")
                    })?;
                match slot.get() {
                    Some(Ok(file)) => descriptor_identity(file, &source.path)?,
                    Some(Err(error)) => return Err(error.clone()),
                    None => {
                        return Err(DeltafinError::new(
                            "identity-pinned deferred source was never opened",
                        ));
                    }
                }
            };
            if actual != expected {
                return Err(DeltafinError::new(format!(
                    "deferred source identity changed during range gather: {}",
                    source.path.display(),
                )));
            }
        }
        Ok(())
    }

    fn wait(&self) -> Result<()> {
        self.wait_for_completion();
        if self.completion.cancelled.load(Ordering::Acquire) {
            return Err(DeltafinError::new("storage read ticket was cancelled"));
        }
        if let Some(error) = self.completion.first_error.lock().unwrap().clone() {
            return Err(error);
        }
        self.validate_deferred_source_identities()
    }
}

pub struct ReadTicket {
    batch: Arc<Batch>,
    started: Instant,
    bytes: u64,
    jobs: usize,
    workers: usize,
}

impl ReadTicket {
    pub fn is_ready(&self) -> bool {
        self.batch.completion.remaining.load(Ordering::Acquire) == 0
    }

    /// Arrival-driven compute: wait until `minimum` sources of this union read
    /// have every chunk landed (or the read finished / was cancelled) and
    /// return the indices of the sources that are complete right now.
    pub fn wait_ready_sources(&self, minimum: usize) -> Vec<usize> {
        self.batch.wait_ready_sources(minimum);
        self.batch.ready_sources()
    }

    /// Indices of the sources whose every chunk has landed, without waiting.
    pub fn ready_sources(&self) -> Vec<usize> {
        self.batch.ready_sources()
    }

    /// Surface the first read error recorded so far without consuming the ticket.
    pub fn check_error(&self) -> Result<()> {
        match self.batch.completion.first_error.lock().unwrap().as_ref() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Borrow the `other` slab while the read is still in flight (arrival-
    /// driven compute). Only byte ranges of sources reported by
    /// `ready_sources` are complete; the lease stays owned by this ticket.
    pub fn peek_other(&self) -> &[u8] {
        self.batch
            .lease
            .buffers()
            .get(BufferKind::Other)
            .as_slice(self.batch.lease.lengths.other)
    }

    pub fn wait(self) -> Result<(LayerBuffers, ReadStats)> {
        let trace_wait = self.batch.trace.as_deref().map(|_| {
            (
                stall_trace_now_ns(),
                self.batch.completion.remaining.load(Ordering::Acquire) == 0,
            )
        });
        self.batch.wait()?;
        if let (Some(trace), Some((wait_start_ns, was_ready))) =
            (self.batch.trace.as_deref(), trace_wait)
        {
            stall_trace_record(
                self.batch.priority,
                wait_start_ns,
                stall_trace_now_ns(),
                was_ready,
                trace,
            );
        }
        let buffers = LayerBuffers {
            lease: Arc::clone(&self.batch.lease),
        };
        Ok((
            buffers,
            ReadStats {
                bytes: self.bytes,
                jobs: self.jobs,
                workers: self.workers,
                elapsed: self.started.elapsed(),
            },
        ))
    }

    /// Cancel every unclaimed job and wait only for work already inside an I/O
    /// syscall. The ticket publishes no bytes; dropping it then releases its
    /// arena slot. Optional callers deliberately ignore read errors here
    /// because authoritative demand I/O will retry any selected expert.
    pub fn cancel_and_wait(self) {
        self.batch.cancel_unclaimed();
        self.batch.wait_for_completion();
    }

    pub fn cancel_unclaimed(&self) {
        self.batch.cancel_unclaimed();
    }

    pub fn drain_cancelled(self) {
        self.batch.wait_for_completion();
    }
}

impl Drop for ReadTicket {
    fn drop(&mut self) {
        self.batch.cancel_unclaimed();
    }
}

struct PriorityQueues<T> {
    demand: VecDeque<T>,
    prefetch: VecDeque<T>,
    demand_streak: usize,
}

impl<T> PriorityQueues<T> {
    fn new() -> Self {
        Self {
            demand: VecDeque::new(),
            prefetch: VecDeque::new(),
            demand_streak: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.demand.is_empty() && self.prefetch.is_empty()
    }

    fn push(&mut self, priority: ReadPriority, value: T) {
        match priority {
            ReadPriority::Demand => self.demand.push_back(value),
            ReadPriority::Prefetch => self.prefetch.push_back(value),
        }
    }

    fn pop(&mut self) -> Option<T> {
        if !self.demand.is_empty()
            && (self.prefetch.is_empty() || self.demand_streak < MAX_DEMAND_STREAK)
        {
            self.demand_streak = self.demand_streak.saturating_add(1);
            return self.demand.pop_front();
        }
        if let Some(value) = self.prefetch.pop_front() {
            self.demand_streak = 0;
            return Some(value);
        }
        self.demand_streak = self.demand_streak.saturating_add(1);
        self.demand.pop_front()
    }
}

struct PoolState {
    inner: Mutex<PoolInner>,
    available: Condvar,
}

struct PoolInner {
    queues: PriorityQueues<Arc<Batch>>,
    closed: bool,
}

pub struct Reader {
    state: Arc<PoolState>,
    arena: Arc<BufferArena>,
    threads: Vec<JoinHandle<()>>,
}

impl Reader {
    pub fn new(workers: usize) -> Result<Self> {
        Self::with_arena_capacity(workers, DEFAULT_ARENA_SLOTS)
    }

    pub fn with_arena_capacity(workers: usize, arena_slots: usize) -> Result<Self> {
        Self::with_arena_capacity_and_retire_hook(workers, arena_slots, None)
    }

    pub(crate) fn with_arena_capacity_and_retire_hook(
        workers: usize,
        arena_slots: usize,
        retire_hook: Option<BufferRetireHook>,
    ) -> Result<Self> {
        Self::with_arena_capacity_retire_hook_and_class(workers, arena_slots, retire_hook, false)
    }

    /// `prefetch_class: true` marks this Reader's workers as speculative I/O,
    /// eligible for K3_PREFETCH_IOPOL kernel deprioritization. Demand readers
    /// pass false and are untouched.
    pub(crate) fn with_arena_capacity_retire_hook_and_class(
        workers: usize,
        arena_slots: usize,
        retire_hook: Option<BufferRetireHook>,
        prefetch_class: bool,
    ) -> Result<Self> {
        if workers == 0 {
            return Err(DeltafinError::new(
                "storage reader needs at least one worker",
            ));
        }
        crate::io_priority::configure_process_for_model_io();
        let state = Arc::new(PoolState {
            inner: Mutex::new(PoolInner {
                queues: PriorityQueues::new(),
                closed: false,
            }),
            available: Condvar::new(),
        });
        let arena = BufferArena::new_with_retire_hook(arena_slots, retire_hook)?;
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(workers);
        for index in 0..workers {
            let worker_state = Arc::clone(&state);
            let handle = match thread::Builder::new()
                .name(format!("deltafin-io-{index}"))
                .spawn(move || worker_main(worker_state, prefetch_class))
            {
                Ok(handle) => handle,
                Err(error) => {
                    state.inner.lock().unwrap().closed = true;
                    state.available.notify_all();
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(DeltafinError::new(format!("start I/O worker: {error}")));
                }
            };
            threads.push(handle);
        }
        Ok(Self {
            state,
            arena,
            threads,
        })
    }

    pub fn workers(&self) -> usize {
        self.threads.len()
    }

    pub(crate) fn replacement_admission_bytes(&self, lengths: BufferLengths) -> Result<u64> {
        self.arena.replacement_admission_bytes(lengths)
    }

    pub(crate) fn reserve_capacity(&self, lengths: BufferLengths) -> Result<()> {
        self.arena.reserve_capacity(lengths)
    }

    pub fn read(&self, plan: &ReadPlan) -> Result<(LayerBuffers, ReadStats)> {
        self.submit(plan, ReadPriority::Demand)?.wait()
    }

    /// Admit a read into the bounded arena and return while its I/O is in
    /// flight. Admission waits when every slot is still leased by a caller.
    pub fn submit(&self, plan: &ReadPlan, priority: ReadPriority) -> Result<ReadTicket> {
        self.submit_inner(plan, priority, true)?.ok_or_else(|| {
            DeltafinError::new("blocking storage submission unexpectedly found no arena slot")
        })
    }

    /// Non-blocking admission for callers that need to apply their own
    /// backpressure. `None` means every bounded arena slot is currently leased.
    pub fn try_submit(
        &self,
        plan: &ReadPlan,
        priority: ReadPriority,
    ) -> Result<Option<ReadTicket>> {
        self.submit_inner(plan, priority, false)
    }

    /// Submit up to sixteen catalogued whole files into adjacent slots of one
    /// destination slab without compiling a `ReadPlan`.
    ///
    /// This is the routed-decode fast path: catalog paths and directory
    /// ownership are immutable session state, jobs live inline in `Batch`, and
    /// source selection is a fixed array of integer indices. The only heap
    /// ownership admitted per read is the shared batch/ticket required by the
    /// persistent worker pool and the already-bounded arena lease.
    pub fn submit_deferred_exact(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
    ) -> Result<ReadTicket> {
        self.submit_deferred_exact_inner(catalog, source_indices, destination, priority, true)?
            .ok_or_else(|| {
                DeltafinError::new("blocking catalog submission unexpectedly found no arena slot")
            })
    }

    /// Non-blocking deferred-exact admission. `None` means every eligible
    /// arena slot is currently leased. Speculative submitters (pilot plans,
    /// true-route top-ups) MUST use this: with `wait=true` the single decode
    /// thread can park on the arena condvar waiting for slots held by
    /// speculative tickets only it can ever claim or drain — a permanent
    /// self-deadlock (observed post-cold-boot at race-scattered tokens once
    /// the top-up path exceeded the sized 32-current + 32-next invariant).
    pub fn try_submit_deferred_exact(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
    ) -> Result<Option<ReadTicket>> {
        self.submit_deferred_exact_inner(catalog, source_indices, destination, priority, false)
    }

    /// Wait-flag passthrough for callers deciding blocking behavior at
    /// runtime (authoritative unions block; speculative ones must not).
    pub fn try_submit_deferred_exact_with_wait(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
        wait_for_slot: bool,
    ) -> Result<Option<ReadTicket>> {
        self.submit_deferred_exact_inner(
            catalog,
            source_indices,
            destination,
            priority,
            wait_for_slot,
        )
    }

    /// Wait-flag passthrough over the planned-submission path; same
    /// contract as `try_submit_deferred_exact_with_wait`.
    pub fn try_submit_with_wait(
        &self,
        plan: &ReadPlan,
        priority: ReadPriority,
        wait_for_slot: bool,
    ) -> Result<Option<ReadTicket>> {
        self.submit_inner(plan, priority, wait_for_slot)
    }

    fn submit_deferred_exact_inner(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
        wait_for_slot: bool,
    ) -> Result<Option<ReadTicket>> {
        let (lengths, source_length) =
            deferred_batch_lengths(catalog, source_indices, destination)?;
        let logical_bytes = lengths.get(destination) as u64;
        let started = Instant::now();
        {
            let inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
        }
        let Some(lease) = self.arena.acquire(lengths, wait_for_slot, priority)? else {
            return Ok(None);
        };
        let batch = Arc::new(Batch::new_deferred_exact_validated(
            catalog,
            source_indices,
            destination,
            source_length,
            lease,
            priority,
        ));
        let participating = self.workers().min(batch.jobs.len());
        let mut inner = self.state.inner.lock().unwrap();
        if inner.closed {
            return Err(DeltafinError::new("storage reader is closed"));
        }
        for _ in 0..participating {
            inner.queues.push(priority, Arc::clone(&batch));
        }
        self.state.available.notify_all();
        drop(inner);
        Ok(Some(ReadTicket {
            batch,
            started,
            bytes: logical_bytes,
            // Logical source count, NOT physical chunk jobs: the expert-union
            // wait contract (experts.rs expected_jobs = selection.len())
            // validates against this. K3_SPLIT_READ multiplies only the
            // worker-facing job count above.
            jobs: source_indices.len(),
            workers: participating,
        }))
    }

    /// K3_SUMMER_READ_INTO_SLOT: the same deferred-exact job, but its
    /// `Other` destination is caller-owned memory (a Summer pool slot) rather
    /// than an arena slot — the bytes land where they will live, no memcpy.
    /// Takes no arena slot, so it carries no arena backpressure: the pool's
    /// own slot reservation is the bound. `keepalive` pins the owner (the
    /// pool span's Arc) for the batch's life, so a pool eviction guard that
    /// counts that Arc's strong references sees this in-flight/served read.
    pub fn try_submit_deferred_exact_into(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
        priority: ReadPriority,
        dest_pointer: NonNull<u8>,
        dest_capacity: usize,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Option<ReadTicket>> {
        let (lengths, source_length) =
            deferred_batch_lengths(catalog, source_indices, destination)?;
        if destination != BufferKind::Other
            || lengths.get(BufferKind::Quantized) != 0
            || lengths.get(BufferKind::Scales) != 0
            || lengths.get(BufferKind::Other) > dest_capacity
        {
            return Ok(None);
        }
        let logical_bytes = lengths.get(destination) as u64;
        let started = Instant::now();
        {
            let inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
        }
        let buffers = Arc::new(SharedBuffers::external_other(
            dest_pointer,
            dest_capacity,
            keepalive,
        )?);
        let lease = Arc::new(BufferLeaseInner {
            arena: Weak::new(),
            slot_index: usize::MAX,
            buffers: Some(buffers),
            lengths,
        });
        let batch = Arc::new(Batch::new_deferred_exact_validated(
            catalog,
            source_indices,
            destination,
            source_length,
            lease,
            priority,
        ));
        let participating = self.workers().min(batch.jobs.len());
        let mut inner = self.state.inner.lock().unwrap();
        if inner.closed {
            return Err(DeltafinError::new("storage reader is closed"));
        }
        for _ in 0..participating {
            inner.queues.push(priority, Arc::clone(&batch));
        }
        self.state.available.notify_all();
        drop(inner);
        Ok(Some(ReadTicket {
            batch,
            started,
            bytes: logical_bytes,
            jobs: source_indices.len(),
            workers: participating,
        }))
    }

    pub fn read_deferred_exact(
        &self,
        catalog: &DeferredExactCatalog,
        source_indices: &[u32],
        destination: BufferKind,
    ) -> Result<(LayerBuffers, ReadStats)> {
        self.submit_deferred_exact(catalog, source_indices, destination, ReadPriority::Demand)?
            .wait()
    }

    fn submit_inner(
        &self,
        plan: &ReadPlan,
        priority: ReadPriority,
        wait_for_slot: bool,
    ) -> Result<Option<ReadTicket>> {
        let started = Instant::now();
        {
            let inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
        }
        let Some(lease) = self
            .arena
            .acquire(plan.buffer_lengths, wait_for_slot, priority)?
        else {
            return Ok(None);
        };
        let batch = Arc::new(Batch::new(plan, lease, priority));
        let participating = self.workers().min(plan.jobs.len());
        if participating != 0 {
            let mut inner = self.state.inner.lock().unwrap();
            if inner.closed {
                return Err(DeltafinError::new("storage reader is closed"));
            }
            for _ in 0..participating {
                inner.queues.push(priority, Arc::clone(&batch));
            }
            self.state.available.notify_all();
        }
        Ok(Some(ReadTicket {
            batch,
            started,
            bytes: plan.logical_bytes,
            jobs: plan.jobs.len(),
            workers: participating,
        }))
    }
}

fn deferred_batch_lengths(
    catalog: &DeferredExactCatalog,
    source_indices: &[u32],
    destination: BufferKind,
) -> Result<(BufferLengths, usize)> {
    if source_indices.is_empty() || source_indices.len() > MAX_INLINE_DEFERRED_FILES {
        return Err(DeltafinError::new(format!(
            "an inline deferred batch needs 1..={MAX_INLINE_DEFERRED_FILES} sources; got {}",
            source_indices.len()
        )));
    }
    for &source in source_indices {
        if source as usize >= catalog.inner.sources.len() {
            return Err(DeltafinError::new(format!(
                "deferred source index {source} is outside catalog length {}",
                catalog.inner.sources.len()
            )));
        }
    }
    let source_length = usize::try_from(catalog.inner.exact_source_length).map_err(|_| {
        DeltafinError::new("deferred source length does not fit this platform's usize")
    })?;
    let batch_length = source_length
        .checked_mul(source_indices.len())
        .ok_or_else(|| DeltafinError::new("deferred batch length overflows usize"))?;
    let lengths = match destination {
        BufferKind::Quantized => BufferLengths::new(batch_length, 0, 0),
        BufferKind::Scales => BufferLengths::new(0, batch_length, 0),
        BufferKind::Other => BufferLengths::new(0, 0, batch_length),
    };
    Ok((lengths, source_length))
}

fn worker_main(state: Arc<PoolState>, prefetch_class: bool) {
    // Prefetch workers may be deprioritized at the kernel's per-device I/O
    // queues (K3_PREFETCH_IOPOL) so demand reads never wait behind
    // speculation; with the knob unset both paths are identical.
    if prefetch_class {
        crate::io_priority::configure_prefetch_io_thread();
    } else {
        crate::io_priority::configure_model_io_thread();
    }
    loop {
        let batch = {
            let mut inner = state.inner.lock().unwrap();
            while inner.queues.is_empty() && !inner.closed {
                inner = state.available.wait(inner).unwrap();
            }
            let Some(batch) = inner.queues.pop() else {
                return;
            };
            batch
        };
        match batch.run_quantum() {
            QuantumOutcome::Requeue => {
                let priority = batch.priority;
                let mut inner = state.inner.lock().unwrap();
                // Internal requeues remain valid during shutdown: Reader::drop
                // drains admitted work before joining the workers.
                inner.queues.push(priority, batch);
                state.available.notify_one();
            }
            QuantumOutcome::Idle => drop(batch),
            QuantumOutcome::Finished(completion) => {
                // Drop this worker's Arc<Batch> -- and with it, if this was
                // the last reference, the batch's arena lease -- before
                // publishing completion. A waiter woken by the notify below
                // must never be able to observe this lease still held.
                drop(batch);
                let _guard = completion.lock.lock().unwrap();
                completion.condvar.notify_one();
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.state.inner.lock().unwrap().closed = true;
        self.state.available.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DeferredSourceIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

fn metadata_identity(metadata: &std::fs::Metadata) -> DeferredSourceIdentity {
    DeferredSourceIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        bytes: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

fn descriptor_identity(file: &File, path: &Path) -> Result<DeferredSourceIdentity> {
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat authenticated source", path, error))?;
    Ok(metadata_identity(&metadata))
}

fn capture_deferred_source_identity(
    path: &Path,
    exact_source_length: u64,
) -> Result<DeferredSourceIdentity> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(open_cloexec_nofollow())
        .open(path)
        .map_err(|error| io_error("open deferred identity source", path, error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat deferred identity source", path, error))?;
    if !metadata.is_file() || metadata.len() != exact_source_length {
        return Err(DeltafinError::new(format!(
            "deferred identity source {} is not an exact regular {exact_source_length}-byte file",
            path.display(),
        )));
    }
    Ok(metadata_identity(&metadata))
}

/// Authenticate every byte in each declared verification range and scatter
/// retained subranges in the same positional-read pass.
///
/// The arena remains batch-private until all source jobs succeed, so a digest
/// or identity failure can leave partially written private pages but can never
/// publish them. Destination ranges were globally proven disjoint when the
/// plan was opened, which also makes concurrent source jobs safe.
fn authenticate_and_scatter_source(
    file: &File,
    source: &Source,
    lease: &BufferLeaseInner,
) -> Result<()> {
    let before = descriptor_identity(file, &source.path)?;
    if before.bytes != source.length {
        return Err(DeltafinError::new(format!(
            "authenticated source {} changed length before verification",
            source.path.display()
        )));
    }

    let mut scratch = vec![0_u8; 256 * 1024];
    for (verification_index, verification) in source.verifications.iter().enumerate() {
        let verification_end = verification
            .source_offset
            .checked_add(verification.length as u64)
            .ok_or_else(|| DeltafinError::new("authenticated source range overflows u64"))?;
        let mut cursor = verification.source_offset;
        let mut digest = DigestState::new();

        for scatter in source
            .scatter_extents
            .iter()
            .filter(|scatter| scatter.verification_index == verification_index)
        {
            if scatter.source_offset < cursor {
                return Err(DeltafinError::new(format!(
                    "authenticated gather order overlaps in {}",
                    source.path.display()
                )));
            }
            while cursor < scatter.source_offset {
                let remaining = usize::try_from(scatter.source_offset - cursor)
                    .map_err(|_| DeltafinError::new("authenticated gap exceeds usize"))?;
                let request = scratch.len().min(remaining);
                let count = match file.read_at(&mut scratch[..request], cursor) {
                    Ok(count) => count,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(io_error(
                            "pread authenticated source gap",
                            &source.path,
                            error,
                        ));
                    }
                };
                EXPERT_READ_BYTES.fetch_add(count as u64, Ordering::Relaxed);
                if count == 0 {
                    return Err(DeltafinError::new(format!(
                        "short authenticated pread from {} at {}",
                        source.path.display(),
                        cursor
                    )));
                }
                digest.update(&scratch[..count]);
                cursor += count as u64;
            }

            // SAFETY: validate_destinations proved all output extents are
            // bounded and globally disjoint. The lease is not exposed as an
            // immutable LayerBuffers value until every batch job completes.
            let pointer = lease
                .buffers()
                .get(scatter.destination)
                .pointer_at(scatter.destination_offset, scatter.length);
            let destination = unsafe { std::slice::from_raw_parts_mut(pointer, scatter.length) };
            let mut completed = 0_usize;
            while completed < destination.len() {
                let count = match file.read_at(
                    &mut destination[completed..],
                    scatter.source_offset + completed as u64,
                ) {
                    Ok(count) => count,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        return Err(io_error("pread authenticated gather", &source.path, error));
                    }
                };
                EXPERT_READ_BYTES.fetch_add(count as u64, Ordering::Relaxed);
                if count == 0 {
                    return Err(DeltafinError::new(format!(
                        "short authenticated gather pread {}/{} from {} at {}",
                        completed,
                        destination.len(),
                        source.path.display(),
                        scatter.source_offset
                    )));
                }
                digest.update(&destination[completed..completed + count]);
                completed += count;
            }
            cursor = scatter.source_offset + scatter.length as u64;
        }

        while cursor < verification_end {
            let remaining = usize::try_from(verification_end - cursor)
                .map_err(|_| DeltafinError::new("authenticated tail exceeds usize"))?;
            let request = scratch.len().min(remaining);
            let count = match file.read_at(&mut scratch[..request], cursor) {
                Ok(count) => count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(io_error(
                        "pread authenticated source tail",
                        &source.path,
                        error,
                    ));
                }
            };
            EXPERT_READ_BYTES.fetch_add(count as u64, Ordering::Relaxed);
            if count == 0 {
                return Err(DeltafinError::new(format!(
                    "short authenticated pread from {} at {}",
                    source.path.display(),
                    cursor
                )));
            }
            digest.update(&scratch[..count]);
            cursor += count as u64;
        }

        if digest.finalize() != verification.expected_digest {
            return Err(DeltafinError::new(format!(
                "source authentication failed SHA-256 verification for {} at {}..{}",
                source.path.display(),
                verification.source_offset,
                verification_end,
            )));
        }
        drop_completed_cache(
            file,
            source.cache_policy,
            verification.source_offset,
            verification.length,
        );
    }

    let after = descriptor_identity(file, &source.path)?;
    if after != before {
        return Err(DeltafinError::new(format!(
            "authenticated source identity changed during verification and gather: {}",
            source.path.display()
        )));
    }
    Ok(())
}

fn open_deferred_source(source: &Source) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(open_cloexec_nofollow())
        .open(&source.path)
        .map_err(|error| {
            if matches!(error.raw_os_error(), Some(23 | 24)) {
                DeltafinError::new(format!(
                    "open deferred source {} without following symlinks: descriptor limit exhausted",
                    source.path.display()
                ))
            } else {
                io_error("open deferred non-symlink source", &source.path, error)
            }
        })?;
    let metadata = file
        .metadata()
        .map_err(|error| io_error("stat deferred source", &source.path, error))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "deferred source is not a regular file: {}",
            source.path.display()
        )));
    }
    if metadata.len() != source.length {
        return Err(DeltafinError::new(format!(
            "deferred source {} is {} bytes; expected exact length {}",
            source.path.display(),
            metadata.len(),
            source.length,
        )));
    }
    if let Some(expected) = source.expected_identity {
        let actual = metadata_identity(&metadata);
        if actual != expected {
            return Err(DeltafinError::new(format!(
                "deferred source identity changed before range gather: {}",
                source.path.display(),
            )));
        }
    }
    configure_cache_policy(&file, &source.path, source.cache_policy)?;
    Ok(file)
}

// Resolved from the active target ABI, never a literal: O_NOFOLLOW is 0x20000
// on x86_64 Linux but 0x8000 on aarch64, so an x86-derived literal decodes to
// O_LARGEFILE there and silently opens these sources through symlinks.
const fn open_cloexec_nofollow() -> i32 {
    libc::O_CLOEXEC | libc::O_NOFOLLOW
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("Deltafin native storage currently supports macOS and Linux");

/// K3_MIRROR_SCHED=1 (default off = upstream static order): least-loaded
/// dispatch across the pools that hold a copy of an expert.
///
/// Hot-tier entries are COPIES — the enclosure original is never removed — so
/// every tier-resident expert is dual-homed, and either source returns the
/// same bytes (length- and identity-checked by the caller). Static resolution
/// always reads the hot tier first, which pins the internal SSD at ~70% of
/// read traffic while both enclosures idle: measured 2026-08-24, internal
/// 9.45 GB/s of 13.73 capability (69%), each enclosure 2.6 of 5.58 (47%),
/// aggregate 14.67 of 24.9. Because a token's ~32 GB of expert traffic is
/// throughput-bound, that imbalance sets the token time directly.
///
/// The dispatcher keeps a virtual clock per pool — the projected time each
/// has been given work for — and sends the next read to whichever pool is
/// furthest ahead. No completion callback is needed: charging at submission
/// converges on the same balance a work-conserving scheduler would reach.
/// Which physical device served an expert open.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MirrorDev {
    Internal = 0,
    DirC = 1,
    DirB = 2,
    Primary = 3,
}

const MIRROR_DEVS: [MirrorDev; 4] = [
    MirrorDev::Internal,
    MirrorDev::DirC,
    MirrorDev::DirB,
    MirrorDev::Primary,
];

/// One counter per cache line. `[AtomicU64; 4]` is 32 bytes — a single line
/// shared by 64 reader threads, so every dispatch invalidates the line for all
/// of them. Flagged as free money by external review 2026-08-31 and folded in
/// here rather than left as a separate patch.
#[repr(align(128))]
struct PaddedU64(AtomicU64);

/// Per-device service clocks, in microseconds of charged work.
///
/// MONOTONIC. This is the defect that makes MODE 1 a weighted round-robin
/// rather than a scheduler: it is a lifetime integral, so its sensitivity to
/// the current dispatch decays as 1/N. Adding 1502 us to a clock already at
/// ~15,000,000 us cannot reorder anything, and by mid-run the ordering is
/// frozen at the long-run bandwidth ratio. It has no representation of what is
/// outstanding right now, so two of a barrier's 16 experts can queue behind
/// each other on K3B while internal idles and the clocks call that balanced —
/// which is exactly the case the layer pays for, since it costs max(16 reads).
static MIRROR_CLOCKS: [PaddedU64; 4] = [
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
];

/// MODE 2. Reads currently IN FLIGHT per device: incremented on issue,
/// decremented on completion. Unlike the clocks this is a level, not an
/// integral, so it answers "what is this device busy with NOW" — the question
/// a barrier actually poses. Estimated completion for a candidate device is
/// `(outstanding + 1) * cost_us(dev)`, and dispatch picks the argmin, i.e.
/// the drive that can finish this expert first rather than the drive that is
/// furthest behind its lifetime quota.
static MIRROR_OUTSTANDING: [PaddedU64; 4] = [
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
    PaddedU64(AtomicU64::new(0)),
];

thread_local! {
    /// Device this worker most recently dispatched to, so the completion site
    /// can decrement the right counter. Safe because in the catalog path the
    /// open and the read are sequential on the SAME worker thread. Cached-open
    /// paths never call the dispatcher, so they never increment, and therefore
    /// must not decrement — `None` records that.
    static MIRROR_INFLIGHT_DEV: std::cell::Cell<Option<MirrorDev>> =
        const { std::cell::Cell::new(None) };
}

/// Estimated microseconds for `dev` to finish one more expert record.
fn mirror_eta_us(dev: MirrorDev) -> u64 {
    let index = dev as usize;
    let queued = MIRROR_OUTSTANDING[index].0.load(Ordering::Relaxed);
    // Bias applies here too: it is an EV correction on the cost estimate, and
    // an estimate used for ordering must carry the same correction the charge
    // does or the two disagree about what a device costs.
    let unit = mirror_cost_us(dev);
    unit.saturating_mul(queued.saturating_add(1))
}

/// Mark a dispatch to `dev` as in flight (MODE 2 only).
fn mirror_issue(dev: MirrorDev) {
    if mirror_sched_mode() != 2 {
        return;
    }
    MIRROR_OUTSTANDING[dev as usize].0.fetch_add(1, Ordering::Relaxed);
    MIRROR_INFLIGHT_DEV.with(|slot| slot.set(Some(dev)));
}

/// Retire the dispatch this worker last issued (MODE 2 only). Idempotent: the
/// slot is cleared, so a second call cannot drive the counter negative — an
/// under-count merely makes a device look freer, whereas an over-count would
/// permanently exile it from selection.
pub(crate) fn mirror_retire() {
    if mirror_sched_mode() != 2 {
        return;
    }
    MIRROR_INFLIGHT_DEV.with(|slot| {
        if let Some(dev) = slot.take() {
            MIRROR_OUTSTANDING[dev as usize]
                .0
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)))
                .ok();
        }
    });
}

/// Per-device in-flight depth, for telemetry: (internal, C, B, A).
pub fn mirror_outstanding_report() -> (u64, u64, u64, u64) {
    let n = |dev: MirrorDev| MIRROR_OUTSTANDING[dev as usize].0.load(Ordering::Relaxed);
    (
        n(MirrorDev::Internal),
        n(MirrorDev::DirC),
        n(MirrorDev::DirB),
        n(MirrorDev::Primary),
    )
}

/// Per-device cost of one 17.5 MB expert record, from the concurrent
/// capability measured 2026-08-29: internal 11.68, K3C 7.08, K3B 5.62,
/// K3A 5.80 GB/s.
///
/// The internal figure derives to 1502 us, which is exactly the corrected
/// two-way constant — the same measurement seen from two directions.
///
/// WHY FOUR CLOCKS AND NOT TWO. The old scheduler weighed internal against
/// all-enclosures-as-one-bucket, and within that bucket the probe order was
/// fixed with dir_c unconditionally first. That is harmless while most
/// experts have exactly one home, which is why it measured null (-0.33%,
/// 2026-08-31 P3). It becomes actively harmful the moment experts are
/// REPLICATED: every enclosure-side candidate resolves on the first probe, so
/// K3C absorbs everything. Simulated on four recorded finance-style traces, replicating the
/// >20x band under the two-way clock costs **-20.1%**, with K3C's read share
/// going 23% -> 43% while K3B and K3A fall to 9%. The same replication under
/// four-way load ordering gains **+17.1%**.
///
/// Replication and per-device scheduling are a PAIR: neither pays alone, and
/// the two-way version of this code makes the pair impossible.
/// K3_MIRROR_COST_ACHIEVED=1 selects constants derived from ACHIEVED steady
/// bandwidth instead of rated. Default stays rated so every arm recorded before
/// today remains comparable.
///
/// The rated figures are sequential-benchmark numbers; measured steady decode
/// never reaches them:
///     internal  rated 11.68  achieved 10.28  -> cost 1502 -> 1707  (+14%)
///     K3C       rated  7.08  achieved  5.37  -> cost 2478 -> 3268  (+32%)
///     K3B       rated  5.62  achieved  4.08  -> cost 3122 -> 4301  (+38%)
///     K3A       rated  5.80  achieved  4.20  -> cost 3025 -> 4178  (+38%)
///
/// This matters more than the percentages suggest. Per-read trace attribution
/// (9,986 barriers, all 243,494 reads joined to a device) shows K3C sets Tmax on
/// 42.7% of barriers while serving 24.5% of reads — 1.74x over-represented as
/// the straggler. The cost model undercharges exactly that device by 32%, so
/// mode 2 sends it more work than it can absorb. Internal is the mirror image:
/// 40.7% of reads, 28.3% of stragglers, 0.69x.
const MIRROR_COST_RATED: [u64; 4] = [1502, 2478, 3122, 3025];
const MIRROR_COST_ACHIEVED: [u64; 4] = [1707, 3268, 4301, 4178];

fn mirror_cost_table() -> &'static [u64; 4] {
    static SEL: OnceLock<bool> = OnceLock::new();
    let achieved = *SEL.get_or_init(|| {
        std::env::var("K3_MIRROR_COST_ACHIEVED").is_ok_and(|v| v == "1")
    });
    if achieved { &MIRROR_COST_ACHIEVED } else { &MIRROR_COST_RATED }
}

/// Which cost table resolved, for the startup line — the knob is absent from
/// `[config] resolved:`.
pub fn mirror_cost_table_report() -> &'static str {
    if std::ptr::eq(mirror_cost_table(), &MIRROR_COST_ACHIEVED) { "achieved" } else { "rated" }
}

/// EV COMPENSATION for the scheduler (K3_MIRROR_BIAS).
///
/// The cost constants are a light meter: they estimate what each device costs
/// from its *sequential* bandwidth. Like a meter they can be systematically
/// off, and like a photographer we would rather dial a known correction than
/// rebuild the meter.
///
/// Format: `internal:0.85,K3A:1.1` — a MULTIPLIER on that device's cost.
/// Below 1 makes the device look cheaper, so it attracts more work; above 1
/// pushes work away. Unlisted devices stay at 1.0. Clamped to [0.25, 4.0].
///
/// The evidence this exists to chase: with no scheduler the layout ran internal
/// at **43%** of reads and was FASTER than the scheduler's capability-proportional
/// **38%** (0.38505 vs 0.3556 tok/s, 2026-08-31). Internal was at only 64% of its
/// capability with headroom, so over-weighting the fastest device beat balancing.
/// A bias below 1.0 on internal should reproduce that split deliberately.
///
/// Deliberately a multiplier and not new constants: it keeps the derivation
/// (RECORD/bandwidth) intact and auditable, and makes the correction an
/// explicit, logged decision rather than a silently retuned magic number.
fn mirror_bias() -> [f64; 4] {
    static BIAS: std::sync::OnceLock<[f64; 4]> = std::sync::OnceLock::new();
    *BIAS.get_or_init(|| {
        let mut bias = [1.0_f64; 4];
        let Ok(raw) = std::env::var("K3_MIRROR_BIAS") else {
            return bias;
        };
        for entry in raw.split(',') {
            let Some((name, value)) = entry.split_once(':') else {
                continue;
            };
            let Ok(factor) = value.trim().parse::<f64>() else {
                continue;
            };
            let index = match name.trim() {
                "internal" => 0,
                "K3C" | "dir_c" => 1,
                "K3B" | "dir_b" => 2,
                "K3A" | "primary" => 3,
                _ => continue,
            };
            bias[index] = factor.clamp(0.25, 4.0);
        }
        bias
    })
}

/// Cost after EV compensation. This is what the clocks are charged.
fn mirror_cost_us(dev: MirrorDev) -> u64 {
    let index = dev as usize;
    ((mirror_cost_table()[index] as f64) * mirror_bias()[index]).round().max(1.0) as u64
}

/// Resolved bias, for the startup line — so a compensated run can never be
/// mistaken for an uncompensated one when the logs are read back.
pub fn mirror_bias_report() -> [f64; 4] {
    mirror_bias()
}

/// K3_MIRROR_SCHED: 0/unset = upstream static probe order.
///   1 = cumulative-clock order (the original; a weighted round-robin, see
///       MIRROR_CLOCKS — kept so every arm recorded before 2026-08-31 stays
///       reproducible against its own logs).
///   2 = least-expected-completion: dispatch to the drive that can finish this
///       expert FIRST, from live in-flight depth rather than lifetime totals.
fn mirror_sched_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("K3_MIRROR_SCHED")
            .ok()
            .and_then(|value| value.trim().parse::<u8>().ok())
            .filter(|mode| *mode <= 2)
            .unwrap_or(0)
    })
}

fn mirror_sched_enabled() -> bool {
    mirror_sched_mode() != 0
}

/// Microseconds one 17,547,264 B expert occupies each pool, from the measured
/// engine-pattern rates. Pool throughput is the right currency rather than
/// per-stream latency, because one stream already saturates a drive (5.23 of
/// 5.58 GB/s single-threaded), so a second concurrent read buys nothing per
/// drive.
///
/// RE-DERIVED 2026-08-31. Both constants were stale and the error was large
/// enough to invert the scheduler's preference:
///
///   internal    13.73 -> 11.68 GB/s  =>  1278 -> 1502 us
///   enclosures  11.16 -> 18.50 GB/s  =>  1572 ->  948 us
///
/// The enclosure figure was derived when there were TWO enclosures totalling
/// 11.16 GB/s. There are now THREE, measured concurrently at
/// 7.08 + 5.62 + 5.80 = 18.50 GB/s (observed maxima across 130+ timed arms,
/// 2026-08-29). Internal's 11.68 GB/s comes from the same set.
///
/// Old ratio internal/enclosure 0.813 biased every dual-homed decision toward
/// internal; the corrected ratio is 1.584, i.e. the scheduler had the sign
/// wrong, not merely the magnitude. Expected speed effect is nonetheless ~0%:
/// the current four-way layout is an index split, so most experts have exactly
/// one home and this path is rarely exercised. This is a correctness fix.
const MIRROR_INTERNAL_COST_US: u64 = 1502;
const MIRROR_ENCLOSURE_COST_US: u64 = 948;

/// Resolved mirror-scheduler state for the startup line: (enabled, internal_us,
/// enclosure_us). Emitted so a stale cost constant can never again sit
/// unnoticed in the tree — the 2026-08-31 re-derivation found both constants
/// wrong by enough to invert the scheduler's preference, and nothing in any log
/// would have shown it.
pub fn mirror_sched_report() -> (bool, u64, u64) {
    (
        mirror_sched_enabled(),
        mirror_cost_table()[MirrorDev::Internal as usize],
        MIRROR_ENCLOSURE_COST_US,
    )
}

/// Which dispatch policy actually resolved. Reported separately because
/// K3_MIRROR_SCHED now has three values and `enabled=true` no longer says which
/// one is running — an arm that meant to test mode 2 and silently got mode 1
/// would produce a plausible, wrong, and unfalsifiable result.
pub fn mirror_sched_mode_report() -> u8 {
    mirror_sched_mode()
}

/// Per-device opens the scheduler actually dispatched: (internal, C, B, A).
/// The two-way version of this was dead code and the dispatch split was
/// therefore unobservable; with four clocks it is the only way to assert the
/// balance BY EFFECT rather than by resolution.
pub fn mirror_device_split() -> (u64, u64, u64, u64) {
    let n = |d: MirrorDev| {
        MIRROR_CLOCKS[d as usize].0.load(Ordering::Relaxed) / mirror_cost_us(d).max(1)
    };
    (n(MirrorDev::Internal), n(MirrorDev::DirC), n(MirrorDev::DirB), n(MirrorDev::Primary))
}

/// Probe order for one expert open.
///
/// Disabled, or with an explicit split-read home hint, this returns the
/// historical fixed order (dir_c, internal, dir_b, primary) so behaviour is
/// byte-identical to before the four-way change. Enabled, devices are ordered
/// by current clock so the least-loaded home is probed first; the order stays
/// a permutation of all four, so resolution remains total and an expert can
/// never become unreachable.
/// K3_PROBE_ORDER=hot_first — probe `hot` (internal) before `dir_c` (K3C).
///
/// WHY. The static chain is dir_c -> hot -> dir_b -> primary, so K3C is probed
/// FIRST despite being the slower of the two mirror tiers (rated 7.08 GB/s vs
/// internal's 11.68; achieved steady 10.28 for internal). Whatever occupies the
/// first-probed slot absorbs hot traffic alone while the other three idle.
/// Measured dose-response, 40 tok, band placed in dir_c:
///
///     0% in dir_c -> +1.1%   25% -> -4.5%   50% -> -15.0%   100% -> -32.6%
///
/// and the per-read trace caught the mechanism directly: a 50/50 internal/K3C
/// split delivered 63% of pool reads to K3C, a 25%-each-of-four split delivered
/// 43% to K3C and 4.6% to K3A. dir_c captures its own share PLUS everything it
/// already held. That is also the mechanism behind the -22.31% R_OFF arm.
///
/// LOW RISK. The `home_hint` branch below already returns exactly this ordering
/// for hinted opens, so hot-first is a known-good permutation — this only
/// changes which case gets it. The order remains a permutation of all four, so
/// resolution stays total and no expert becomes unreachable.
///
/// BEHIND A FLAG, never a silent default: a silent reorder would make every
/// arm recorded before today incomparable with every arm after it.
///
/// CAVEAT recorded before measuring: this moves K3C's 20,516 experts onto
/// internal, taking its read share from ~41% toward ~55%. The recorded
/// simulation for pushing internal past ~81% was -34.5%, so the effect is not
/// obviously monotonic in internal's share.
fn probe_hot_first() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("K3_PROBE_ORDER").is_ok_and(|v| v.trim().eq_ignore_ascii_case("hot_first"))
    })
}

/// Resolved probe order, for the startup line — K3_PROBE_ORDER does not appear
/// in `[config] resolved:`, so this is the only assertable record of it.
pub fn probe_order_report() -> &'static str {
    if probe_hot_first() { "hot_first" } else { "dir_c_first" }
}

fn mirror_probe_order(home_hint: Option<bool>) -> [MirrorDev; 4] {
    const DIR_C_FIRST: [MirrorDev; 4] = [
        MirrorDev::DirC,
        MirrorDev::Internal,
        MirrorDev::DirB,
        MirrorDev::Primary,
    ];
    const HOT_FIRST: [MirrorDev; 4] = [
        MirrorDev::Internal,
        MirrorDev::DirC,
        MirrorDev::DirB,
        MirrorDev::Primary,
    ];
    let legacy = if probe_hot_first() { HOT_FIRST } else { DIR_C_FIRST };
    if !mirror_sched_enabled() {
        return legacy;
    }
    if let Some(prefer_internal) = home_hint {
        return if prefer_internal {
            HOT_FIRST
        } else {
            legacy
        };
    }
    let mut order = MIRROR_DEVS;
    if mirror_sched_mode() == 2 {
        // Least expected completion. Ties broken by raw speed so that when the
        // machine is idle (all depths zero, the common case at a barrier
        // boundary) the fastest device wins instead of whatever the array
        // order happened to be — without this the policy degenerates to the
        // static order exactly when it matters most.
        order.sort_by_key(|dev| (mirror_eta_us(*dev), mirror_cost_us(*dev)));
    } else {
        order.sort_by_key(|dev| MIRROR_CLOCKS[*dev as usize].0.load(Ordering::Relaxed));
    }
    order
}

// ---------------------------------------------------------------------------
// PER-READ PROVENANCE TRACE  (K3_READ_TRACE=<path>)
//
// Answers, per second and per barrier, WHICH expert was served from WHERE:
// the RAM pools (summer / retain / pinned) or one of the four drives. Nothing
// in the engine could answer that before this: `[summer]` and `[mirror-split]`
// are END-OF-RUN TOTALS, and the disk-sampler .csv has 200 ms time resolution but
// is an OS byte counter with no expert identity, so it cannot say which expert
// or which barrier a read belonged to.
//
// LOCK-FREE ON PURPOSE. 64 reader threads hit this path; a Mutex<Vec> would
// serialise them and the instrument would change the thing it measures. Each
// record is packed into two u64 stored in preallocated atomic arrays, claimed
// with one fetch_add. Cost is ~2 relaxed stores per expert read.
//
//   w0: t_us(40) | layer(7) | expert(10) | source(3)   ... 60 bits
//   w1: dur_ns(32) | barrier(32)
//
// Records are DROPPED, not wrapped, once the buffer is full, and the drop
// count is reported — a silently wrapping ring would look like a complete
// trace with a plausible but wrong tail.
const TRACE_CAP: usize = 1 << 21; // 2M records ~= 32 MB; a 200-tok run makes ~250k

pub(crate) const SRC_SUMMER: u8 = 0;
pub(crate) const SRC_RETAIN: u8 = 1;
pub(crate) const SRC_PINNED: u8 = 2;
pub(crate) const SRC_INTERNAL: u8 = 3;
pub(crate) const SRC_K3C: u8 = 4;
pub(crate) const SRC_K3B: u8 = 5;
pub(crate) const SRC_K3A: u8 = 6;
// The TRANSFER, recorded separately from the open. The open site is the only
// place that knows the resolved device, and the read site is the only place
// that knows how long the 17.5 MB actually took — and for catalog-resolved
// experts neither can see the other's fact without threading state through a
// cached-descriptor path where opens do not happen per read. Keeping them as
// two records is honest: `read` carries the duration that sets the barrier
// tail, the device records carry placement, and the reducer uses each for what
// it actually measured instead of inventing a join that would silently
// mis-attribute every cached open.
pub(crate) const SRC_READ: u8 = 7;
const SRC_NAMES: [&str; 8] =
    ["summer", "retain", "pinned", "internal", "K3C", "K3B", "K3A", "read"];

static TRACE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static TRACE_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
static TRACE_IDX: AtomicU64 = AtomicU64::new(0);
static TRACE_DROPPED: AtomicU64 = AtomicU64::new(0);
static TRACE_BARRIER: AtomicU64 = AtomicU64::new(0);
static TRACE_W0: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
static TRACE_W1: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
// w2 (2026-09-06): target_barrier(32) | tile_layer(8) | offset_units(12) | length_units(12),
// units of BUFFER_ALIGNMENT bytes. target_barrier = the pass a prefetch read
// serves (the current pass for demand/union reads), so consumers no longer
// guess it from the pass current at completion; offset/length give the exact
// byte range so nothing is inferred from record counts.
static TRACE_W2: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
static TRACE_LAYER: AtomicU32 = AtomicU32::new(0);

fn trace_enabled() -> bool {
    *TRACE_ON.get_or_init(|| {
        let on = std::env::var("K3_READ_TRACE").is_ok_and(|v| !v.trim().is_empty());
        if on {
            TRACE_T0.get_or_init(std::time::Instant::now);
            TRACE_W0.get_or_init(|| (0..TRACE_CAP).map(|_| AtomicU64::new(0)).collect());
            TRACE_W1.get_or_init(|| (0..TRACE_CAP).map(|_| AtomicU64::new(0)).collect());
            TRACE_W2.get_or_init(|| (0..TRACE_CAP).map(|_| AtomicU64::new(0)).collect());
        }
        on
    })
}

/// Open a new barrier (one layer's top-16 union). Returns its id.
pub(crate) fn trace_barrier_begin(layer: u32) -> u64 {
    if !trace_enabled() {
        return 0;
    }
    TRACE_LAYER.store(layer, Ordering::Relaxed);
    let id = TRACE_BARRIER.fetch_add(1, Ordering::Relaxed);
    // Pass-begin marker (2026-09-06): source "read", expert 1023 (no such
    // expert; ids stop at 895), duration 0. Consumers measure a pass's storage
    // span from this stamp, so a prefetch that landed a whole chunk earlier
    // cannot stretch the span backwards (the "layer 20 = 5.8 s" artefact).
    // Reads issued during this pass record the counter AFTER the increment
    // (id + 1), so the marker carries id + 1 too (05:35 fix: the first
    // captures had it one behind, which the parser detects and shifts).
    trace_read_prio(id + 1, layer, 0x3FF, SRC_READ, 0, None, 0, 0);
    id
}

/// The pass a read serves. Demand and union reads serve the current pass. A
/// prefetch for layer L issued while the pass of layer Lb is current serves the
/// pass of L in this chunk when L > Lb, otherwise in the next chunk: barriers
/// advance one MoE layer per pass, 92 passes per chunk sweep.
fn trace_target_barrier(current: u64, layer: u32, prefetch: Option<bool>) -> u64 {
    let tile_layer = TRACE_LAYER.load(Ordering::Relaxed);
    if prefetch != Some(true) || tile_layer == 0 || layer == tile_layer {
        return current;
    }
    let passes = 92_i64;
    let delta = ((layer as i64 - tile_layer as i64) % passes + passes) % passes;
    current + delta as u64
}

/// Record one expert arrival. `dur_ns` is 0 for RAM hits (no I/O performed).
pub(crate) fn trace_read(barrier: u64, layer: u32, expert: u16, source: u8, dur_ns: u64) {
    trace_read_prio(barrier, layer, expert, source, dur_ns, None, 0, 0);
}

/// Same record with the read's priority: `Some(true)` prefetch, `Some(false)`
/// demand, `None` unknown (legacy sites). Bits 60/61 of w0 were spare (t_us
/// uses 40 bits from bit 20), so the layout stays a two-u64 packed record.
pub(crate) fn trace_read_prio(
    barrier: u64,
    layer: u32,
    expert: u16,
    source: u8,
    dur_ns: u64,
    prefetch: Option<bool>,
    offset: u64,
    length: u64,
) {
    if !trace_enabled() {
        return;
    }
    let (Some(w0s), Some(w1s), Some(w2s)) = (TRACE_W0.get(), TRACE_W1.get(), TRACE_W2.get()) else {
        return;
    };
    let slot = TRACE_IDX.fetch_add(1, Ordering::Relaxed) as usize;
    if slot >= TRACE_CAP {
        TRACE_DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let units = BUFFER_ALIGNMENT as u64;
    let offset_units = (offset / units).min(0xFFF);
    let length_units = (length.div_ceil(units)).min(0xFFF);
    let w2 = (trace_target_barrier(barrier, layer, prefetch) & 0xFFFF_FFFF) << 32
        | ((TRACE_LAYER.load(Ordering::Relaxed) as u64) & 0xFF) << 24
        | (offset_units << 12)
        | length_units;
    w2s[slot].store(w2, Ordering::Relaxed);
    let t_us = TRACE_T0
        .get()
        .map(|t0| t0.elapsed().as_micros() as u64)
        .unwrap_or(0)
        & 0xFF_FFFF_FFFF;
    let prio_bits: u64 = match prefetch {
        None => 0,
        Some(false) => 1 << 60,
        Some(true) => (1 << 60) | (1 << 61),
    };
    let w0 = prio_bits
        | (t_us << 20)
        | (((layer as u64) & 0x7F) << 13)
        | (((expert as u64) & 0x3FF) << 3)
        | ((source as u64) & 0x7);
    let w1 = ((dur_ns.min(u32::MAX as u64)) << 32) | (barrier & 0xFFFF_FFFF);
    w0s[slot].store(w0, Ordering::Relaxed);
    w1s[slot].store(w1, Ordering::Relaxed);
}

/// Write the trace as CSV. Called at end of run.
pub(crate) fn trace_dump() {
    if !trace_enabled() {
        return;
    }
    let Ok(path) = std::env::var("K3_READ_TRACE") else {
        return;
    };
    let (Some(w0s), Some(w1s), Some(w2s)) = (TRACE_W0.get(), TRACE_W1.get(), TRACE_W2.get()) else {
        return;
    };
    let n = (TRACE_IDX.load(Ordering::Relaxed) as usize).min(TRACE_CAP);
    let dropped = TRACE_DROPPED.load(Ordering::Relaxed);
    let mut out = String::with_capacity(n * 56);
    out.push_str("t_us,barrier,layer,expert,source,dur_ns,prio,target_barrier,tile_layer,offset,bytes\n");
    let units = BUFFER_ALIGNMENT as u64;
    for i in 0..n {
        let w0 = w0s[i].load(Ordering::Relaxed);
        let w1 = w1s[i].load(Ordering::Relaxed);
        let w2 = w2s[i].load(Ordering::Relaxed);
        let src = (w0 & 0x7) as usize;
        let prio = match (w0 >> 60) & 0x3 {
            0 => "-",
            1 => "D",
            _ => "P",
        };
        let offset = ((w2 >> 12) & 0xFFF) * units;
        let length_units = w2 & 0xFFF;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            (w0 >> 20) & 0xFF_FFFF_FFFF,
            w1 & 0xFFFF_FFFF,
            (w0 >> 13) & 0x7F,
            (w0 >> 3) & 0x3FF,
            SRC_NAMES.get(src).copied().unwrap_or("?"),
            w1 >> 32,
            prio,
            w2 >> 32,
            (w2 >> 24) & 0xFF,
            offset,
            length_units * units
        ));
    }
    match std::fs::write(&path, out) {
        Ok(()) => eprintln!(
            "[read-trace] wrote {n} records -> {path} (dropped={dropped}, barriers={})",
            TRACE_BARRIER.load(Ordering::Relaxed)
        ),
        Err(error) => eprintln!("[read-trace] FAILED to write {path}: {error}"),
    }
}

/// `L<layer>-E<expert>[.bin]` -> (layer, expert). Returns None for any name
/// that is not an expert record, so non-expert opens never enter the trace.
pub(crate) fn parse_expert_name(name: &str) -> Option<(u32, u16)> {
    let rest = name.strip_prefix('L')?;
    let (layer, rest) = rest.split_once("-E")?;
    let expert = rest.strip_suffix(".bin").unwrap_or(rest);
    Some((layer.parse().ok()?, expert.parse().ok()?))
}

/// Map the scheduler's device enum onto a trace source tag.
pub(crate) fn trace_src_of(dev: MirrorDev) -> u8 {
    match dev {
        MirrorDev::Internal => SRC_INTERNAL,
        MirrorDev::DirC => SRC_K3C,
        MirrorDev::DirB => SRC_K3B,
        MirrorDev::Primary => SRC_K3A,
    }
}

fn mirror_charge_dev(served: MirrorDev) {
    if !mirror_sched_enabled() {
        return;
    }
    let index = served as usize;
    MIRROR_CLOCKS[index].0.fetch_add(mirror_cost_us(served), Ordering::Relaxed);
}

/// Wall time spent inside expert file opens (openat probes across the resolve
/// chain) and how many opens were issued. The read-path "hint" phase wraps
/// take_prefetch_hint + try_schedule_expert_prefetch, and the latter opens up
/// to 16 experts per layer with up to three directory probes each — this
/// separates that cost from the prediction it is bundled with.
static EXPERT_OPEN_NS: AtomicU64 = AtomicU64::new(0);
static EXPERT_OPEN_COUNT: AtomicU64 = AtomicU64::new(0);
/// Bytes the engine requested from expert files via pread/preadv (logical read
/// volume; compare with the physical SSD counters to see page-cache avoidance).
static EXPERT_READ_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn expert_read_bytes_total() -> u64 {
    EXPERT_READ_BYTES.load(Ordering::Relaxed)
}

pub fn expert_open_totals() -> (u64, u64) {
    (
        EXPERT_OPEN_NS.load(Ordering::Relaxed),
        EXPERT_OPEN_COUNT.load(Ordering::Relaxed),
    )
}

/// Diagnostic for the A/B: how the dispatcher actually split the traffic.
pub fn mirror_schedule_split() -> (u64, u64) {
    let (internal, c, b, a) = mirror_device_split();
    (internal, c + b + a)
}

fn open_deferred_catalog_source(
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    home_hint: Option<bool>,
) -> Result<File> {
    unsafe extern "C" {
        // `c_char` is signed on x86_64 and unsigned on aarch64; spelling the
        // C types keeps this declaration valid on both.
        fn openat(
            directory: libc::c_int,
            path: *const libc::c_char,
            flags: libc::c_int,
            ...
        ) -> libc::c_int;
    }
    // SAFETY: the catalog retains live directory descriptors, `source` is a
    // validated NUL-terminated direct child name, and no mode argument is
    // required because these flags never create a file. The hot tier is
    // probed first; only ENOENT falls through to the primary directory, so a
    // present-but-wrong hot entry fails closed below like a primary one.
    // Load-ordered probe across all four homes. Each device is tried at most
    // once and the order is a permutation of all four, so resolution stays
    // TOTAL: an expert present anywhere is still found, whatever the clocks
    // say. Only ENOENT falls through; any other error fails closed exactly as
    // the fixed cascade did.
    let mut descriptor = -1;
    let mut served = MirrorDev::Primary;
    let open_started = std::time::Instant::now();
    for dev in mirror_probe_order(home_hint) {
        let dir_fd = match dev {
            MirrorDev::Internal => catalog.hot_directory.as_ref().map(|d| d.as_raw_fd()),
            MirrorDev::DirC => catalog.tertiary_directory.as_ref().map(|d| d.as_raw_fd()),
            MirrorDev::DirB => catalog.secondary_directory.as_ref().map(|d| d.as_raw_fd()),
            MirrorDev::Primary => Some(catalog.directory.as_raw_fd()),
        };
        let Some(dir_fd) = dir_fd else {
            continue;
        };
        descriptor = unsafe {
            openat(dir_fd, source.as_c_str().as_ptr(), open_cloexec_nofollow())
        };
        if descriptor >= 0 {
            served = dev;
            break;
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::NotFound {
            return Err(catalog_io_error(
                "open deferred non-symlink expert source",
                catalog,
                source,
                error,
            ));
        }
    }
    // Hinted (K3_SPLIT_READ) opens never consulted the clocks and open once
    // per CHUNK; charging them the whole-expert cost would inflate the clocks
    // ~chunks-fold and bias any remaining clock-decided opens.
    if home_hint.is_none() {
        mirror_charge_dev(served);
        // MODE 2: this device now owes one more record. Gated identically to
        // the charge so the level and the integral always describe the same
        // population of dispatches — hinted (K3_SPLIT_READ) opens fire once per
        // CHUNK rather than per expert, and counting them here would inflate
        // the depth chunk-fold and permanently exile whichever device happened
        // to be serving a split read.
        mirror_issue(served);
    }
    // Provenance: which drive actually served this expert, and when. Recorded
    // here because this is the only point that knows BOTH the resolved device
    // and the expert identity — `[mirror-split]` counts the same decision but
    // only as a run total, with no timestamp and no expert.
    if trace_enabled() {
        if let Some((layer, expert)) = parse_expert_name(source.as_str()) {
            trace_read(
                TRACE_BARRIER.load(Ordering::Relaxed),
                layer,
                expert,
                trace_src_of(served),
                open_started.elapsed().as_nanos() as u64,
            );
        }
    }
    EXPERT_OPEN_NS.fetch_add(
        open_started.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );
    EXPERT_OPEN_COUNT.fetch_add(1, Ordering::Relaxed);
    if descriptor < 0 {
        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(23 | 24)) {
            return Err(DeltafinError::new(format!(
                "open deferred source {}/{} without following symlinks: descriptor limit exhausted",
                catalog.directory_path.display(),
                source.as_str(),
            )));
        }
        return Err(catalog_io_error(
            "open deferred non-symlink source",
            catalog,
            source,
            error,
        ));
    }
    finish_catalog_open(catalog, source, descriptor)
}

/// Validate + configure a freshly opened catalog descriptor (shared by the
/// legacy probe chain and the K3_SPLIT_READ per-home probes).
fn finish_catalog_open(
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    descriptor: i32,
) -> Result<File> {
    // SAFETY: `openat` returned a new owned descriptor. This is its unique
    // owner and `File` closes it on every subsequent success/error path.
    let file = unsafe { File::from_raw_fd(descriptor) };
    let metadata = file
        .metadata()
        .map_err(|error| catalog_io_error("stat deferred source", catalog, source, error))?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "deferred source is not a regular file: {}/{}",
            catalog.directory_path.display(),
            source.as_str(),
        )));
    }
    if metadata.len() != catalog.exact_source_length {
        return Err(DeltafinError::new(format!(
            "deferred source {}/{} is {} bytes; expected exact length {}",
            catalog.directory_path.display(),
            source.as_str(),
            metadata.len(),
            catalog.exact_source_length,
        )));
    }
    configure_catalog_cache_policy(&file, catalog, source)?;
    Ok(file)
}

/// K3_SPLIT_READ: probe exactly ONE home class for a catalog source.
/// `internal` probes the hot tier only; otherwise secondary then primary.
/// Ok(None) = ENOENT everywhere in that class; other errors fail closed.
/// Chunk jobs cache the result per (source, home) on the batch, so a file
/// pays at most two probe sequences however many chunks it has.
fn open_catalog_home(
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    internal: bool,
) -> Result<Option<File>> {
    unsafe extern "C" {
        fn openat(
            directory: libc::c_int,
            path: *const libc::c_char,
            flags: libc::c_int,
            ...
        ) -> libc::c_int;
    }
    let open_started = std::time::Instant::now();
    // SAFETY: live directory descriptors retained by the catalog; validated
    // NUL-terminated direct child name; flags never create a file.
    let probe = |directory: &File| -> Result<Option<i32>> {
        let descriptor = unsafe {
            openat(
                directory.as_raw_fd(),
                source.as_c_str().as_ptr(),
                open_cloexec_nofollow(),
            )
        };
        if descriptor >= 0 {
            return Ok(Some(descriptor));
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::NotFound {
            Ok(None)
        } else {
            Err(catalog_io_error(
                "open split-read catalog source",
                catalog,
                source,
                error,
            ))
        }
    };
    // Which tier answered (0 hot, 1 dir_c, 2 dir_b, 3 primary), recorded per
    // descriptor for the K3_SPLIT_TRACE report.
    let mut served_tier: u8 = 0;
    let descriptor = if internal {
        match &catalog.hot_directory {
            Some(hot) => probe(hot)?,
            None => None,
        }
    } else if tier_balance_enabled() {
        // Same three tiers, probed lightest-first. Whichever answers gets
        // charged, so the next expert in this tile prefers a different device.
        let mut found = None;
        for index in tier_probe_order() {
            let directory = match index {
                0 => catalog.tertiary_directory.as_ref(),
                1 => catalog.secondary_directory.as_ref(),
                _ => Some(&catalog.directory),
            };
            let Some(directory) = directory else { continue };
            if let Some(descriptor) = probe(directory)? {
                TIER_LOAD[index].fetch_add(1, Ordering::Relaxed);
                served_tier = match index {
                    0 => 1,
                    1 => 2,
                    _ => 3,
                };
                found = Some(descriptor);
                break;
            }
        }
        found
    } else {
        let mut found = match &catalog.tertiary_directory {
            Some(tertiary) => probe(tertiary)?,
            None => None,
        };
        if found.is_some() {
            served_tier = 1;
        }
        if found.is_none() {
            found = match &catalog.secondary_directory {
                Some(secondary) => probe(secondary)?,
                None => None,
            };
            if found.is_some() {
                served_tier = 2;
            }
        }
        if found.is_none() {
            found = probe(&catalog.directory)?;
            if found.is_some() {
                served_tier = 3;
            }
        }
        found
    };
    if let Some(descriptor) = descriptor {
        split_fd_tier_set(descriptor, served_tier);
    }
    EXPERT_OPEN_NS.fetch_add(
        open_started.elapsed().as_nanos() as u64,
        Ordering::Relaxed,
    );
    EXPERT_OPEN_COUNT.fetch_add(1, Ordering::Relaxed);
    match descriptor {
        Some(descriptor) => finish_catalog_open(catalog, source, descriptor).map(Some),
        None => Ok(None),
    }
}

/// Per-tile tier balancing (K3_TIER_BALANCE=1).
///
/// 65.6% of the corpus has TWO homes, but `resolve_expert_path` probes a fixed
/// order (dir_c -> hot -> dir_b -> primary) and takes the first hit, so the same
/// copy is chosen every time. A layer's routed experts are then a multinomial
/// load over the devices and the barrier waits on the busiest: measured
/// max/mean 1.55x, i.e. the busiest device carries 55% more than fair share on
/// every barrier.
///
/// Reordering the probe by current in-tile load turns those duplicate homes
/// into a choice. Simulated on real router traces: barrier 13.78ms -> 12.45ms
/// (-9.6%). Note a STATIC reshuffle is much worse than the fixed chain (-30.6%)
/// -- the existing order is a well-balanced static assignment, and only
/// per-barrier dynamics beat it. Hence the epoch reset rather than a global
/// counter, which would converge back to the static split.
///
/// Correctness is unaffected: the probe still falls through every tier, so a
/// different order finds the same file. Only WHICH copy is read changes, and
/// the copies are byte-identical corpus content.
const TIER_COUNT: usize = 3;
static TIER_LOAD: [AtomicU32; TIER_COUNT] =
    [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];

fn tier_balance_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("K3_TIER_BALANCE").is_ok_and(|value| value == "1"))
}

/// Measured four-way concurrent bandwidth, GB/s, in probe order
/// (tertiary=K3C, secondary=K3B, primary=K3A). Override as
/// K3_TIER_BW="7.01,5.63,5.93" after re-benchmarking or a re-plug.
fn tier_bandwidth() -> [f32; TIER_COUNT] {
    static BW: std::sync::OnceLock<[f32; TIER_COUNT]> = std::sync::OnceLock::new();
    *BW.get_or_init(|| {
        let mut bw = [7.01_f32, 5.63, 5.93];
        if let Ok(raw) = std::env::var("K3_TIER_BW") {
            for (slot, field) in bw.iter_mut().zip(raw.split(',')) {
                if let Ok(value) = field.trim().parse::<f32>() {
                    if value > 0.0 {
                        *slot = value;
                    }
                }
            }
        }
        bw
    })
}

/// Start a new balancing epoch. Called once per submitted union so the counts
/// describe THIS tile's load rather than the whole run.
pub(crate) fn tier_balance_new_epoch() {
    if !tier_balance_enabled() {
        return;
    }
    for slot in &TIER_LOAD {
        slot.store(0, Ordering::Relaxed);
    }
}

/// Probe order for this read: tiers sorted by projected completion time
/// (assigned reads / bandwidth), lightest first. Ties keep the canonical order.
fn tier_probe_order() -> [usize; TIER_COUNT] {
    let bw = tier_bandwidth();
    let mut order = [0_usize, 1, 2];
    order.sort_by(|&a, &b| {
        let cost = |i: usize| TIER_LOAD[i].load(Ordering::Relaxed) as f32 / bw[i];
        cost(a)
            .partial_cmp(&cost(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

fn catalog_io_error(
    operation: &str,
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
    error: io::Error,
) -> DeltafinError {
    DeltafinError::new(format!(
        "{operation} {}/{}: {error}",
        catalog.directory_path.display(),
        source.as_str(),
    ))
}

#[cfg(target_os = "macos")]
fn configure_catalog_cache_policy(
    file: &File,
    catalog: &DeferredExactCatalogInner,
    source: &DeferredSourceName,
) -> Result<()> {
    if catalog.cache_policy != CachePolicy::Streaming {
        return Ok(());
    }
    const F_NOCACHE: i32 = 48;
    unsafe extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    // SAFETY: the descriptor is live and F_NOCACHE accepts an integer.
    if unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) } == -1 {
        return Err(catalog_io_error(
            "enable F_NOCACHE",
            catalog,
            source,
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_catalog_cache_policy(
    _file: &File,
    _catalog: &DeferredExactCatalogInner,
    _source: &DeferredSourceName,
) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_cache_policy(file: &File, path: &Path, policy: CachePolicy) -> Result<()> {
    if policy != CachePolicy::Streaming {
        return Ok(());
    }
    const F_NOCACHE: i32 = 48;
    unsafe extern "C" {
        fn fcntl(fd: i32, command: i32, ...) -> i32;
    }
    // SAFETY: the descriptor is live, F_NOCACHE accepts an integer argument,
    // and this call does not transfer descriptor ownership.
    let result = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
    if result == -1 {
        return Err(io_error(
            "enable F_NOCACHE",
            path,
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn configure_cache_policy(_file: &File, _path: &Path, _policy: CachePolicy) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn drop_completed_cache(file: &File, policy: CachePolicy, offset: u64, length: usize) {
    if policy != CachePolicy::Streaming {
        return;
    }
    const POSIX_FADV_DONTNEED: i32 = 4;
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, length: i64, advice: i32) -> i32;
    }
    if let (Ok(offset), Ok(length)) = (i64::try_from(offset), i64::try_from(length)) {
        // Best effort, matching the existing Linux behavior. Some filesystems
        // legitimately reject advisory cache control.
        let _ = unsafe { posix_fadvise(file.as_raw_fd(), offset, length, POSIX_FADV_DONTNEED) };
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_completed_cache(_file: &File, _policy: CachePolicy, _offset: u64, _length: usize) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn deferred_source_open_flags_keep_the_symlink_guard() {
        // Asserted as bits, not a literal: O_NOFOLLOW's value differs between
        // x86_64 and aarch64 Linux, and every caller of this helper depends on
        // the guard actually being present.
        let flags = open_cloexec_nofollow();
        assert_ne!(flags & libc::O_NOFOLLOW, 0, "symlink guard dropped");
        assert_ne!(flags & libc::O_CLOEXEC, 0, "close-on-exec dropped");
    }

    static NEXT_TEST_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let serial = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "deltafin-storage-{}-{nonce}-{serial}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            let mut file = File::create(&path).unwrap();
            file.write_all(bytes).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn assembles_three_buffers_with_chunked_parallel_reads() {
        let directory = TestDirectory::new();
        let first = directory.write("first", &(0_u8..64).collect::<Vec<_>>());
        let second = directory.write("second", &(100_u8..164).collect::<Vec<_>>());
        let plan = ReadPlan::open(
            [
                Extent::new(&first, 4, BufferKind::Quantized, 0, 20),
                Extent::new(&second, 8, BufferKind::Quantized, 20, 16),
                Extent::new(&second, 30, BufferKind::Scales, 0, 12),
                Extent::zero(BufferKind::Other, 0, 3),
                Extent::new(&first, 40, BufferKind::Other, 3, 8),
            ],
            BufferLengths::new(36, 12, 11),
            7,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(4).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(&buffers.quantized()[..20], &(4_u8..24).collect::<Vec<_>>());
        assert_eq!(
            &buffers.quantized()[20..],
            &(108_u8..124).collect::<Vec<_>>()
        );
        assert_eq!(buffers.scales(), &(130_u8..142).collect::<Vec<_>>());
        assert_eq!(&buffers.other()[..3], &[0, 0, 0]);
        assert_eq!(&buffers.other()[3..], &(40_u8..48).collect::<Vec<_>>());
        assert_eq!(stats.bytes, 56);
        assert_eq!(stats.jobs, 11);
        assert_eq!(stats.workers, 4);
        assert_eq!(
            buffers.pointer(BufferKind::Quantized) as usize % BUFFER_ALIGNMENT,
            0
        );
        assert_eq!(
            buffers.allocation_lengths(),
            BufferLengths::new(BUFFER_ALIGNMENT, BUFFER_ALIGNMENT, BUFFER_ALIGNMENT)
        );
    }

    #[test]
    fn replacement_admission_is_zero_only_for_a_free_fitting_slot() {
        let small = BufferLengths::new(0, 0, 100);
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        assert_eq!(
            reader.replacement_admission_bytes(small).unwrap(),
            (3 * BUFFER_ALIGNMENT) as u64
        );
        let plan = ReadPlan::open(
            [Extent::zero(BufferKind::Other, 0, 100)],
            small,
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        // A busy slot is unknown to a future waiter, even when its observed
        // capacity happens to fit, so admission remains conservative.
        assert_eq!(
            reader.replacement_admission_bytes(small).unwrap(),
            (3 * BUFFER_ALIGNMENT) as u64
        );
        drop(buffers);
        assert_eq!(reader.replacement_admission_bytes(small).unwrap(), 0);

        let growth = BufferLengths::new(0, 0, BUFFER_ALIGNMENT + 1);
        assert_eq!(
            reader.replacement_admission_bytes(growth).unwrap(),
            (4 * BUFFER_ALIGNMENT) as u64
        );
    }

    #[test]
    fn explicit_capacity_reservation_grows_once_and_failure_leaves_a_free_slot() {
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        let small = BufferLengths::new(0, 0, 100);
        reader.reserve_capacity(small).unwrap();
        assert_eq!(reader.replacement_admission_bytes(small).unwrap(), 0);

        let large = BufferLengths::new(0, 0, BUFFER_ALIGNMENT + 1);
        reader.reserve_capacity(large).unwrap();
        assert_eq!(reader.replacement_admission_bytes(large).unwrap(), 0);
        reader.reserve_capacity(large).unwrap();
        assert_eq!(reader.replacement_admission_bytes(large).unwrap(), 0);

        assert!(
            reader
                .reserve_capacity(BufferLengths::new(0, 0, usize::MAX))
                .is_err()
        );
        assert!(reader.replacement_admission_bytes(small).is_ok());
    }

    #[test]
    fn wide_scale4_deferred_bounds_match_the_exact_nine_row_ceiling() {
        const ROUTED_EXPERTS_PER_ROW: usize = 16;
        const MAXIMUM_ROWS: usize = 9;
        let experts = ROUTED_EXPERTS_PER_ROW * MAXIMUM_ROWS;
        assert_eq!(MAX_DEFERRED_AUTHENTICATED_SOURCES, experts + 1);
        assert_eq!(MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS, experts * 2);
        assert_eq!(MAX_DEFERRED_MANIFEST_SOURCES, 64);
        assert!(MAX_DEFERRED_AUTHENTICATED_VERIFICATIONS < MAX_PLAN_JOBS);
        assert!(MAX_VECTORED_DESTINATIONS >= 4);
    }

    #[test]
    fn authenticated_gather_requalifies_each_deferred_descriptor_identity() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let path = directory.write("authenticated", &original);
        let plan = ReadPlan::open_deferred_authenticated(
            [
                Extent::new(&path, 2, BufferKind::Other, 0, 2),
                Extent::new(&path, 6, BufferKind::Other, 2, 2),
            ],
            [DeferredSourceVerification::new(
                &path,
                original.len() as u64,
                0,
                original.len(),
                crate::packfile::digest_bytes(&original),
            )],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.persistent_source_count(), 0);
        assert_eq!(plan.jobs(), 1);
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        drop(buffers);

        // A second admission reopens and reauthenticates the path. It may not
        // inherit the first descriptor's successful digest bit. Corrupt a gap
        // that is authenticated but not copied to prove the one-pass gather
        // still hashes every byte in the enclosing contract.
        fs::write(&path, [1_u8, 2, 3, 4, 0xff, 6, 7, 8]).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("replacement authenticated source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn authenticated_gather_rejects_an_extent_outside_its_digest_range() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("partially-authenticated", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::new(&path, 2, BufferKind::Other, 0, 2)],
            [DeferredSourceVerification::new(
                &path,
                bytes.len() as u64,
                0,
                2,
                crate::packfile::digest_bytes(&bytes[..2]),
            )],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("outside every authenticated range")
        );
    }

    #[test]
    fn authenticated_gather_rejects_ambiguous_overlapping_digest_ranges() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4, 5, 6];
        let path = directory.write("overlapping-authentication", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::new(&path, 2, BufferKind::Other, 0, 2)],
            [
                DeferredSourceVerification::new(
                    &path,
                    bytes.len() as u64,
                    0,
                    4,
                    crate::packfile::digest_bytes(&bytes[..4]),
                ),
                DeferredSourceVerification::new(
                    &path,
                    bytes.len() as u64,
                    2,
                    4,
                    crate::packfile::digest_bytes(&bytes[2..]),
                ),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(error.to_string().contains("ranges overlap"));
    }

    #[test]
    fn authenticated_gather_rejects_a_second_per_extent_digest_contract() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("double-authentication", &bytes);
        let error = ReadPlan::open_deferred_authenticated(
            [Extent::verified(
                &path,
                1,
                BufferKind::Other,
                0,
                2,
                [0xff; 32],
            )],
            [DeferredSourceVerification::new(
                &path,
                bytes.len() as u64,
                0,
                bytes.len(),
                crate::packfile::digest_bytes(&bytes),
            )],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(error.to_string().contains("per-extent digest"));
    }

    #[test]
    fn rejects_overlapping_or_out_of_file_extents_before_workers_start() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4]);
        assert!(
            ReadPlan::open(
                [
                    Extent::new(&path, 0, BufferKind::Other, 0, 3),
                    Extent::new(&path, 1, BufferKind::Other, 2, 2),
                ],
                BufferLengths::new(0, 0, 4),
                0,
                CachePolicy::Resident,
            )
            .unwrap_err()
            .to_string()
            .contains("overlapping")
        );
        assert!(
            ReadPlan::open(
                [Extent::new(&path, 3, BufferKind::Other, 0, 2)],
                BufferLengths::new(0, 0, 2),
                0,
                CachePolicy::Resident,
            )
            .unwrap_err()
            .to_string()
            .contains("exceeds")
        );
    }

    #[test]
    fn exact_length_contract_rejects_an_oversized_source() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4, 5]);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        assert!(plan.require_all_sources_exact_length(5).is_ok());
        assert!(
            plan.require_all_sources_exact_length(4)
                .unwrap_err()
                .to_string()
                .contains("canonical length is 4")
        );
    }

    #[test]
    fn deferred_exact_sources_open_in_workers_without_persistent_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-deferred", &[1, 2, 3, 4]);
        let second = directory.write("second-deferred", &[5, 6, 7, 8]);
        let plan = ReadPlan::open_deferred_exact(
            [
                Extent::new(&first, 0, BufferKind::Other, 0, 4),
                Extent::new(&second, 0, BufferKind::Other, 4, 4),
            ],
            BufferLengths::new(0, 0, 8),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);

        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stats.jobs, 2);
    }

    #[test]
    fn deferred_exact_source_validates_length_on_the_open_descriptor() {
        let directory = TestDirectory::new();
        let path = directory.write("wrong-length", &[1, 2, 3, 4, 5]);
        let plan = ReadPlan::open_deferred_exact(
            [Extent::new(&path, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("wrong-sized deferred source was unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));
    }

    #[test]
    fn deferred_exact_source_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("target", &[1, 2, 3, 4]);
        let link = directory.0.join("link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_deferred_exact(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
            4,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("deferred source symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_manifest_reads_variable_whole_file_lengths_without_retained_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-manifest", &[1, 2, 3]);
        let second = directory.write("second-manifest", &[4, 5, 6, 7, 8]);
        let plan = ReadPlan::open_deferred_manifest(
            [
                Extent::new(&first, 0, BufferKind::Other, 0, 3),
                Extent::new(&second, 0, BufferKind::Other, 3, 5),
            ],
            BufferLengths::new(0, 0, 8),
            2,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);
        // Do not sample the process-wide descriptor table here: Rust runs
        // unrelated storage/server tests concurrently, so their transient
        // sockets and files make that observation inherently racy. The plan's
        // zero persistent-source count is the local invariant; the ignored
        // single-test audit below proves worker descriptors close in practice.

        let reader = Reader::new(2).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Demand).unwrap();
        let completed_batch = Arc::clone(&ticket.batch);
        let (buffers, stats) = ticket.wait().unwrap();
        assert_eq!(buffers.other(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.jobs, 5);
        let BatchSources::Plan {
            deferred_files: Some(files),
            ..
        } = &completed_batch.sources
        else {
            panic!("manifest batch did not own deferred descriptor slots")
        };
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|slot| matches!(slot.get(), Some(Ok(_)))));
    }

    #[test]
    fn deferred_ranges_gather_partial_heterogeneous_sources_without_retained_descriptors() {
        let directory = TestDirectory::new();
        let first = directory.write("first-ranges", &[1, 2, 3, 4, 5, 6]);
        let second = directory.write("second-ranges", &[7, 8, 9, 10]);
        let plan = ReadPlan::open_deferred_ranges(
            [
                Extent::new(&first, 1, BufferKind::Other, 0, 2),
                Extent::new(&first, 4, BufferKind::Other, 2, 2),
                Extent::new(&second, 0, BufferKind::Other, 4, 1),
                Extent::new(&second, 2, BufferKind::Other, 5, 2),
            ],
            [
                DeferredSourceLength::new(&first, 6),
                DeferredSourceLength::new(&second, 4),
            ],
            BufferLengths::new(0, 0, 7),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.source_count(), 2);
        assert_eq!(plan.persistent_source_count(), 0);
        assert_eq!(plan.jobs(), 4);
        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), &[2, 3, 5, 6, 7, 9, 10]);
        assert_eq!(stats.jobs, 4);
    }

    #[test]
    fn identity_pinned_deferred_range_rejects_same_length_path_replacement() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4];
        let replacement = [9_u8, 8, 7, 6];
        let path = directory.write("identity-pinned-range", &original);
        let source =
            DeferredSourceLength::new_with_live_identity(&path, original.len() as u64).unwrap();
        let original_identity = source.identity().unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [Extent::new(&path, 0, BufferKind::Other, 0, original.len())],
            [source],
            BufferLengths::new(0, 0, original.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();

        let displaced = directory.0.join("identity-pinned-range-original");
        fs::rename(&path, displaced).unwrap();
        fs::write(&path, replacement).unwrap();
        let replacement_source =
            DeferredSourceLength::new_with_live_identity(&path, replacement.len() as u64).unwrap();
        assert_ne!(replacement_source.identity().unwrap(), original_identity);

        let error = match Reader::new(1).unwrap().read(&plan) {
            Ok(_) => panic!("same-length replacement satisfied an identity-pinned range"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("identity changed before range gather")
        );

        // Identity pinning is opt-in: the original exact-length contract still
        // admits the replacement and reads from the descriptor opened by the
        // worker for this batch.
        let unpinned = ReadPlan::open_deferred_ranges(
            [Extent::new(
                &path,
                0,
                BufferKind::Other,
                0,
                replacement.len(),
            )],
            [DeferredSourceLength::new(&path, replacement.len() as u64)],
            BufferLengths::new(0, 0, replacement.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let (buffers, _) = Reader::new(1).unwrap().read(&unpinned).unwrap();
        assert_eq!(buffers.other(), replacement);
    }

    #[test]
    fn identity_pinned_deferred_range_rechecks_descriptor_after_read_completion() {
        use std::os::unix::fs::FileExt;

        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("identity-pinned-tail-check", &bytes);
        let source =
            DeferredSourceLength::new_with_live_identity(&path, bytes.len() as u64).unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [Extent::new(&path, 0, BufferKind::Other, 0, bytes.len())],
            [source],
            BufferLengths::new(0, 0, bytes.len()),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Demand).unwrap();
        while !ticket.is_ready() {
            std::thread::yield_now();
        }

        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert_eq!(file.write_at(&[9], 0).unwrap(), 1);
        file.sync_all().unwrap();
        let error = match ticket.wait() {
            Ok(_) => panic!("post-read mutation escaped the descriptor identity tail check"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("identity changed during range gather")
        );
    }

    #[test]
    fn deferred_vectored_read_scatters_one_contiguous_source_range_in_one_job() {
        let directory = TestDirectory::new();
        let path = directory.write("vectored-range", &(0_u8..16).collect::<Vec<_>>());
        let extent = Extent::vectored(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Scales, 0, 2),
                VectoredDestination::new(BufferKind::Other, 3, 2),
                VectoredDestination::new(BufferKind::Quantized, 0, 1),
            ],
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, 16)],
            BufferLengths::new(1, 2, 5),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        assert_eq!(plan.jobs(), 1);
        assert_eq!(plan.logical_bytes(), 8);

        let (buffers, stats) = Reader::new(2).unwrap().read(&plan).unwrap();
        assert_eq!(buffers.quantized(), &[9]);
        assert_eq!(buffers.scales(), &[5, 6]);
        assert_eq!(buffers.other(), &[2, 3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.workers, 1);
    }

    #[test]
    fn verified_vectored_read_hashes_destinations_in_contiguous_source_order() {
        let directory = TestDirectory::new();
        let bytes: Vec<_> = (0_u8..16).collect();
        let path = directory.write("verified-vectored-range", &bytes);
        let extent = Extent::vectored_verified(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Scales, 0, 2),
                VectoredDestination::new(BufferKind::Other, 3, 2),
                VectoredDestination::new(BufferKind::Quantized, 0, 1),
            ],
            crate::packfile::digest_bytes(&bytes[2..10]),
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, bytes.len() as u64)],
            BufferLengths::new(1, 2, 5),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();

        let (buffers, stats) = Reader::new(2).unwrap().read(&plan).unwrap();
        assert_eq!(buffers.quantized(), &[9]);
        assert_eq!(buffers.scales(), &[5, 6]);
        assert_eq!(buffers.other(), &[2, 3, 4, 7, 8]);
        assert_eq!(stats.jobs, 1);
        drop(buffers);

        // The same deferred plan reopens its source on each batch. A previous
        // successful digest must never qualify a later descriptor generation.
        let mut changed = bytes;
        changed[6] ^= 0xff;
        fs::write(&path, changed).unwrap();
        let error = match Reader::new(2).unwrap().read(&plan) {
            Ok(_) => panic!("same-plan vectored reread reused a stale digest result"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn verified_vectored_read_rejects_corruption_before_publication() {
        let directory = TestDirectory::new();
        let original: Vec<_> = (0_u8..16).collect();
        let path = directory.write("corrupt-verified-vectored", &original);
        let extent = Extent::vectored_verified(
            &path,
            2,
            [
                VectoredDestination::new(BufferKind::Other, 0, 3),
                VectoredDestination::new(BufferKind::Other, 3, 5),
            ],
            crate::packfile::digest_bytes(&original[2..10]),
        )
        .unwrap();
        let plan = ReadPlan::open_deferred_ranges(
            [extent],
            [DeferredSourceLength::new(&path, original.len() as u64)],
            BufferLengths::new(0, 0, 8),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let mut corrupted = original;
        corrupted[6] ^= 0xff;
        fs::write(&path, corrupted).unwrap();

        let error = match Reader::new(2).unwrap().read(&plan) {
            Ok(_) => panic!("corrupted verified vectored source was published"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("failed SHA-256 verification"));
    }

    #[test]
    fn deferred_vectored_read_retains_live_exact_length_and_no_follow_contracts() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("vectored-target", &[1, 2, 3, 4, 5]);
        let link = directory.0.join("vectored-link");
        symlink(&target, &link).unwrap();
        let make_extent = |path: &Path| {
            Extent::vectored(
                path,
                1,
                [
                    VectoredDestination::new(BufferKind::Other, 0, 1),
                    VectoredDestination::new(BufferKind::Other, 1, 2),
                ],
            )
            .unwrap()
        };

        let wrong_length = ReadPlan::open_deferred_ranges(
            [make_extent(&target)],
            [DeferredSourceLength::new(&target, 4)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&wrong_length) {
            Ok(_) => panic!("wrong-sized vectored source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));

        let symlinked = ReadPlan::open_deferred_ranges(
            [make_extent(&link)],
            [DeferredSourceLength::new(&link, 5)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&symlinked) {
            Ok(_) => panic!("vectored source symlink was followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_range_plan_rejects_manually_constructed_empty_vectored_extents() {
        let directory = TestDirectory::new();
        let path = directory.write("malformed-vectored", &[1, 2, 3, 4]);
        for destinations in [
            Vec::new().into_boxed_slice(),
            vec![VectoredDestination::new(BufferKind::Other, 0, 0)].into_boxed_slice(),
        ] {
            let error = ReadPlan::open_deferred_ranges(
                [Extent::ReadVectored {
                    path: path.clone(),
                    source_offset: 0,
                    destinations,
                    expected_digest: Some(crate::packfile::digest_bytes(&[])),
                }],
                [DeferredSourceLength::new(&path, 4)],
                BufferLengths::new(0, 0, 0),
                0,
                CachePolicy::Streaming,
            )
            .unwrap_err();
            assert!(error.to_string().contains("non-empty destinations"));
        }
    }

    #[test]
    fn deferred_ranges_reject_missing_duplicate_unused_and_wrong_length_contracts() {
        let directory = TestDirectory::new();
        let path = directory.write("range-contract", &[1, 2, 3, 4]);
        let other = directory.write("unused-range-contract", &[5, 6]);
        let extent = || Extent::new(&path, 1, BufferKind::Other, 0, 2);

        let missing = ReadPlan::open_deferred_ranges(
            [extent()],
            [DeferredSourceLength::new(&other, 2)],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(missing.to_string().contains("no exact-length contract"));

        let duplicate = ReadPlan::open_deferred_ranges(
            [extent()],
            [
                DeferredSourceLength::new(&path, 4),
                DeferredSourceLength::new(&path, 4),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(duplicate.to_string().contains("declared more than once"));

        let unused = ReadPlan::open_deferred_ranges(
            [extent()],
            [
                DeferredSourceLength::new(&path, 4),
                DeferredSourceLength::new(&other, 2),
            ],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap_err();
        assert!(unused.to_string().contains("unused source contract"));

        let plan = ReadPlan::open_deferred_ranges(
            [extent()],
            [DeferredSourceLength::new(&path, 5)],
            BufferLengths::new(0, 0, 2),
            0,
            CachePolicy::Streaming,
        )
        .unwrap();
        let error = match Reader::new(1).unwrap().read(&plan) {
            Ok(_) => panic!("wrong-sized deferred range source was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 5"));
    }

    #[test]
    fn persistent_manifest_opens_once_and_pins_the_validated_inode() {
        let directory = TestDirectory::new();
        let original = [1_u8, 2, 3, 4];
        let path = directory.write("persistent-manifest", &original);
        let plan = ReadPlan::open_persistent_deferred_manifest(
            [Extent::new(&path, 0, BufferKind::Other, 0, original.len())],
            BufferLengths::new(0, 0, original.len()),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(plan.persistent_source_count(), 1);
        assert_eq!(plan.opened_persistent_source_count(), 0);

        let reader = Reader::new(1).unwrap();
        let (first, _) = reader.read(&plan).unwrap();
        assert_eq!(first.other(), original);
        drop(first);
        assert_eq!(plan.opened_persistent_source_count(), 1);

        let displaced = directory.0.join("persistent-manifest-original");
        fs::rename(&path, &displaced).unwrap();
        fs::write(&path, [9_u8, 9, 9, 9]).unwrap();
        let (second, _) = reader.read(&plan).unwrap();
        assert_eq!(second.other(), original);
        drop(second);
        assert_eq!(plan.opened_persistent_source_count(), 1);
    }

    #[test]
    fn persistent_manifest_first_open_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("persistent-target", &[1, 2, 3, 4]);
        let link = directory.0.join("persistent-link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_persistent_deferred_manifest(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("persistent manifest unexpectedly followed a symlink"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
        assert_eq!(plan.opened_persistent_source_count(), 0);
    }

    #[test]
    fn deferred_manifest_validates_each_live_source_length() {
        let directory = TestDirectory::new();
        let oversized = directory.write("oversized-manifest", &[1, 2, 3, 4, 5]);
        let short = directory.write("short-manifest", &[1, 2, 3]);
        let reader = Reader::new(2).unwrap();
        for path in [&oversized, &short] {
            let plan = ReadPlan::open_deferred_manifest(
                [Extent::new(path, 0, BufferKind::Other, 0, 4)],
                BufferLengths::new(0, 0, 4),
                0,
                CachePolicy::Resident,
            )
            .unwrap();
            let error = match reader.read(&plan) {
                Ok(_) => panic!("wrong-sized manifest source was unexpectedly accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("expected exact length 4"));
        }
    }

    #[test]
    fn deferred_manifest_rejects_partial_or_conflicting_source_contracts() {
        let directory = TestDirectory::new();
        let path = directory.0.join("need-not-exist");
        let partial = ReadPlan::open_deferred_manifest(
            [Extent::new(&path, 1, BufferKind::Other, 0, 3)],
            BufferLengths::new(0, 0, 3),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(partial.to_string().contains("offset zero"));

        let conflicting = ReadPlan::open_deferred_manifest(
            [
                Extent::new(&path, 0, BufferKind::Other, 0, 3),
                Extent::new(&path, 0, BufferKind::Other, 3, 4),
            ],
            BufferLengths::new(0, 0, 7),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(
            conflicting
                .to_string()
                .contains("conflicting whole-file lengths")
        );
    }

    #[test]
    fn deferred_manifest_has_a_hard_per_batch_descriptor_bound() {
        let directory = TestDirectory::new();
        let extents = (0..=MAX_DEFERRED_MANIFEST_SOURCES)
            .map(|index| {
                Extent::new(
                    directory.0.join(format!("source-{index}")),
                    0,
                    BufferKind::Other,
                    index,
                    1,
                )
            })
            .collect::<Vec<_>>();
        let error = ReadPlan::open_deferred_manifest(
            extents,
            BufferLengths::new(0, 0, MAX_DEFERRED_MANIFEST_SOURCES + 1),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(error.to_string().contains("bounded maximum is 64"));
    }

    #[test]
    fn deferred_manifest_never_follows_a_symlink() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        let target = directory.write("manifest-target", &[1, 2, 3, 4]);
        let link = directory.0.join("manifest-link");
        symlink(target, &link).unwrap();
        let plan = ReadPlan::open_deferred_manifest(
            [Extent::new(&link, 0, BufferKind::Other, 0, 4)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("deferred manifest symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn deferred_catalog_reads_integer_selected_sources_in_adjacent_order() {
        let directory = TestDirectory::new();
        directory.write("first.bin", &[1, 2, 3, 4]);
        directory.write("second.bin", &[5, 6, 7, 8]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [
                DeferredSourceName::new("first.bin").unwrap(),
                DeferredSourceName::new("second.bin").unwrap(),
            ],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(catalog.source_count(), 2);
        assert_eq!(catalog.exact_source_length(), 4);
        assert_eq!(catalog.source_name(0), Some("first.bin"));

        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader
            .read_deferred_exact(&catalog, &[1, 0], BufferKind::Other)
            .unwrap();
        assert_eq!(buffers.other(), &[5, 6, 7, 8, 1, 2, 3, 4]);
        assert_eq!(stats.bytes, 8);
        assert_eq!(stats.jobs, 2);
        assert_eq!(stats.workers, 2);
    }

    #[test]
    fn deferred_catalog_rejects_unsafe_names_and_bounds_requests() {
        assert!(DeferredSourceName::new("").is_err());
        assert!(DeferredSourceName::new(".").is_err());
        assert!(DeferredSourceName::new("..").is_err());
        assert!(DeferredSourceName::new("nested/file").is_err());
        assert!(DeferredSourceName::new(&"x".repeat(32)).is_err());

        let directory = TestDirectory::new();
        directory.write("only.bin", &[1, 2, 3, 4]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("only.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        assert!(
            reader
                .submit_deferred_exact(&catalog, &[], BufferKind::Other, ReadPriority::Demand)
                .is_err()
        );
        assert!(
            reader
                .submit_deferred_exact(&catalog, &[1], BufferKind::Other, ReadPriority::Demand)
                .is_err()
        );
        assert!(
            reader
                .submit_deferred_exact(
                    &catalog,
                    &[0; MAX_INLINE_DEFERRED_FILES + 1],
                    BufferKind::Other,
                    ReadPriority::Demand,
                )
                .is_err()
        );
    }

    #[test]
    fn deferred_catalog_openat_never_follows_source_or_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new();
        directory.write("target.bin", &[1, 2, 3, 4]);
        symlink(directory.0.join("target.bin"), directory.0.join("link.bin")).unwrap();
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("link.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read_deferred_exact(&catalog, &[0], BufferKind::Other) {
            Ok(_) => panic!("catalog source symlink was unexpectedly followed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("non-symlink"));

        let parent = TestDirectory::new();
        let directory_link = parent.0.join("directory-link");
        symlink(&directory.0, &directory_link).unwrap();
        assert!(
            DeferredExactCatalog::open(
                &directory_link,
                [DeferredSourceName::new("target.bin").unwrap()],
                4,
                CachePolicy::Resident,
            )
            .is_err()
        );
    }

    #[test]
    fn deferred_catalog_validates_exact_length_on_the_openat_descriptor() {
        let directory = TestDirectory::new();
        directory.write("wrong.bin", &[1, 2, 3, 4, 5]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("wrong.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read_deferred_exact(&catalog, &[0], BufferKind::Other) {
            Ok(_) => panic!("wrong-sized catalog source was unexpectedly accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("expected exact length 4"));
    }

    #[test]
    #[ignore = "global descriptor-count audit; run this test alone"]
    fn deferred_catalog_closes_every_ephemeral_source_descriptor() {
        let directory = TestDirectory::new();
        directory.write("source.bin", &[1, 2, 3, 4]);
        let catalog = DeferredExactCatalog::open(
            &directory.0,
            [DeferredSourceName::new("source.bin").unwrap()],
            4,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(2).unwrap();
        let before = count_open_descriptors().expect("supported host exposes descriptor count");
        for _ in 0..64 {
            let (buffers, _) = reader
                .read_deferred_exact(&catalog, &[0], BufferKind::Other)
                .unwrap();
            assert_eq!(buffers.other(), &[1, 2, 3, 4]);
            drop(buffers);
        }
        let after = count_open_descriptors().expect("supported host exposes descriptor count");
        assert_eq!(after, before, "catalog source descriptor leaked");
    }

    #[test]
    fn authenticated_extent_hashes_the_first_read_without_a_second_io_pass() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(32 * 1024).collect();
        let path = directory.write("authenticated", &bytes);
        let plan = ReadPlan::open(
            [Extent::verified(
                &path,
                0,
                BufferKind::Other,
                0,
                bytes.len(),
                crate::packfile::digest_bytes(&bytes),
            )],
            BufferLengths::new(0, 0, bytes.len()),
            // A verified range is already one canonical integrity unit and
            // must not be silently split by the ordinary scheduling chunk.
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        assert_eq!(plan.jobs(), 1);
        assert!(!plan.verified_extents[0].load(Ordering::Acquire));

        let reader = Reader::new(2).unwrap();
        let (buffers, _) = reader.read(&plan).unwrap();
        assert_eq!(buffers.other(), bytes);
        assert!(plan.verified_extents[0].load(Ordering::Acquire));

        // A later read through the same immutable plan and opened descriptor
        // reuses the successful first-read qualification.
        let (again, _) = reader.read(&plan).unwrap();
        assert_eq!(again.other(), bytes);
    }

    #[test]
    fn authenticated_extent_never_publishes_bytes_with_the_wrong_digest() {
        let directory = TestDirectory::new();
        let bytes = [1_u8, 2, 3, 4];
        let path = directory.write("corrupt", &bytes);
        let plan = ReadPlan::open(
            [Extent::verified(
                &path,
                0,
                BufferKind::Other,
                0,
                bytes.len(),
                [0; 32],
            )],
            BufferLengths::new(0, 0, bytes.len()),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::new(1).unwrap();
        let error = match reader.read(&plan) {
            Ok(_) => panic!("wrong authenticated bytes were unexpectedly published"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("SHA-256"));
        assert!(!plan.verified_extents[0].load(Ordering::Acquire));
    }

    #[test]
    fn rejects_implicit_gaps_and_missing_trailing_bytes() {
        let directory = TestDirectory::new();
        let path = directory.write("weights", &[1, 2, 3, 4]);
        let gap = ReadPlan::open(
            [
                Extent::new(&path, 0, BufferKind::Other, 0, 2),
                Extent::new(&path, 3, BufferKind::Other, 3, 1),
            ],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(gap.to_string().contains("Extent::zero"));

        let trailing = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Other, 0, 3)],
            BufferLengths::new(0, 0, 4),
            0,
            CachePolicy::Resident,
        )
        .unwrap_err();
        assert!(trailing.to_string().contains("3..4"));
    }

    #[test]
    fn reuses_fixed_workers_across_many_batches() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(32 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(3, 1).unwrap();
        let mut arena_pointer = None;
        for _ in 0..20 {
            let (result, stats) = reader.read(&plan).unwrap();
            assert_eq!(result.quantized(), bytes);
            assert_eq!(stats.workers, 3);
            let pointer = result.pointer(BufferKind::Quantized);
            assert_eq!(*arena_pointer.get_or_insert(pointer), pointer);
        }
    }

    #[test]
    fn empty_plan_returns_without_scheduling_workers() {
        let plan =
            ReadPlan::open([], BufferLengths::default(), 1024, CachePolicy::Resident).unwrap();
        let reader = Reader::new(2).unwrap();
        let (buffers, stats) = reader.read(&plan).unwrap();
        assert!(buffers.quantized().is_empty());
        assert_eq!(stats.jobs, 0);
        assert_eq!(stats.workers, 0);
    }

    #[test]
    fn asynchronous_tickets_preserve_exact_bytes() {
        let directory = TestDirectory::new();
        let first_bytes: Vec<u8> = (0..=127).cycle().take(16 * 1024).collect();
        let second_bytes: Vec<u8> = (128..=255).cycle().take(16 * 1024).collect();
        let first = directory.write("first", &first_bytes);
        let second = directory.write("second", &second_bytes);
        let first_plan = ReadPlan::open(
            [Extent::new(
                &first,
                0,
                BufferKind::Quantized,
                0,
                first_bytes.len(),
            )],
            BufferLengths::new(first_bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let second_plan = ReadPlan::open(
            [Extent::new(
                &second,
                0,
                BufferKind::Quantized,
                0,
                second_bytes.len(),
            )],
            BufferLengths::new(second_bytes.len(), 0, 0),
            1024,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 2).unwrap();
        let prefetch = reader.submit(&first_plan, ReadPriority::Prefetch).unwrap();
        let demand = reader.submit(&second_plan, ReadPriority::Demand).unwrap();
        let (demand_buffers, _) = demand.wait().unwrap();
        let (prefetch_buffers, _) = prefetch.wait().unwrap();
        assert_eq!(demand_buffers.quantized(), second_bytes);
        assert_eq!(prefetch_buffers.quantized(), first_bytes);
    }

    #[test]
    fn arena_is_bounded_and_returns_a_slot_when_the_cpu_lease_drops() {
        let arena = BufferArena::new(1).unwrap();
        let lengths = BufferLengths::new(64, 8, 0);
        let first = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let first_pointer = first.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert!(
            arena
                .acquire(lengths, false, ReadPriority::Demand)
                .unwrap()
                .is_none()
        );
        drop(first);
        let second = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            second.buffers().get(BufferKind::Quantized).pointer.as_ptr(),
            first_pointer
        );
    }

    #[test]
    fn arena_retire_hook_skips_initial_and_fitting_allocations() {
        let calls = Arc::new(AtomicUsize::new(0));
        let hook_calls = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let first = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = first.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(first);

        let fitting = arena
            .acquire(BufferLengths::new(32, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            fitting
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(fitting);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn arena_retire_hook_runs_before_the_old_slab_final_arc_drops() {
        let retired = Arc::new(Mutex::new(None::<Weak<SharedBuffers>>));
        let retired_for_hook = Arc::clone(&retired);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            calls_for_hook.fetch_add(1, Ordering::SeqCst);
            assert!(
                retired_for_hook
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(Weak::upgrade)
                    .is_some(),
                "retirement hook must run while the old slab is still owned"
            );
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let old = Arc::downgrade(initial.buffers.as_ref().unwrap());
        *retired.lock().unwrap() = Some(old.clone());
        drop(initial);

        let grown = arena
            .acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(old.upgrade().is_none());
        // Do not make final arena teardown inspect the already-retired weak
        // pointer; this test is solely about the growth boundary.
        *retired.lock().unwrap() = Some(Arc::downgrade(grown.buffers.as_ref().unwrap()));
        drop(grown);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arena_retire_hook_error_restores_the_exact_slab_and_slot() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            if calls_for_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(DeltafinError::new("injected cache flush failure"))
            } else {
                Ok(())
            }
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = initial
            .buffers()
            .get(BufferKind::Quantized)
            .pointer
            .as_ptr();
        let old = Arc::downgrade(initial.buffers.as_ref().unwrap());
        drop(initial);

        let error = match arena.acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
        {
            Err(error) => error,
            Ok(_) => panic!("injected retirement-hook error should reject arena growth"),
        };
        assert!(error.to_string().contains("injected cache flush failure"));
        assert!(old.upgrade().is_some());
        let restored = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            restored
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(restored);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arena_retire_hook_panic_is_caught_and_restores_the_slot() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_hook = Arc::clone(&calls);
        let hook: BufferRetireHook = Arc::new(move || {
            if calls_for_hook.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("injected cache flush panic");
            }
            Ok(())
        });
        let arena = BufferArena::new_with_retire_hook(1, Some(hook)).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let pointer = initial
            .buffers()
            .get(BufferKind::Quantized)
            .pointer
            .as_ptr();
        drop(initial);

        let error = match arena.acquire(BufferLengths::new(128, 0, 0), false, ReadPriority::Demand)
        {
            Err(error) => error,
            Ok(_) => panic!("injected retirement-hook panic should reject arena growth"),
        };
        assert!(error.to_string().contains("hook panicked"));
        let restored = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            restored
                .buffers()
                .get(BufferKind::Quantized)
                .pointer
                .as_ptr(),
            pointer
        );
        drop(restored);
        drop(arena);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn descriptor_budget_rejects_before_overcommit_and_releases_exactly() {
        let budget = DescriptorBudget::fixed(2);
        let reservation = budget.reserve(2).unwrap();
        let error = budget.reserve(1).unwrap_err();
        assert!(error.to_string().contains("only 0 remain"));
        assert_eq!(*budget.in_use.lock().unwrap(), 2);
        drop(reservation);
        assert_eq!(*budget.in_use.lock().unwrap(), 0);
        let replacement = budget.reserve(2).unwrap();
        assert_eq!(*budget.in_use.lock().unwrap(), 2);
        drop(replacement);
    }

    #[test]
    fn arena_reuses_the_smallest_fitting_free_slot() {
        let arena = BufferArena::new(2).unwrap();
        let small_lengths = BufferLengths::new(64, 0, 0);
        let large_lengths = BufferLengths::new(4096, 0, 0);

        let small = arena
            .acquire(small_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let small_pointer = small.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        // Slot zero remains occupied, forcing this allocation into slot one.
        let large = arena
            .acquire(large_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let large_pointer = large.buffers().get(BufferKind::Quantized).pointer.as_ptr();
        assert_ne!(small_pointer, large_pointer);
        drop((small, large));

        // Both slots are now free. First-free selection would grow slot zero;
        // best-fit selection must reuse slot one's already sufficient slab.
        let reused = arena
            .acquire(large_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(
            reused.buffers().get(BufferKind::Quantized).pointer.as_ptr(),
            large_pointer
        );
    }

    #[test]
    fn arena_retires_an_undersized_slab_before_growth_allocation() {
        let arena = BufferArena::new(1).unwrap();
        let initial = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let retired = Arc::downgrade(initial.buffers.as_ref().unwrap());
        drop(initial);

        // This fails before attempting a real allocation, after the reserved
        // slot has retired its old buffers. Retaining the old slab until a new
        // one succeeded would leave this Weak reference live.
        assert!(
            arena
                .acquire(
                    BufferLengths::new(usize::MAX, 0, 0),
                    false,
                    ReadPriority::Demand,
                )
                .is_err()
        );
        assert!(retired.upgrade().is_none());

        let retry = arena
            .acquire(BufferLengths::new(64, 0, 0), false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        assert_eq!(retry.lengths.quantized, 64);
    }

    #[test]
    fn prefetch_cannot_consume_the_last_demand_arena_slot() {
        let arena = BufferArena::new(2).unwrap();
        let lengths = BufferLengths::new(64, 0, 0);
        let prefetch = arena
            .acquire(lengths, false, ReadPriority::Prefetch)
            .unwrap()
            .unwrap();
        assert!(
            arena
                .acquire(lengths, false, ReadPriority::Prefetch)
                .unwrap()
                .is_none()
        );
        let demand = arena
            .acquire(lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        drop((demand, prefetch));
    }

    #[test]
    fn demand_preempts_prefetch_but_prefetch_cannot_starve() {
        let mut queues = PriorityQueues::new();
        queues.push(ReadPriority::Prefetch, 10_000);
        for demand in 0..(MAX_DEMAND_STREAK + 2) {
            queues.push(ReadPriority::Demand, demand);
        }
        for expected in 0..MAX_DEMAND_STREAK {
            assert_eq!(queues.pop(), Some(expected));
        }
        assert_eq!(queues.pop(), Some(10_000));
        assert_eq!(queues.pop(), Some(MAX_DEMAND_STREAK));

        let mut preemption = PriorityQueues::new();
        preemption.push(ReadPriority::Prefetch, 1);
        preemption.push(ReadPriority::Prefetch, 2);
        preemption.push(ReadPriority::Demand, 3);
        assert_eq!(preemption.pop(), Some(3));
        assert_eq!(preemption.pop(), Some(1));
    }

    #[test]
    fn worker_executes_only_a_bounded_quantum_before_requeue() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..(WORK_QUANTUM * 2 + 1) as u8).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        let arena = BufferArena::new(1).unwrap();
        let lease = arena
            .acquire(plan.buffer_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let batch = Batch::new(&plan, Arc::clone(&lease), ReadPriority::Prefetch);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Requeue));
        assert_eq!(batch.next_job.load(Ordering::Relaxed), WORK_QUANTUM);
        assert_eq!(
            batch.completion.remaining.load(Ordering::Relaxed),
            bytes.len() - WORK_QUANTUM
        );
        while matches!(batch.run_quantum(), QuantumOutcome::Requeue) {}
        batch.wait().unwrap();
        assert_eq!(
            lease
                .buffers()
                .get(BufferKind::Quantized)
                .as_slice(bytes.len()),
            bytes
        );
    }

    #[test]
    fn cancellation_claims_every_unstarted_job_before_arena_reuse() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (1..=(WORK_QUANTUM * 2 + 1) as u8).collect();
        let path = directory.write("cancel-weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            1,
            CachePolicy::Resident,
        )
        .unwrap();
        let arena = BufferArena::new(1).unwrap();
        let lease = arena
            .acquire(plan.buffer_lengths, false, ReadPriority::Demand)
            .unwrap()
            .unwrap();
        let batch = Batch::new(&plan, Arc::clone(&lease), ReadPriority::Prefetch);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Requeue));
        batch.cancel_unclaimed();
        assert_eq!(batch.completion.remaining.load(Ordering::Acquire), 0);
        assert!(matches!(batch.run_quantum(), QuantumOutcome::Idle));
        assert!(batch.wait().unwrap_err().to_string().contains("cancelled"));
        let stored = lease
            .buffers()
            .get(BufferKind::Quantized)
            .as_slice(bytes.len());
        assert_eq!(&stored[..WORK_QUANTUM], &bytes[..WORK_QUANTUM]);
        assert!(stored[WORK_QUANTUM..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn shutdown_drains_admitted_work_and_leaves_ticket_readable() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(64 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let ticket = reader.submit(&plan, ReadPriority::Prefetch).unwrap();
        drop(reader);
        let (buffers, _) = ticket.wait().unwrap();
        assert_eq!(buffers.quantized(), bytes);

        for _ in 0..32 {
            drop(Reader::new(2).unwrap());
        }
    }

    #[test]
    fn dropping_a_ticket_does_not_reuse_its_slot_before_io_finishes() {
        let directory = TestDirectory::new();
        let bytes: Vec<u8> = (0..=255).cycle().take(128 * 1024).collect();
        let path = directory.write("weights", &bytes);
        let plan = ReadPlan::open(
            [Extent::new(&path, 0, BufferKind::Quantized, 0, bytes.len())],
            BufferLengths::new(bytes.len(), 0, 0),
            64,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(2, 1).unwrap();
        let abandoned = reader.submit(&plan, ReadPriority::Demand).unwrap();
        drop(abandoned);
        // Admission may wait for an active syscall, but it cannot observe or
        // overwrite the abandoned slot until active I/O drains and every
        // unclaimed job has been atomically cancelled.
        let (buffers, _) = reader.read(&plan).unwrap();
        assert_eq!(buffers.quantized(), bytes);
    }

    #[test]
    fn failed_read_releases_its_arena_slot() {
        let directory = TestDirectory::new();
        let broken_path = directory.write("broken", &[1, 2, 3, 4]);
        let broken = ReadPlan::open(
            [Extent::new(&broken_path, 0, BufferKind::Quantized, 0, 4)],
            BufferLengths::new(4, 0, 0),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        File::options()
            .write(true)
            .open(&broken_path)
            .unwrap()
            .set_len(0)
            .unwrap();

        let good_bytes = [9, 8, 7, 6];
        let good_path = directory.write("good", &good_bytes);
        let good = ReadPlan::open(
            [Extent::new(
                &good_path,
                0,
                BufferKind::Quantized,
                0,
                good_bytes.len(),
            )],
            BufferLengths::new(good_bytes.len(), 0, 0),
            0,
            CachePolicy::Resident,
        )
        .unwrap();
        let reader = Reader::with_arena_capacity(1, 1).unwrap();
        assert!(reader.read(&broken).is_err());
        let (buffers, _) = reader.read(&good).unwrap();
        assert_eq!(buffers.quantized(), good_bytes);
    }
}
