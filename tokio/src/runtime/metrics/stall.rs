//! PSI-style stall accounting for the multi-threaded scheduler.
//!
//! Modeled on the Linux kernel's PSI (`kernel/sched/psi.c`): instead of keeping
//! a timeline of state transitions, each tracked condition keeps the instant it
//! last became true and the accumulated time it has been true. Readers add the
//! in-progress interval, so totals are exact integrals of the condition and
//! can be diffed over any window.

use crate::loom::sync::atomic::{AtomicBool, AtomicU64};
use crate::loom::sync::Arc;

use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::time::{Duration, Instant};

/// Accumulated time during which a condition held.
///
/// `open` is called by the thread that observes the condition becoming true,
/// `close_if` by any thread that observes it becoming false. Concurrent closes
/// are resolved by a compare-exchange on `start`, so each interval is
/// accounted for exactly once.
#[derive(Debug)]
pub(crate) struct StallTime {
    /// Basis for the nanosecond timestamps stored below.
    base: Instant,

    /// When the condition last became true, in nanoseconds since `base`
    /// (at least 1). Zero while the condition does not hold.
    start: AtomicU64,

    /// Nanoseconds the condition held across completed intervals.
    total: AtomicU64,

    /// Largest value returned by [`StallTime::total`], keeping it monotonic
    /// across racing readers and writers.
    reported: AtomicU64,
}

impl StallTime {
    pub(crate) fn new(base: Instant) -> StallTime {
        StallTime {
            base,
            start: AtomicU64::new(0),
            total: AtomicU64::new(0),
            reported: AtomicU64::new(0),
        }
    }

    fn now(&self) -> u64 {
        (self.base.elapsed().as_nanos() as u64).max(1)
    }

    /// Records that the condition holds. Reads the clock only when the
    /// condition was not already recorded as holding.
    pub(crate) fn open(&self) {
        if self.start.load(SeqCst) == 0 {
            let _ = self.start.compare_exchange(0, self.now(), SeqCst, Relaxed);
        }
    }

    /// Records that the condition no longer holds, if `still_clear` confirms
    /// it after the open interval has been observed.
    ///
    /// Callers that cannot exclude a concurrent `open` (stealers, inject queue
    /// poppers) re-check the condition here: a writer makes the condition true
    /// *before* calling `open`, so observing its new `start` implies observing
    /// the condition. The remaining race (re-check passes, then the condition
    /// becomes true while the old interval is still open) closes an interval
    /// early; the next owner-side sync reopens it.
    pub(crate) fn close_if(&self, still_clear: impl FnOnce() -> bool) {
        let start = self.start.load(SeqCst);
        if start == 0 || !still_clear() {
            return;
        }

        let now = self.now();
        if self
            .start
            .compare_exchange(start, 0, SeqCst, Relaxed)
            .is_ok()
        {
            self.total.fetch_add(now.saturating_sub(start), SeqCst);
        }
    }

    /// Reconciles the recorded state with `holds`, as observed by a thread
    /// that cannot race with other openers.
    pub(crate) fn set(&self, holds: bool) {
        if holds {
            self.open();
        } else {
            self.close_if(|| true);
        }
    }

    /// Total time the condition held, including the in-progress interval.
    pub(crate) fn total(&self) -> Duration {
        // Load `total` before `start`: a close racing between the two loads
        // then undercounts (the closed interval is missed) rather than
        // double-counting it. `reported` hides the transient undercount.
        let total = self.total.load(SeqCst);
        let start = self.start.load(SeqCst);
        let mut value = total;
        if start != 0 {
            value += self.now().saturating_sub(start);
        }

        let mut prev = self.reported.load(Relaxed);
        while value > prev {
            match self
                .reported
                .compare_exchange_weak(prev, value, Relaxed, Relaxed)
            {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }

        Duration::from_nanos(value.max(prev))
    }
}

/// Per-worker stall state, shared between the worker's core (owner) and other
/// workers (stealers, metrics readers).
#[derive(Debug)]
pub(crate) struct WorkerStall {
    /// Time during which at least one task was runnable on this worker (in
    /// its LIFO slot or local run queue) but not being polled.
    pub(crate) waiting: StallTime,

    /// Time this worker spent parked. Wall time minus this is the worker's
    /// non-idle time, used to weight per-worker stall like the kernel weights
    /// per-CPU stall.
    pub(crate) parked: StallTime,

    /// Whether the LIFO slot is occupied. The LIFO slot is not stealable, so
    /// a stealer that empties the run queue must not close `waiting` while
    /// this is set.
    pub(crate) lifo_occupied: AtomicBool,

    /// Whether tasks in the worker's defer list are waiting on another
    /// task's poll. Tasks that called `yield_now` sit in the defer list,
    /// runnable but in neither the LIFO slot nor the run queue, until the
    /// worker next parks or polls its drivers.
    pub(crate) deferred: AtomicBool,
}

impl WorkerStall {
    pub(crate) fn new(base: Instant) -> Arc<WorkerStall> {
        Arc::new(WorkerStall {
            waiting: StallTime::new(base),
            parked: StallTime::new(base),
            lifo_occupied: AtomicBool::new(false),
            deferred: AtomicBool::new(false),
        })
    }

    /// Whether runnable tasks are held where stealers cannot reach them.
    pub(crate) fn has_unstealable(&self) -> bool {
        self.lifo_occupied.load(SeqCst) || self.deferred.load(SeqCst)
    }

    /// Owner-side update of the defer list state. Becoming non-empty opens
    /// `waiting`; becoming empty only clears the flag, as the deferred tasks
    /// have just been scheduled and the next `sync` reconciles.
    pub(crate) fn set_deferred(&self, deferred: bool) {
        if self.deferred.load(Relaxed) != deferred {
            self.deferred.store(deferred, SeqCst);
        }
        if deferred {
            self.waiting.open();
        }
    }

    /// Owner-side reconciliation after any change to the LIFO slot or local
    /// run queue.
    pub(crate) fn sync(&self, lifo_occupied: bool, run_queue_has_tasks: bool) {
        if self.lifo_occupied.load(Relaxed) != lifo_occupied {
            self.lifo_occupied.store(lifo_occupied, SeqCst);
        }
        self.waiting
            .set(lifo_occupied || run_queue_has_tasks || self.deferred.load(Relaxed));
    }
}
