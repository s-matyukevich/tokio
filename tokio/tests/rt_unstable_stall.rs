#![warn(rust_2018_idioms)]
#![cfg(all(
    feature = "full",
    tokio_unstable,
    not(target_os = "wasi"),
    target_has_atomic = "64"
))]

//! Tests for PSI-style stall tracking (`enable_metrics_stall_tracking`).

use std::sync::mpsc;
use std::time::{Duration, Instant};

use tokio::runtime::{Builder, Runtime};

const BUSY: Duration = Duration::from_millis(200);

fn threaded(workers: usize) -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .enable_metrics_stall_tracking()
        .build()
        .unwrap()
}

fn spin(d: Duration) {
    let start = Instant::now();
    while start.elapsed() < d {
        std::hint::spin_loop();
    }
}

/// Runs on the CPU for `total`, yielding to the scheduler every `slice`.
async fn spin_yielding(total: Duration, slice: Duration) {
    let start = Instant::now();
    while start.elapsed() < total {
        spin(slice);
        tokio::task::yield_now().await;
    }
}

#[test]
fn disabled_by_default() {
    let rt = Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    let metrics = rt.metrics();
    rt.block_on(async {
        tokio::spawn(spin_yielding(Duration::from_millis(20), Duration::from_micros(50)))
            .await
            .unwrap();
    });

    assert!(!metrics.stall_tracking_enabled());
    assert_eq!(metrics.worker_total_stall_duration(0), Duration::ZERO);
    assert_eq!(metrics.worker_total_park_duration(0), Duration::ZERO);
    assert_eq!(metrics.global_queue_total_stall_duration(), Duration::ZERO);
}

#[test]
fn current_thread_reports_disabled() {
    let rt = Builder::new_current_thread()
        .enable_metrics_stall_tracking()
        .build()
        .unwrap();
    assert!(!rt.metrics().stall_tracking_enabled());
}

#[test]
fn idle_worker_is_parked_not_stalled() {
    let rt = threaded(2);
    let metrics = rt.metrics();
    assert!(metrics.stall_tracking_enabled());

    rt.block_on(async { tokio::time::sleep(BUSY).await });

    for worker in 0..2 {
        let stall = metrics.worker_total_stall_duration(worker);
        let parked = metrics.worker_total_park_duration(worker);
        assert!(stall < Duration::from_millis(10), "worker {worker} stall {stall:?}");
        assert!(parked > BUSY / 2, "worker {worker} parked {parked:?}");
    }
}

#[test]
fn lone_busy_task_does_not_stall() {
    // A single task yielding frequently is re-queued on every yield, but is
    // the next task to run, so the queue is non-empty only momentarily.
    let rt = threaded(1);
    let metrics = rt.metrics();

    let start = Instant::now();
    rt.block_on(async {
        tokio::spawn(spin_yielding(BUSY, Duration::from_micros(50)))
            .await
            .unwrap();
    });
    let elapsed = start.elapsed();

    let stall = metrics.worker_total_stall_duration(0);
    assert!(stall < elapsed / 5, "stall {stall:?} of {elapsed:?}");
}

#[test]
fn contending_tasks_stall_for_the_whole_busy_period() {
    // Two tasks share one worker. Whichever is not running is always queued,
    // so the worker is stalled for the whole busy period regardless of how
    // finely the tasks yield. This is the case schedule-latency sampling
    // undercounts.
    for slice in [Duration::from_micros(25), Duration::from_millis(5)] {
        let rt = threaded(1);
        let metrics = rt.metrics();

        let before = metrics.worker_total_stall_duration(0);
        let start = Instant::now();
        rt.block_on(async {
            let a = tokio::spawn(spin_yielding(BUSY, slice));
            let b = tokio::spawn(spin_yielding(BUSY, slice));
            a.await.unwrap();
            b.await.unwrap();
        });
        let elapsed = start.elapsed();
        let stall = metrics.worker_total_stall_duration(0) - before;

        assert!(
            stall > elapsed * 7 / 10,
            "slice {slice:?}: stall {stall:?} of {elapsed:?}"
        );
        assert!(stall <= elapsed, "slice {slice:?}: stall {stall:?} of {elapsed:?}");
    }
}

#[test]
fn global_queue_stall_while_worker_is_blocked() {
    // Tasks spawned from outside the runtime go to the global queue and stay
    // there while the only worker is busy in a long, non-yielding poll.
    let rt = threaded(1);
    let metrics = rt.metrics();

    let (started_tx, started_rx) = mpsc::channel();
    let blocker = rt.spawn(async move {
        started_tx.send(()).unwrap();
        spin(BUSY);
    });
    started_rx.recv().unwrap();

    let queued: Vec<_> = (0..8).map(|_| rt.spawn(async {})).collect();
    rt.block_on(async {
        blocker.await.unwrap();
        for task in queued {
            task.await.unwrap();
        }
    });

    let stall = metrics.global_queue_total_stall_duration();
    assert!(stall > BUSY / 2, "global queue stall {stall:?}");
}

#[test]
fn totals_are_monotonic() {
    let rt = threaded(2);
    let metrics = rt.metrics();

    rt.block_on(async {
        let tasks: Vec<_> = (0..4)
            .map(|_| tokio::spawn(spin_yielding(Duration::from_millis(50), Duration::from_micros(20))))
            .collect();

        let mut prev = [Duration::ZERO; 2];
        let deadline = Instant::now() + Duration::from_millis(50);
        while Instant::now() < deadline {
            for (worker, prev) in prev.iter_mut().enumerate() {
                let total = metrics.worker_total_stall_duration(worker);
                assert!(total >= *prev, "worker {worker}: {total:?} < {prev:?}");
                *prev = total;
            }
            std::thread::yield_now();
        }

        for task in tasks {
            task.await.unwrap();
        }
    });
}
