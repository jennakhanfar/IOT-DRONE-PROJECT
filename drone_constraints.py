"""
drone_constraints.py
---------------------
Simulates drone-level hardware constraints for model inference ONLY.

Architecture:
  - PC side  : loads images, runs data pipeline, displays results  (full CPU/RAM)
  - Drone side: runs face recognition model inference               (400MHz / 128MB)

Usage:
    from drone_constraints import DroneInferenceContext

    with DroneInferenceContext() as ctx:
        embeddings = model.get_embeddings(images)

    print(ctx.peak_ram_mb)   # peak RAM used during inference
    print(ctx.latency_ms)    # wall-clock time under constraint
"""

import os
import time
import threading
import platform
import ctypes
import ctypes.wintypes

# ── Drone hardware spec constants ─────────────────────────────────────────────
DRONE_RAM_MB    = 128   # RAM budget for model inference on drone
DRONE_CPU_MHZ   = 400   # target drone CPU speed in MHz
THROTTLE_MS     = 50    # throttle window size in ms

_IS_WINDOWS = platform.system() == "Windows"


def _detect_cpu_mhz():
    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq and freq.max > 0:
            return freq.max
    except Exception:
        pass
    return 2600.0


_HOST_CPU_MHZ = _detect_cpu_mhz()
_CPU_FRACTION = min(max(DRONE_CPU_MHZ / _HOST_CPU_MHZ, 0.01), 1.0)


def set_drone_cpu_mhz(mhz):
    """
    Override the drone CPU target at runtime (for CPU ablation experiments).
    Must be called BEFORE any DroneInferenceContext is entered.
    """
    global DRONE_CPU_MHZ, _CPU_FRACTION, _throttle_thread
    DRONE_CPU_MHZ = int(mhz)
    _CPU_FRACTION = min(max(DRONE_CPU_MHZ / _HOST_CPU_MHZ, 0.01), 1.0)
    # Reset throttle thread so it picks up the new fraction on next use
    _throttle_thread = None


# ── CPU affinity helpers ───────────────────────────────────────────────────────

def _pin_to_core(core=0):
    try:
        if _IS_WINDOWS:
            ctypes.windll.kernel32.SetProcessAffinityMask(
                ctypes.windll.kernel32.GetCurrentProcess(), 1 << core)
        else:
            os.sched_setaffinity(0, {core})
        return True
    except Exception:
        return False


def _restore_all_cores():
    try:
        import psutil
        mask = (1 << psutil.cpu_count(logical=True)) - 1
        if _IS_WINDOWS:
            ctypes.windll.kernel32.SetProcessAffinityMask(
                ctypes.windll.kernel32.GetCurrentProcess(), mask)
        else:
            os.sched_setaffinity(0, set(range(psutil.cpu_count(logical=True))))
    except Exception:
        pass


# ── CPU throttle thread ────────────────────────────────────────────────────────

class _ThrottleThread(threading.Thread):
    """
    Daemon thread that throttles CPU to drone speed by sleeping.
    Only throttles while active_event is set.
    """

    def __init__(self, fraction, interval_ms):
        super(_ThrottleThread, self).__init__(daemon=True)
        self.active_event = threading.Event()
        self._fraction = fraction
        self._interval_s = interval_ms / 1000.0

    def run(self):
        active_s = self._interval_s * self._fraction
        sleep_s  = self._interval_s * (1.0 - self._fraction)
        while True:
            if self.active_event.is_set():
                time.sleep(active_s)
                time.sleep(sleep_s)
            else:
                time.sleep(0.005)  # idle check


# Single shared throttle thread (started once, paused when not in context)
_throttle_thread = None
_throttle_lock   = threading.Lock()


def _get_throttle_thread():
    global _throttle_thread
    with _throttle_lock:
        if _throttle_thread is None:
            _throttle_thread = _ThrottleThread(_CPU_FRACTION, THROTTLE_MS)
            _throttle_thread.start()
    return _throttle_thread


# ── RAM monitor ────────────────────────────────────────────────────────────────

class _RamMonitor(threading.Thread):
    """
    Monitors RSS during inference. Kills the process if it exceeds
    baseline_mb + DRONE_RAM_MB (hard enforcement).
    """

    def __init__(self, baseline_mb, limit_mb):
        super(_RamMonitor, self).__init__(daemon=True)
        self.baseline_mb = baseline_mb
        self.limit_mb    = limit_mb
        self.peak_mb     = baseline_mb
        self.exceeded    = False
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        import psutil
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                if rss_mb > self.peak_mb:
                    self.peak_mb = rss_mb
                delta_mb = rss_mb - self.baseline_mb
                if delta_mb > self.limit_mb:
                    self.exceeded = True
                    print(
                        "\n[DRONE SIM] *** OUT OF MEMORY ***"
                        "\n[DRONE SIM] Inference used %.1f MB over baseline "
                        "(drone limit: %d MB). Terminating." % (delta_mb, self.limit_mb)
                    )
                    os.kill(os.getpid(), 9)
            except Exception:
                pass
            time.sleep(0.02)  # check every 20ms


# ── Context manager ────────────────────────────────────────────────────────────

class DroneInferenceContext(object):
    """
    Context manager that applies drone constraints only during inference.

    PC side (outside context) : full CPU, full RAM — data loading, GUI, etc.
    Drone side (inside context): 400MHz single core, 128MB RAM — model inference.

    Example
    -------
        ctx = DroneInferenceContext()
        with ctx:
            emb = model.get_embeddings(images)
        print("%.1f ms | %.1f MB peak" % (ctx.latency_ms, ctx.delta_ram_mb))
    """

    def __init__(self, ram_mb=DRONE_RAM_MB):
        self.ram_mb       = ram_mb
        self.latency_ms   = 0.0
        self.peak_ram_mb  = 0.0
        self.delta_ram_mb = 0.0
        self._t0          = None
        self._ram_monitor = None
        self._baseline_mb = 0.0

    def __enter__(self):
        import psutil
        proc = psutil.Process(os.getpid())
        self._baseline_mb = proc.memory_info().rss / (1024 * 1024)

        # Pin to single core + enable CPU throttle
        _pin_to_core(0)
        _get_throttle_thread().active_event.set()

        # Start RAM monitor (hard-kills if inference exceeds 128MB delta)
        self._ram_monitor = _RamMonitor(self._baseline_mb, self.ram_mb)
        self._ram_monitor.start()

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._t0
        self.latency_ms = elapsed * 1000.0

        # Disable throttle + restore all cores
        _get_throttle_thread().active_event.clear()
        _restore_all_cores()

        # Stop RAM monitor
        if self._ram_monitor:
            self._ram_monitor.stop()
            self._ram_monitor.join(timeout=0.1)
            self.peak_ram_mb  = self._ram_monitor.peak_mb
            self.delta_ram_mb = self.peak_ram_mb - self._baseline_mb

        return False  # do not suppress exceptions


# ── Summary printer ────────────────────────────────────────────────────────────

def print_constraint_summary():
    print("\n" + "=" * 55)
    print("  DRONE INFERENCE CONSTRAINTS")
    print("=" * 55)
    print("  Host CPU   : %.0f MHz (%s)" % (_HOST_CPU_MHZ, platform.processor()[:40]))
    print("  Drone CPU  : %d MHz -> throttle = %.1f%% of 1 core"
          % (DRONE_CPU_MHZ, _CPU_FRACTION * 100))
    print("  Drone RAM  : %d MB hard limit (inference delta only)" % DRONE_RAM_MB)
    print("  Scope      : model inference only (not data loading)")
    print("=" * 55 + "\n")


# ── Backward-compat shim (old apply_drone_constraints callers) ────────────────

def apply_drone_constraints(**kwargs):
    print_constraint_summary()
