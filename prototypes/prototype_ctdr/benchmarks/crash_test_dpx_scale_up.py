"""
CRASH TEST (PHASE 3): DPX SCALE-UP + MEMOIZATION @ LARGE N

What this does (REAL measurements):
- Loads a DPX LCP index with N candidates (short2/uint16 encoding, fixed length).
- Runs DPX top-1 queries for a fixed time window while sampling nvidia-smi:
  power.draw, temperature.gpu, utilization.gpu, utilization.memory, memory.used/total
- Computes:
  - index load time (s)
  - query throughput (qps), avg/p95 latency (ms)
  - energy (J) via power integration over the query window
  - GPU memory used (MB)
  - memoization speedup (cold DPX query vs RLAStack cache get)
- Also prints the physical baseline memory requirement for an N×N float16 attention matrix
  as a *limit check* (bytes and equivalent # of H100 80GB for the matrix alone).

No "pretty theory" is used as evidence here: the DPX path is real, timed, and measured.
The only computation is the unavoidable physical memory requirement N^2×2 bytes for float16.

Output:
- benchmarks/results/dpx_scale_up.json
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.encoding import encode_to_short2
from src.rla_stack import RLAStack

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "dpx_scale_up.json"


def _run_cmd(cmd: List[str], timeout_s: float) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout: {e}"
    except FileNotFoundError as e:
        return 127, "", f"not_found: {e}"
    except Exception as e:
        return 1, "", f"error: {e}"


def get_nvidia_smi_line(gpu_id: int) -> Optional[Dict[str, float]]:
    code, out, _err = _run_cmd(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=3.0,
    )
    if code != 0 or not out:
        return None
    try:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 6:
            return None
        return {
            "power_w": float(parts[0]),
            "temp_c": float(parts[1]),
            "gpu_util_pct": float(parts[2]),
            "mem_util_pct": float(parts[3]),
            "mem_used_mb": float(parts[4]),
            "mem_total_mb": float(parts[5]),
        }
    except ValueError:
        return None


def _sample_smi(stop: threading.Event, interval_s: float, out: List[Dict[str, Any]], gpu_id: int) -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        m = get_nvidia_smi_line(gpu_id=gpu_id)
        if m is not None:
            out.append({"t": time.perf_counter() - t0, **m})
        time.sleep(interval_s)


def _integrate_energy_j(samples: List[Dict[str, Any]], duration_s: float) -> Optional[float]:
    if duration_s <= 0 or not samples:
        return None
    xs = sorted(samples, key=lambda s: float(s.get("t", 0.0)))
    if "power_w" not in xs[0]:
        return None

    energy = 0.0
    # Left edge extrapolation
    t_first = float(xs[0].get("t", 0.0))
    p_first = float(xs[0]["power_w"])
    if t_first > 0.0:
        energy += p_first * min(t_first, duration_s)

    # Trapezoids
    for a, b in zip(xs, xs[1:]):
        ta = float(a.get("t", 0.0))
        tb = float(b.get("t", 0.0))
        if tb <= ta:
            continue
        if ta >= duration_s:
            break
        seg_a = max(0.0, ta)
        seg_b = min(duration_s, tb)
        if seg_b <= seg_a:
            continue
        pa = float(a["power_w"])
        pb = float(b["power_w"])
        dt = seg_b - seg_a
        energy += 0.5 * (pa + pb) * dt

    # Right edge extrapolation
    t_last = float(xs[-1].get("t", 0.0))
    if t_last < duration_s and "power_w" in xs[-1]:
        p_last = float(xs[-1]["power_w"])
        energy += p_last * (duration_s - t_last)

    return float(energy)


def _p95_ms(lat_ms: List[float]) -> Optional[float]:
    if not lat_ms:
        return None
    return float(np.percentile(np.array(lat_ms, dtype=np.float64), 95))


def _bytes_human(b: int) -> str:
    if b >= 1024**4:
        return f"{b / (1024**4):.2f} TB"
    if b >= 1024**3:
        return f"{b / (1024**3):.2f} GB"
    if b >= 1024**2:
        return f"{b / (1024**2):.2f} MB"
    return f"{b} B"


def _estimate_attention_matrix_bytes(n: int) -> int:
    # float16 attention matrix only: N×N×2 bytes
    return int(n) * int(n) * 2


def _estimate_h100_count(bytes_needed: int, h100_bytes: int = int(80e9)) -> int:
    return int(math.ceil(float(bytes_needed) / float(h100_bytes)))


def _build_candidates_uint16(n: int, query_len: int) -> np.ndarray:
    """
    Build an (N, query_len) uint16 matrix:
    - Base filled with 'a'
    - One mismatch per row at position (row % query_len) set to 'b'
    This creates a spread of LCP lengths from 0..query_len-1 and forces the kernel to work.
    """
    base = np.full((n, query_len), ord("a"), dtype=np.uint16)
    rows = np.arange(n, dtype=np.int64)
    cols = rows % int(query_len)
    base[rows, cols] = ord("b")
    return base


def _try_buffer(obj: Any) -> Any:
    """
    Prefer passing a zero-copy buffer to ctdr_python if supported.
    Fallback: bytes copy (RAM heavy).
    """
    try:
        return memoryview(obj)
    except TypeError:
        return obj


def run() -> Dict[str, Any]:
    print("=" * 70, flush=True)
    print("CRASH TEST (PHASE 3): DPX SCALE-UP + MEMOIZATION", flush=True)
    print("REAL nvidia-smi sampling. REAL DPX queries.", flush=True)
    print("=" * 70, flush=True)

    gpu_id = int(os.environ.get("CTDR_GPU_ID", "0"))
    sample_interval_s = float(os.environ.get("CTDR_SCALE_SAMPLE_INTERVAL_S", "0.25"))
    query_seconds = float(os.environ.get("CTDR_SCALE_QUERY_SECONDS", "6"))
    query_len = int(os.environ.get("CTDR_SCALE_QUERY_LEN", "256"))
    n_list_raw = str(os.environ.get("CTDR_SCALE_N_LIST", "500000,1000000,2000000"))

    try:
        import ctdr_python  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ctdr_python not available (DPX required): {e}")

    # GPU snapshot (pre)
    pre = get_nvidia_smi_line(gpu_id=gpu_id) or {}
    print(
        f"GPU pre: {pre.get('power_w', 0):.0f}W, {pre.get('temp_c', 0):.0f}°C, "
        f"util={pre.get('gpu_util_pct', 0):.0f}%, mem={pre.get('mem_used_mb', 0):.0f}/{pre.get('mem_total_mb', 0):.0f}MB",
        flush=True,
    )

    # Stable query used for all tests
    query = "a" * query_len
    query_bytes = encode_to_short2(query)

    n_values = [int(x.strip()) for x in n_list_raw.split(",") if x.strip()]
    if not n_values:
        raise ValueError("CTDR_SCALE_N_LIST is empty")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "gpu_id": gpu_id,
            "sample_interval_s": sample_interval_s,
            "query_seconds": query_seconds,
            "query_len": query_len,
            "n_list": n_values,
        },
        "runs": [],
    }

    for n in n_values:
        print("\n" + "-" * 70, flush=True)
        print(f"[N={n:,}] Building candidates (uint16) ...", flush=True)

        t_build0 = time.perf_counter()
        candidates_u16 = _build_candidates_uint16(n=n, query_len=query_len)
        build_s = time.perf_counter() - t_build0
        bytes_candidates = int(candidates_u16.nbytes)
        print(f"  Built in {build_s:.2f}s, bytes={_bytes_human(bytes_candidates)}", flush=True)

        # Load index (outside query window)
        print("  Loading DPX index to GPU ...", flush=True)
        t_load0 = time.perf_counter()
        try:
            buf = _try_buffer(candidates_u16)
            ok = bool(ctdr_python.dpx_lcp_index_load(buf, int(n)))
            load_input_kind = "buffer"
        except TypeError:
            # Fallback if extension doesn't accept buffer protocol
            load_input_kind = "tobytes"
            ok = bool(ctdr_python.dpx_lcp_index_load(candidates_u16.tobytes(order="C"), int(n)))
        load_s = time.perf_counter() - t_load0
        if not ok:
            raise RuntimeError("dpx_lcp_index_load returned false")

        after_load = get_nvidia_smi_line(gpu_id=gpu_id) or {}
        print(
            f"  Load time: {load_s:.2f}s | GPU mem used: {after_load.get('mem_used_mb', 0):.0f}MB",
            flush=True,
        )

        # Prepare query/top1
        ok2 = bool(ctdr_python.dpx_lcp_index_set_query(query_bytes))
        if not ok2:
            raise RuntimeError("dpx_lcp_index_set_query returned false")
        _ = ctdr_python.dpx_lcp_index_query_top1()  # warmup

        # Measure cold query once (for memoization speedup)
        t_cold0 = time.perf_counter()
        best_idx, best_lcp = ctdr_python.dpx_lcp_index_query_top1()
        cold_ms = (time.perf_counter() - t_cold0) * 1000.0

        # Memoization (RLAStack) — same API as our other benchmarks
        rla = RLAStack()
        key = f"dpx_top1:N={n}:L={query_len}"
        rla.memoize(key, np.array([int(best_idx), int(best_lcp)], dtype=np.int32))
        hot_iters = int(os.environ.get("CTDR_SCALE_HOT_ITERS", "200000"))
        t_hot0 = time.perf_counter()
        for _ in range(hot_iters):
            _ = rla.get(key)
        hot_ms = (time.perf_counter() - t_hot0) * 1000.0 / float(hot_iters)
        memo_speedup = (cold_ms / hot_ms) if hot_ms > 0 else None

        # Query window with nvidia-smi sampling
        print(f"  Query window: {query_seconds:.1f}s (sampling {sample_interval_s:.2f}s) ...", flush=True)
        samples: List[Dict[str, Any]] = []
        lat_ms: List[float] = []
        stop = threading.Event()
        th = threading.Thread(target=_sample_smi, args=(stop, sample_interval_s, samples, gpu_id), daemon=True)
        th.start()

        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < query_seconds:
            s = time.perf_counter()
            _ = ctdr_python.dpx_lcp_index_query_top1()
            lat_ms.append((time.perf_counter() - s) * 1000.0)
        dt_s = time.perf_counter() - t0

        stop.set()
        th.join(timeout=3.0)

        energy_j = _integrate_energy_j(samples=samples, duration_s=dt_s)
        qps = (len(lat_ms) / dt_s) if dt_s > 0 else None
        avg_ms = (float(sum(lat_ms)) / len(lat_ms)) if lat_ms else None
        p95 = _p95_ms(lat_ms)

        # Baseline physical limit (matrix only)
        std_bytes = _estimate_attention_matrix_bytes(n)
        std_h100 = _estimate_h100_count(std_bytes)

        run = {
            "n": n,
            "build_candidates_s": float(build_s),
            "candidates_bytes": int(bytes_candidates),
            "load_index_s": float(load_s),
            "load_input_kind": load_input_kind,
            "gpu_after_load": after_load,
            "query_window_seconds": float(dt_s),
            "queries": int(len(lat_ms)),
            "qps": float(qps) if qps is not None else None,
            "latency_avg_ms": float(avg_ms) if avg_ms is not None else None,
            "latency_p95_ms": float(p95) if p95 is not None else None,
            "energy_joules": float(energy_j) if energy_j is not None else None,
            "energy_per_query_joules": (float(energy_j) / float(len(lat_ms))) if (energy_j and len(lat_ms) > 0) else None,
            "nvidia_smi_samples": samples,
            "cold_query_ms": float(cold_ms),
            "memoized_get_ms": float(hot_ms),
            "memoization_speedup": float(memo_speedup) if memo_speedup is not None else None,
            "baseline_attention_matrix": {
                "bytes_float16": int(std_bytes),
                "human": _bytes_human(int(std_bytes)),
                "equivalent_h100_count_matrix_only": int(std_h100),
                "note": "Matrix-only lower bound; real attention needs more (Q/K/V/KV + overhead + comms).",
            },
        }

        # Summarize GPU window metrics
        if samples:
            avg_util = float(sum(float(s["gpu_util_pct"]) for s in samples) / len(samples))
            avg_power = float(sum(float(s["power_w"]) for s in samples) / len(samples))
            avg_temp = float(sum(float(s["temp_c"]) for s in samples) / len(samples))
            run["gpu_window"] = {
                "avg_gpu_util_pct": avg_util,
                "avg_power_w": avg_power,
                "avg_temp_c": avg_temp,
                "meets_sm_util_min70": avg_util >= 70.0,
            }
            print(
                f"  QPS={run['qps']:.1f} | avg={run['latency_avg_ms']:.3f}ms p95={run['latency_p95_ms']:.3f}ms | "
                f"GPU avg util={avg_util:.1f}% avg power={avg_power:.0f}W | energy={run['energy_joules']:.1f}J",
                flush=True,
            )
        else:
            print(f"  QPS={run['qps']:.1f} | avg={run['latency_avg_ms']:.3f}ms p95={run['latency_p95_ms']:.3f}ms", flush=True)
            print("  WARNING: no nvidia-smi samples collected for query window.", flush=True)

        print(
            f"  Memoization: cold={cold_ms:.3f}ms, hot={hot_ms:.6f}ms, speedup={run['memoization_speedup']:.0f}×",
            flush=True,
        )
        print(
            f"  Baseline matrix-only @N={n:,}: {run['baseline_attention_matrix']['human']} (~{std_h100}× H100)",
            flush=True,
        )

        results["runs"].append(run)

        # Cleanup GPU index + local big arrays before next N
        try:
            ctdr_python.dpx_lcp_index_clear()
        except Exception:
            pass
        del candidates_u16

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Saved: {RESULTS_FILE}", flush=True)
    print("=" * 70, flush=True)

    post = get_nvidia_smi_line(gpu_id=gpu_id) or {}
    print(
        f"GPU post: {post.get('power_w', 0):.0f}W, {post.get('temp_c', 0):.0f}°C, "
        f"util={post.get('gpu_util_pct', 0):.0f}%, mem={post.get('mem_used_mb', 0):.0f}/{post.get('mem_total_mb', 0):.0f}MB",
        flush=True,
    )

    return results


if __name__ == "__main__":
    run()


