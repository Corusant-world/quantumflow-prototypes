"""
CTDR Landauer Gap Benchmark (REAL GPU ENERGY)

Goal:
- Measure REAL GPU energy (J) for two workloads while sampling nvidia-smi:
  1) Baseline Tensor-Core matmul (torch fp16/bf16)
  2) DPX LCP retrieval (ctdr_python full-array query)
- Compute an explicit Landauer lower bound using a CLEAR proxy:
  - "bits overwritten" ~= bits written into the *output buffer* per iteration
    (baseline: output matrix C; DPX: LCP output array for all candidates).
- Report the "gap" in orders of magnitude: log10(measured_energy / landauer_bound)

No simulations: power/temp/util are read from nvidia-smi during execution.
Landauer bound is necessarily a calculation, but it is tied to a concrete,
explicitly stated overwrite proxy and the *measured* average temperature.

Output:
- benchmarks/results/landauer_gap.json
"""

from __future__ import annotations

import json
import math
import os
import random
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoding import encode_to_short2

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "landauer_gap.json"


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


def _get_nvidia_smi_metrics(gpu_id: int) -> Optional[Dict[str, float]]:
    """
    Returns REAL instantaneous metrics via nvidia-smi.
    """
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
            "power_draw_watts": float(parts[0]),
            "temperature_gpu_c": float(parts[1]),
            "gpu_utilization_percent": float(parts[2]),
            "memory_utilization_percent": float(parts[3]),
            "memory_used_mb": float(parts[4]),
            "memory_total_mb": float(parts[5]),
        }
    except ValueError:
        return None


def _sample_nvidia_smi(stop: threading.Event, interval_s: float, out: List[Dict[str, Any]], gpu_id: int) -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        m = _get_nvidia_smi_metrics(gpu_id=gpu_id)
        if m is not None:
            out.append({"t": time.perf_counter() - t0, **m})
        time.sleep(interval_s)


def _integrate_energy_joules(samples: List[Dict[str, Any]], duration_s: float) -> Optional[float]:
    """
    Integrate energy from sampled power (Watts) over time (seconds) => Joules.
    Uses piecewise trapezoid integration + constant extrapolation to window edges.
    """
    if duration_s <= 0:
        return None
    if not samples:
        return None

    xs = sorted(samples, key=lambda s: float(s.get("t", 0.0)))
    if "power_draw_watts" not in xs[0]:
        return None

    energy = 0.0

    # Left edge extrapolation (0 -> first sample)
    t_first = float(xs[0].get("t", 0.0))
    p_first = float(xs[0]["power_draw_watts"])
    if t_first > 0.0:
        energy += p_first * min(t_first, duration_s)

    # Trapezoids between samples (clamped to duration)
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
        pa = float(a["power_draw_watts"])
        pb = float(b["power_draw_watts"])
        dt_used = seg_b - seg_a
        energy += 0.5 * (pa + pb) * dt_used

    # Right edge extrapolation (last sample -> duration)
    t_last = float(xs[-1].get("t", 0.0))
    if "power_draw_watts" in xs[-1] and t_last < duration_s:
        p_last = float(xs[-1]["power_draw_watts"])
        energy += p_last * (duration_s - t_last)

    return float(energy)


def _summary_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"available": False}

    def _avg(key: str) -> Optional[float]:
        vals = [float(s[key]) for s in samples if key in s]
        return float(sum(vals) / len(vals)) if vals else None

    def _max(key: str) -> Optional[float]:
        vals = [float(s[key]) for s in samples if key in s]
        return float(max(vals)) if vals else None

    def _min(key: str) -> Optional[float]:
        vals = [float(s[key]) for s in samples if key in s]
        return float(min(vals)) if vals else None

    return {
        "available": True,
        "num_samples": len(samples),
        "average_power_watts": _avg("power_draw_watts"),
        "peak_power_watts": _max("power_draw_watts"),
        "average_temp_c": _avg("temperature_gpu_c"),
        "peak_temp_c": _max("temperature_gpu_c"),
        "min_temp_c": _min("temperature_gpu_c"),
        "average_gpu_util_percent": _avg("gpu_utilization_percent"),
        "peak_gpu_util_percent": _max("gpu_utilization_percent"),
        "average_mem_util_percent": _avg("memory_utilization_percent"),
        "peak_mem_util_percent": _max("memory_utilization_percent"),
        "average_mem_used_mb": _avg("memory_used_mb"),
        "peak_mem_used_mb": _max("memory_used_mb"),
        "memory_total_mb": _avg("memory_total_mb"),
    }


def _landauer_e_min_per_bit_joule(temp_k: float) -> float:
    # k_B exact SI: 1.380649e-23 J/K
    k_b = 1.380649e-23
    return float(k_b * float(temp_k) * math.log(2.0))


def _safe_log10(x: Optional[float]) -> Optional[float]:
    if x is None or x <= 0:
        return None
    return float(math.log10(x))


def _measure_window(
    name: str,
    fn: Callable[[], Dict[str, Any]],
    gpu_id: int,
    sample_interval_s: float,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    stop = threading.Event()
    t = threading.Thread(target=_sample_nvidia_smi, args=(stop, sample_interval_s, samples, gpu_id), daemon=True)
    t.start()
    workload = fn()
    stop.set()
    t.join(timeout=3.0)

    duration_s = float(workload.get("elapsed_seconds", 0.0))
    energy_j = _integrate_energy_joules(samples, duration_s=duration_s) if duration_s > 0 else None
    stats = _summary_stats(samples)

    avg_temp_c = stats.get("average_temp_c")
    temp_k = (float(avg_temp_c) + 273.15) if isinstance(avg_temp_c, (int, float)) else 300.0
    e_min_bit = _landauer_e_min_per_bit_joule(temp_k)

    return {
        "name": name,
        "workload": workload,
        "nvidia_smi_samples": samples,
        "nvidia_smi_summary": stats,
        "energy_joules": energy_j,
        "energy_per_op_joules": (float(energy_j) / float(workload["logical_ops"]))
        if (energy_j is not None and workload.get("logical_ops"))
        else None,
        "landauer": {
            "temperature_k_used": float(temp_k),
            "e_min_per_bit_joule": float(e_min_bit),
        },
    }


def _prepare_baseline_tc_matmul(n: int, dtype_name: str) -> Dict[str, Any]:
    """
    Allocate + warmup OUTSIDE the energy window (so the window is only steady-state matmuls).
    """
    import torch  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in torch (baseline TC matmul requires GPU)")

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {sorted(dtype_map.keys())}")
    dtype = dtype_map[dtype_name.lower()]

    device = torch.device("cuda:0")
    torch.manual_seed(123)

    A = torch.randn((n, n), device=device, dtype=dtype)
    B = torch.randn((n, n), device=device, dtype=dtype)
    C = torch.empty((n, n), device=device, dtype=dtype)

    for _ in range(3):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()

    return {"torch": torch, "A": A, "B": B, "C": C, "n": n, "dtype": dtype}


def _run_baseline_tc_matmul_prepared(state: Dict[str, Any], max_seconds: float) -> Dict[str, Any]:
    torch = state["torch"]
    A = state["A"]
    B = state["B"]
    C = state["C"]
    n = int(state["n"])
    dtype = state["dtype"]

    t0 = time.perf_counter()
    it = 0
    while (time.perf_counter() - t0) < max_seconds:
        torch.mm(A, B, out=C)
        # Critical: make the loop time reflect REAL GPU work time.
        # Without this, the CPU can enqueue kernels faster than the GPU executes them,
        # and the final synchronize inflates elapsed_seconds far beyond max_seconds.
        torch.cuda.synchronize()
        it += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    bytes_written = int(it) * int(C.numel()) * int(C.element_size())
    bits_written = int(bytes_written) * 8

    return {
        "workload": "baseline_tc_matmul",
        "iterations": int(it),
        "elapsed_seconds": float(dt),
        "logical_ops": int(it),  # matmuls
        "matmul_shape": f"{n}x{n} @ {n}x{n}",
        "dtype": str(dtype).replace("torch.", ""),
        "output_bytes_written_proxy": int(bytes_written),
        "output_bits_written_proxy": int(bits_written),
        "note": "Overwrite proxy counts ONLY output buffer C writes (one overwrite per element per iteration).",
    }


def _build_dpx_candidates_bytes(query: str, query_len: int, num_candidates: int, seed: int) -> bytes:
    """
    Build candidates bytes WITHOUT storing all Python strings (memory-safe).
    Candidates are length=query_len to match the DPX kernel expectations used elsewhere.
    """
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits

    def rand_str(n: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(n))

    out = bytearray()
    for i in range(num_candidates):
        common = i % (query_len + 1)
        cand = query[:common] + rand_str(query_len - common)
        out.extend(encode_to_short2(cand))
    return bytes(out)


def _prepare_dpx_index(num_candidates: int, query_len: int, seed: int) -> Dict[str, Any]:
    """
    Load DPX candidates index + warmup OUTSIDE the energy window.
    """
    try:
        import ctdr_python  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ctdr_python not available (DPX required): {e}")

    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits
    query = "".join(rng.choice(alphabet) for _ in range(query_len))

    cand_bytes = _build_dpx_candidates_bytes(query=query, query_len=query_len, num_candidates=num_candidates, seed=seed + 1)
    ok = bool(ctdr_python.dpx_lcp_index_load(cand_bytes, num_candidates))
    if not ok:
        raise RuntimeError("dpx_lcp_index_load returned false")

    q_bytes = encode_to_short2(query)
    _ = ctdr_python.dpx_lcp_index_query(q_bytes)  # warmup

    return {"ctdr": ctdr_python, "query": query, "query_bytes": q_bytes, "num_candidates": num_candidates, "query_len": query_len}


def _run_dpx_lcp_retrieval_full_prepared(state: Dict[str, Any], max_seconds: float) -> Dict[str, Any]:
    """
    DPX retrieval workload that returns the FULL LCP array (one value per candidate).
    This makes the overwrite proxy explicit: num_candidates * value_bytes per iteration.
    """
    ctdr_python = state["ctdr"]
    q_bytes = state["query_bytes"]
    num_candidates = int(state["num_candidates"])
    query_len = int(state["query_len"])

    t0 = time.perf_counter()
    it = 0
    last_len = None
    while (time.perf_counter() - t0) < max_seconds:
        lcps = ctdr_python.dpx_lcp_index_query(q_bytes)
        try:
            last_len = int(len(lcps))
        except Exception:
            last_len = None
        it += 1
    dt = time.perf_counter() - t0

    n_out = int(last_len) if last_len is not None else int(num_candidates)

    # Proxy: output buffer holds one LCP value per candidate.
    # Underlying type may be uint16 (2 bytes) or int32 (4 bytes); we report BOTH.
    bytes_u16 = int(it) * n_out * 2
    bytes_i32 = int(it) * n_out * 4

    return {
        "workload": "dpx_lcp_retrieval_full_array",
        "iterations": int(it),
        "elapsed_seconds": float(dt),
        "logical_ops": int(it),  # queries
        "num_candidates": int(num_candidates),
        "query_len": int(query_len),
        "output_values_per_query": int(n_out),
        "output_bytes_written_proxy_u16": int(bytes_u16),
        "output_bits_written_proxy_u16": int(bytes_u16 * 8),
        "output_bytes_written_proxy_i32": int(bytes_i32),
        "output_bits_written_proxy_i32": int(bytes_i32 * 8),
        "note": "Overwrite proxy assumes DPX query materializes an output array of LCP values (1 per candidate) on GPU each iteration.",
    }


def _compute_landauer_metrics_for_window(window: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute measured-vs-Landauer comparisons for a window using its overwrite proxy.
    For DPX we provide two bounds (u16 and i32 output element sizes).
    """
    energy_j = window.get("energy_joules")
    e_min_bit = window.get("landauer", {}).get("e_min_per_bit_joule")
    if energy_j is None or e_min_bit is None:
        return {"available": False}

    wl = window.get("workload", {})
    logical_ops = wl.get("logical_ops")

    def _pack(bits: Optional[int]) -> Dict[str, Any]:
        if bits is None or bits <= 0:
            return {"available": False}
        landauer_j = float(bits) * float(e_min_bit)
        ratio = (float(energy_j) / landauer_j) if landauer_j > 0 else None
        return {
            "available": True,
            "bits_overwritten_proxy": int(bits),
            "landauer_lower_bound_joules": float(landauer_j),
            "measured_energy_joules": float(energy_j),
            "gap_ratio_measured_over_landauer": float(ratio) if ratio is not None else None,
            "gap_orders_of_magnitude_log10": _safe_log10(ratio),
            "measured_energy_per_op_joules": (float(energy_j) / float(logical_ops)) if logical_ops else None,
            "landauer_per_op_joules": (float(landauer_j) / float(logical_ops)) if logical_ops else None,
        }

    if "output_bits_written_proxy" in wl:
        return {"proxy": _pack(int(wl["output_bits_written_proxy"]))}

    bits_u16 = wl.get("output_bits_written_proxy_u16")
    bits_i32 = wl.get("output_bits_written_proxy_i32")
    return {
        "proxy_u16": _pack(int(bits_u16)) if bits_u16 is not None else {"available": False},
        "proxy_i32": _pack(int(bits_i32)) if bits_i32 is not None else {"available": False},
    }


def run_landauer_gap_benchmark() -> Dict[str, Any]:
    print("=" * 70, flush=True)
    print("CTDR LANDAUER GAP BENCHMARK (REAL GPU ENERGY)", flush=True)
    print("ALL POWER/TEMP/UTIL NUMBERS FROM nvidia-smi. NO SIMULATIONS.", flush=True)
    print("=" * 70, flush=True)

    gpu_id = int(os.environ.get("CTDR_GPU_ID", "0"))
    sample_interval_s = float(os.environ.get("CTDR_LANDAUER_SAMPLE_INTERVAL_S", "0.25"))

    baseline_seconds = float(os.environ.get("CTDR_LANDAUER_BASELINE_SECONDS", "6"))
    ctdr_seconds = float(os.environ.get("CTDR_LANDAUER_CTDR_SECONDS", "6"))

    matmul_n = int(os.environ.get("CTDR_LANDAUER_MATMUL_N", "16384"))
    matmul_dtype = str(os.environ.get("CTDR_LANDAUER_MATMUL_DTYPE", "float16"))

    num_candidates = int(os.environ.get("CTDR_LANDAUER_NUM_CANDIDATES", "8192"))
    query_len = int(os.environ.get("CTDR_LANDAUER_QUERY_LEN", "2048"))

    initial = _get_nvidia_smi_metrics(gpu_id=gpu_id) or {}
    if initial:
        print(
            f"Initial GPU: {initial.get('power_draw_watts', 0):.0f}W, "
            f"{initial.get('temperature_gpu_c', 0):.0f}°C, "
            f"util={initial.get('gpu_utilization_percent', 0):.0f}%, "
            f"mem={initial.get('memory_used_mb', 0):.0f}/{initial.get('memory_total_mb', 0):.0f}MB",
            flush=True,
        )
    else:
        print("Initial GPU: nvidia-smi metrics unavailable (will still try to run).", flush=True)

    # --- PREPARE OUTSIDE MEASUREMENT WINDOWS (critical for honest measurement) ---
    print("\n[SETUP] Preparing baseline TC matmul tensors (outside window)...", flush=True)
    baseline_state = _prepare_baseline_tc_matmul(n=matmul_n, dtype_name=matmul_dtype)
    print("[SETUP] Preparing DPX index (outside window)...", flush=True)
    dpx_state = _prepare_dpx_index(num_candidates=num_candidates, query_len=query_len, seed=42)

    # --- MEASURE WINDOWS ---
    baseline_win = _measure_window(
        "baseline_tc_matmul",
        fn=lambda: _run_baseline_tc_matmul_prepared(state=baseline_state, max_seconds=baseline_seconds),
        gpu_id=gpu_id,
        sample_interval_s=sample_interval_s,
    )
    ctdr_win = _measure_window(
        "dpx_lcp_retrieval_full_array",
        fn=lambda: _run_dpx_lcp_retrieval_full_prepared(state=dpx_state, max_seconds=ctdr_seconds),
        gpu_id=gpu_id,
        sample_interval_s=sample_interval_s,
    )

    # Best-effort cleanup (outside window)
    try:
        dpx_state["ctdr"].dpx_lcp_index_clear()
    except Exception:
        pass
    try:
        baseline_state["torch"].cuda.empty_cache()
    except Exception:
        pass

    baseline_landauer = _compute_landauer_metrics_for_window(baseline_win)
    ctdr_landauer = _compute_landauer_metrics_for_window(ctdr_win)

    result: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "gpu_id": gpu_id,
            "sample_interval_s": sample_interval_s,
            "baseline_seconds": baseline_seconds,
            "ctdr_seconds": ctdr_seconds,
            "matmul_n": matmul_n,
            "matmul_dtype": matmul_dtype,
            "dpx_num_candidates": num_candidates,
            "dpx_query_len": query_len,
        },
        "windows": {"baseline": baseline_win, "ctdr": ctdr_win},
        "landauer_gap": {"baseline": baseline_landauer, "ctdr": ctdr_landauer},
        "notes": [
            "Setup (allocations + DPX index load + warmups) is performed OUTSIDE measurement windows.",
            "Energy is integrated from nvidia-smi power.draw samples over the measured workload duration.",
            "Landauer bound uses measured average GPU temperature for the window (fallback 300K if unavailable).",
            "Overwrite proxy is explicit and conservative: counts ONLY output-buffer overwrites per iteration.",
            "For DPX full-array query we report two proxy variants (u16/i32) because the output element size may differ.",
        ],
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)

    # Tight summary
    b = baseline_landauer.get("proxy", {})
    c_u16 = ctdr_landauer.get("proxy_u16", {})
    c_i32 = ctdr_landauer.get("proxy_i32", {})

    def _fmt_gap(x: Dict[str, Any]) -> str:
        if not x or not x.get("available"):
            return "N/A"
        return f"log10 gap={x.get('gap_orders_of_magnitude_log10'):.2f} (ratio={x.get('gap_ratio_measured_over_landauer'):.2e})"

    print("\n=== SUMMARY (Measured vs Landauer) ===", flush=True)
    print(f"Baseline (TC matmul): { _fmt_gap(b) }", flush=True)
    print(f"CTDR (DPX query, u16): { _fmt_gap(c_u16) }", flush=True)
    print(f"CTDR (DPX query, i32): { _fmt_gap(c_i32) }", flush=True)

    return result


if __name__ == "__main__":
    run_landauer_gap_benchmark()


