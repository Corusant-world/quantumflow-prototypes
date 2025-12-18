"""
CTDR Energy Efficiency Benchmarks

Metrics:
- Average/peak power draw (W) during workload
- Total energy (J) during workload
- Energy per logical operation (J/op)

Note:
- This is NOT LLM token energy (we don't generate tokens here).
- We report energy per repeatable CTDR workload operation so the run is reproducible.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoding import encode_to_short2

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "energy_efficiency.json"


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


def get_power_draw_watts() -> Optional[float]:
    code, out, _err = _run_cmd(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        timeout_s=2.0,
    )
    if code != 0 or not out:
        return None
    try:
        return float(out.splitlines()[0].strip())
    except ValueError:
        return None


def _sample_power(stop: threading.Event, interval_s: float, out: List[Dict[str, Any]]) -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        p = get_power_draw_watts()
        if p is not None:
            out.append({"t": time.perf_counter() - t0, "power_draw_watts": p})
        time.sleep(interval_s)


def _run_baseline_dot_retrieval(max_seconds: float, dim: int, num_candidates: int) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for baseline dot-product retrieval")

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Candidates matrix: [num_candidates, dim]
    candidates = torch.randn((num_candidates, dim), device=device, dtype=torch.float16)
    query = torch.randn((dim,), device=device, dtype=torch.float16)

    # Warmup (TC)
    for _ in range(5):
        _ = torch.matmul(candidates, query)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    it = 0
    while True:
        scores = torch.matmul(candidates, query)  # [num_candidates]
        _ = int(torch.argmax(scores).item())
        it += 1
        if (time.perf_counter() - t0) >= max_seconds:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return {
        "iterations": it,
        "elapsed_seconds": dt,
        "workload": "baseline_dot_product_retrieval",
        "dim": dim,
        "num_candidates": num_candidates,
        "logical_ops": it,  # retrieval queries
    }


def _run_ctdr_lcp_retrieval(max_seconds: float, query_len: int, num_candidates: int) -> Dict[str, Any]:
    import random
    import string

    try:
        import ctdr_python  # type: ignore
    except Exception as e:
        raise RuntimeError(f"ctdr_python not available: {e}")

    random.seed(42)
    alphabet = string.ascii_lowercase + string.digits

    def rand_str(n: int) -> str:
        return "".join(random.choice(alphabet) for _ in range(n))

    query = rand_str(query_len)
    candidates = []
    for i in range(num_candidates):
        common = i % (query_len + 1)
        candidates.append(query[:common] + rand_str(query_len - common))

    cand_bytes = b"".join(encode_to_short2(s) for s in candidates)
    ok = bool(ctdr_python.dpx_lcp_index_load(cand_bytes, num_candidates))
    if not ok:
        raise RuntimeError("dpx_lcp_index_load returned false")

    q_bytes = encode_to_short2(query)
    ok2 = bool(ctdr_python.dpx_lcp_index_set_query(q_bytes))
    if not ok2:
        raise RuntimeError("dpx_lcp_index_set_query returned false")
    _ = ctdr_python.dpx_lcp_index_query_top1()  # warmup

    t0 = time.perf_counter()
    it = 0
    while True:
        _idx, _val = ctdr_python.dpx_lcp_index_query_top1()
        it += 1
        if (time.perf_counter() - t0) >= max_seconds:
            break
    dt = time.perf_counter() - t0

    return {
        "iterations": it,
        "elapsed_seconds": dt,
        "workload": "ctdr_lcp_retrieval",
        "query_len": query_len,
        "num_candidates": num_candidates,
        "logical_ops": it,  # retrieval queries
    }


def _measure_window(name: str, fn: Callable[[], Dict[str, Any]], sample_interval_s: float = 0.25) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    stop = threading.Event()
    t = threading.Thread(target=_sample_power, args=(stop, sample_interval_s, samples), daemon=True)
    t.start()
    workload = fn()
    stop.set()
    t.join(timeout=2.0)

    if samples:
        powers = [float(s["power_draw_watts"]) for s in samples]
        avg_power = sum(powers) / len(powers)
        peak_power = max(powers)
        energy_j = float(avg_power) * float(workload["elapsed_seconds"])
        energy_per_op = energy_j / float(workload["logical_ops"]) if workload.get("logical_ops") else None
        return {
            "name": name,
            "workload": workload,
            "power_samples": samples,
            "average_power_watts": float(avg_power),
            "peak_power_watts": float(peak_power),
            "energy_joules": float(energy_j),
            "energy_per_op_joules": float(energy_per_op) if energy_per_op is not None else None,
        }

    return {
        "name": name,
        "workload": workload,
        "power_samples": [],
        "average_power_watts": None,
        "peak_power_watts": None,
        "energy_joules": None,
        "energy_per_op_joules": None,
        "note": "No power samples collected for this window (nvidia-smi unavailable or window too short).",
    }


def run_energy_efficiency_benchmark(duration_seconds: float = 10.0) -> Dict[str, Any]:
    """
    Run workload while sampling power draw and compute energy metrics.
    """
    print("=" * 60, flush=True)
    print("CTDR ENERGY EFFICIENCY BENCHMARK", flush=True)
    print("=" * 60, flush=True)

    # A/B windows with separate power sampling -> we can compute Joules/query per paradigm.
    dim = int(os.environ.get("CTDR_BASELINE_DIM", "4096"))
    num_candidates = int(os.environ.get("CTDR_NUM_CANDIDATES", "8192"))
    query_len = int(os.environ.get("CTDR_QUERY_LEN", "2048"))
    baseline_seconds = float(os.environ.get("CTDR_BASELINE_SECONDS", "8"))
    ctdr_seconds = float(os.environ.get("CTDR_CTDR_SECONDS", "8"))

    baseline_win = _measure_window(
        "baseline_dot_product_retrieval",
        lambda: _run_baseline_dot_retrieval(max_seconds=baseline_seconds, dim=dim, num_candidates=num_candidates),
    )
    ctdr_win = _measure_window(
        "ctdr_lcp_retrieval",
        lambda: _run_ctdr_lcp_retrieval(max_seconds=ctdr_seconds, query_len=query_len, num_candidates=num_candidates),
    )

    if (baseline_win.get("average_power_watts") is None) and (ctdr_win.get("average_power_watts") is None):
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_power_available": False,
            "note": "No power samples collected. Run on GPU server with nvidia-smi.",
            "windows": {"baseline": baseline_win, "ctdr": ctdr_win},
        }
    else:
        baseline_e = baseline_win.get("energy_per_op_joules")
        ctdr_e = ctdr_win.get("energy_per_op_joules")
        ratio = (float(baseline_e) / float(ctdr_e)) if (baseline_e and ctdr_e and ctdr_e > 0) else None
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_power_available": True,
            "config": {
                "baseline_seconds": baseline_seconds,
                "ctdr_seconds": ctdr_seconds,
                "baseline_dim": dim,
                "query_len": query_len,
                "num_candidates": num_candidates,
            },
            "windows": {"baseline": baseline_win, "ctdr": ctdr_win},
            "delta": {
                "energy_per_query_ratio_baseline_over_ctdr": ratio,
            },
            "formulas": {
                "energy_joules_window": "E = avg_power_watts * elapsed_seconds",
                "energy_per_query_window": "E_q = energy_joules / queries",
                "ratio": "ratio = E_q_baseline / E_q_ctdr",
            },
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {RESULTS_FILE}", flush=True)
    return result


if __name__ == "__main__":
    seconds = float(os.environ.get("CTDR_ENERGY_DURATION_SECONDS", "10"))
    run_energy_efficiency_benchmark(duration_seconds=seconds)


