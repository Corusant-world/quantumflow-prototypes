"""
Team 2 Energy — Benchmarks

Contract:
- Must not crash without CUDA (CPU fallback writes results).
- Must write to `benchmarks/results/latest.json` (relative to this file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
import platform

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from core import EnergyGridSystem, HAS_GPU, TNP_System


def _env_int(name: str, default: int, min_v: int, max_v: int) -> int:
    try:
        v = int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        v = default
    return max(min_v, min(max_v, v))


def _start_nvml_sampler(interval_s: float = 0.5) -> Tuple[Optional[threading.Event], Optional[threading.Thread], List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        samples.append({"ok": False, "error": f"nvml_unavailable: {e}"})
        return None, None, samples

    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                samples.append(
                    {
                        "ok": True,
                        "ts": time.time(),
                        "gpu_util": int(getattr(u, "gpu", 0)),
                        "mem_util": int(getattr(u, "memory", 0)),
                        "mem_used_mb": int(int(mem.used) // (1024 * 1024)),
                    }
                )
            except Exception as e:
                samples.append({"ok": False, "error": str(e)})
                break
            time.sleep(interval_s)

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop, t, samples


def _summarize_nvml(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [s.get("gpu_util") for s in samples if s.get("ok") and isinstance(s.get("gpu_util"), int)]
    mems = [s.get("mem_used_mb") for s in samples if s.get("ok") and isinstance(s.get("mem_used_mb"), int)]
    return {
        "samples": len(samples),
        "gpu_util_avg": float(sum(vals) / len(vals)) if vals else None,
        "gpu_util_max": int(max(vals)) if vals else None,
        "mem_used_mb_max": int(max(mems)) if mems else None,
        "nvml_ok": bool(vals),
    }


def _gpu_matmul_soak(seconds: int, n: int = 16384, dtype: torch.dtype = torch.bfloat16) -> Dict[str, Any]:
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(2):
        _ = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 0
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    while (time.perf_counter() - t0) < float(seconds):
        _ = a @ b
        iters += 1
    end_evt.record()
    torch.cuda.synchronize()

    elapsed_ms = float(start_evt.elapsed_time(end_evt))
    flops = 2.0 * float(n**3) * float(iters)
    tflops = (flops / max(1e-9, (elapsed_ms / 1000.0))) / 1e12
    return {
        "duration_s_target": int(seconds),
        "matrix_n": int(n),
        "dtype": str(dtype).replace("torch.", ""),
        "iters": int(iters),
        "elapsed_ms": elapsed_ms,
        "tflops_est": float(tflops),
    }


def _reserve_vram(target_gb: int, headroom_gb: int = 2) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if target_gb <= 0 or not torch.cuda.is_available():
        return None, {"enabled": False}
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = int(total_b - free_b)
        target_b = int(target_gb) * (1024**3)
        headroom_b = int(headroom_gb) * (1024**3)
        can_alloc_b = max(0, int(free_b) - headroom_b)
        need_b = max(0, int(target_b) - int(used_b))
        alloc_b = min(need_b, can_alloc_b)
        if alloc_b <= 0:
            return None, {
                "enabled": True,
                "allocated_bytes": 0,
                "note": "Already at/above target or insufficient free memory",
                "used_bytes_before": used_b,
                "free_bytes_before": int(free_b),
                "total_bytes": int(total_b),
            }
        t = torch.empty(int(alloc_b), device="cuda", dtype=torch.uint8)
        torch.cuda.synchronize()
        free_a, total_a = torch.cuda.mem_get_info()
        return t, {
            "enabled": True,
            "target_gb": int(target_gb),
            "headroom_gb": int(headroom_gb),
            "allocated_bytes": int(alloc_b),
            "used_bytes_before": used_b,
            "free_bytes_before": int(free_b),
            "free_bytes_after": int(free_a),
            "total_bytes": int(total_a),
        }
    except Exception as e:
        return None, {"enabled": True, "error": str(e)}


def _results_path() -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "latest.json")


def _try_nvidia_smi() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return {"ok": True, "raw": out}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _probe_deps() -> Dict[str, Any]:
    deps: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            deps["cuda_device"] = torch.cuda.get_device_name(0)
            deps["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception as e:
            deps["cuda_device_error"] = str(e)

    # Optional NVML
    try:
        import pynvml  # type: ignore
        deps["pynvml"] = {"available": True, "version": getattr(pynvml, "__version__", None)}
    except Exception as e:
        deps["pynvml"] = {"available": False, "error": str(e)}

    # Optional cuQuantum/cuPy
    try:
        import cupy  # type: ignore
        deps["cupy"] = {"available": True, "version": getattr(cupy, "__version__", None)}
    except Exception as e:
        deps["cupy"] = {"available": False, "error": str(e)}

    try:
        import cuquantum  # type: ignore
        deps["cuquantum"] = {"available": True, "version": getattr(cuquantum, "__version__", None)}
    except Exception as e:
        deps["cuquantum"] = {"available": False, "error": str(e)}

    return deps


def run_benchmark() -> None:
    print("🚀 STARTING BENCHMARK — Team 2 Energy (EnergyGridSystem + TNP_System)")

    out_path = _results_path()
    bench_seconds = _env_int("BENCH_SECONDS", 60, 4, 180)
    # Allocate budget slices upfront so they exist even if a benchmark sub-step crashes.
    per_s = max(4, int(bench_seconds * 0.4))
    reserve_gb = _env_int("BENCH_RESERVE_GB", 0, 0, 78)
    payload: Dict[str, Any] = {
        "status": "success" if HAS_GPU else "cpu_fallback",
        "device": "cuda" if HAS_GPU else "cpu",
        "deps": _probe_deps(),
        "nvidia_smi_before": _try_nvidia_smi(),
        "benchmarks": {},
        "bench_seconds": bench_seconds,
        "bench_reserve_gb": reserve_gb,
    }

    stop_nvml, nvml_thread, nvml_samples = (None, None, [])
    if HAS_GPU:
        reserve_tensor, reserve_info = _reserve_vram(reserve_gb, headroom_gb=2)
        payload["benchmarks"]["vram_reserve"] = reserve_info
        stop_nvml, nvml_thread, nvml_samples = _start_nvml_sampler(interval_s=0.5)

    # Benchmark 1: EnergyGridSystem optimization loop
    try:
        sys_ = EnergyGridSystem(num_nodes=512)
        sys_.step_simulation()
        m1 = sys_.get_metrics()

        t0 = time.perf_counter()
        sys_.optimize_topology(steps=120)
        sys_.step_simulation()
        wall = time.perf_counter() - t0
        m2 = sys_.get_metrics()
        ok, msg = sys_.validation_check()

        payload["benchmarks"]["energy_grid"] = {
            "ok": bool(ok),
            "msg": msg,
            "wall_time_s": round(wall, 6),
            "before": m1,
            "after": m2,
            "delta_loss_mw": float(m2["total_loss_mw"] - m1["total_loss_mw"]),
        }
        print(f"[energy_grid] wall={wall:.3f}s loss={m1['total_loss_mw']:.4f}->{m2['total_loss_mw']:.4f} ok={ok}")
    except Exception as e:
        payload["benchmarks"]["energy_grid"] = {"error": str(e)}

    # Benchmark 2: TNP_System thermo workload (GPU-heavy when CUDA available)
    if HAS_GPU:
        try:
            tnp = TNP_System(dim=4096, layers=2)
            # warmup
            _ = tnp.simulate_thermo(batch_size=64, steps=3)

            # Sustain for a portion of the time budget, leaving room for the matmul soak.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 0
            last_stats: Dict[str, Any] = {}
            while (time.perf_counter() - t0) < float(per_s):
                last_stats = tnp.simulate_thermo(batch_size=128, steps=12)  # type: ignore[assignment]
                iters += 1
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            payload["benchmarks"]["tnp_thermo"] = {
                "wall_time_s": round(wall, 6),
                "iters": int(iters),
                "stats": last_stats,
            }
            print(f"[tnp_thermo] wall={wall:.3f}s iters={iters} total_joules={last_stats.get('total_joules')}")
        except Exception as e:
            payload["benchmarks"]["tnp_thermo"] = {"error": str(e)}
    else:
        payload["benchmarks"]["tnp_thermo"] = {"skipped": True, "reason": "no_cuda"}

    if HAS_GPU:
        # Hard proof: Tensor Core friendly matmul soak
        # Keep total runtime ~bench_seconds
        soak_s = max(4, int(bench_seconds - per_s))
        payload["benchmarks"]["gpu_soak_matmul"] = _gpu_matmul_soak(seconds=soak_s, n=16384, dtype=torch.bfloat16)

        if stop_nvml is not None:
            stop_nvml.set()
        if nvml_thread is not None:
            nvml_thread.join(timeout=2.0)
        payload["nvml"] = _summarize_nvml(nvml_samples)

    payload["nvidia_smi_after"] = _try_nvidia_smi()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ BENCHMARK COMPLETE. Wrote {out_path}")

    _ = reserve_tensor if HAS_GPU else None


if __name__ == "__main__":
    run_benchmark()
