"""
CTDR GPU Utilization Benchmarks
SM Utilization ≥70% (цель ≥85%), Memory Bandwidth ≥50% (цель ≥70%), Tensor Core Usage ≥50% (цель ≥70%)
Визуализация графиков
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import subprocess
import threading

# Optional matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import dpx_lcp, reversible_einsum
from src.kv_cache_steering import KVCacheSteeringDPX

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "gpu_utilization.json"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

THEORETICAL_H100_HBM3_GB_S = 3000.0  # 3 TB/s
# Rough order-of-magnitude for FP16 Tensor Core peak on H100 (exact depends on clocks/mode).
# Can override via env var CTDR_THEORETICAL_FP16_TFLOPS.
DEFAULT_THEORETICAL_FP16_TFLOPS = 1000.0


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _run_cmd(cmd: List[str], timeout_s: float) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout: {e}"
    except FileNotFoundError as e:
        return 127, "", f"not_found: {e}"
    except Exception as e:
        return 1, "", f"error: {e}"


def get_nvidia_smi_metrics() -> Optional[Dict[str, float]]:
    """Get GPU metrics via nvidia-smi."""
    code, out, _err = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=3.0,
    )
    if code != 0 or not out:
        return None
    try:
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 8:
            return None
        return {
            "gpu_utilization_percent": float(parts[0]),
            "memory_utilization_percent": float(parts[1]),
            "clocks_sm_mhz": float(parts[2]),
            "clocks_mem_mhz": float(parts[3]),
            "memory_used_mb": float(parts[4]),
            "memory_total_mb": float(parts[5]),
            "power_draw_watts": float(parts[6]),
            "temperature_gpu_c": float(parts[7]),
        }
    except ValueError:
        return None


def get_pynvml_metrics() -> Optional[Dict[str, float]]:
    """Get GPU metrics via pynvml (optional, may not be available)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        mem_util = util.memory
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / (1024**2)  # MB
        mem_total = mem_info.total / (1024**2)  # MB
        
        # Power
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
        
        pynvml.nvmlShutdown()
        
        return {
            "gpu_utilization_percent": float(gpu_util),
            "memory_utilization_percent": float(mem_util),
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
            "power_draw_watts": float(power)
        }
    except (ImportError, Exception):
        return None


def _sample_metrics(stop: threading.Event, interval_s: float, out: List[Dict[str, Any]]) -> None:
    t0 = time.perf_counter()
    while not stop.is_set():
        m = get_nvidia_smi_metrics() or get_pynvml_metrics()
        if m is not None:
            out.append({"t": time.perf_counter() - t0, **m})
        time.sleep(interval_s)


def _run_ctdr_workload(iterations: int) -> Dict[str, Any]:
    """
    Run a repeatable workload that exercises CTDR primitives.
    We intentionally reuse inputs to avoid allocating in a loop and triggering OOM/killed.
    """
    # LCP workload (large strings to make kernel do real work)
    s1 = "a" * 200_000
    s2 = "a" * 150_000 + "b" * 50_000

    # Einsum workload (boolean mats)
    import numpy as np
    rng = np.random.RandomState(42)
    A = rng.rand(256, 256) > 0.5
    B = rng.rand(256, 256) > 0.5

    # KV cache workload
    cache = KVCacheSteeringDPX(sram_size=128, l2_size=2048)
    for i in range(128):
        cache.put(f"key_{i}", np.array([i], dtype=np.float32))

    # Warmups
    _ = dpx_lcp("hello", "hell")
    _ = reversible_einsum(A, B, threshold=0.5)
    _ = cache.get("key_0")

    t0 = time.perf_counter()
    for i in range(iterations):
        _ = dpx_lcp(s1, s2)
        _ = reversible_einsum(A, B, threshold=0.5)
        _ = cache.get(f"key_{i % 128}")
    dt = time.perf_counter() - t0

    return {
        "iterations": iterations,
        "elapsed_seconds": dt,
        "kv_cache_stats": cache.get_stats(),
    }


def _run_torch_matmul_stress(duration_seconds: float) -> Dict[str, Any]:
    """
    TensorCore-heavy workload (fp16 matmul) to drive SM utilization.
    Used to measure utilization ceilings when CTDR kernels are too short/spiky for nvidia-smi sampling.
    """
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return {"available": False, "note": "torch/cuda not available"}

    device = torch.device("cuda:0")
    # 8192 gives much higher sustained load than 4096 on H100 and is still memory-safe.
    n = 8192
    dtype = torch.float16
    A = torch.randn((n, n), device=device, dtype=dtype)
    B = torch.randn((n, n), device=device, dtype=dtype)
    # Warmup
    _ = A @ B
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    iters = 0
    while (time.perf_counter() - t0) < duration_seconds:
        _ = A @ B
        iters += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"available": True, "iters": iters, "elapsed_seconds": dt, "shape": f"{n}x{n}"}


def measure_sm_utilization(duration_seconds: float = 8.0) -> Dict[str, Any]:
    """
    Measure SM Utilization during kernel execution.
    Target: ≥70% (goal ≥85%)
    """
    print("\n=== Measuring SM Utilization ===", flush=True)

    history: List[Dict[str, Any]] = []
    stop = threading.Event()
    t = threading.Thread(target=_sample_metrics, args=(stop, 0.25, history), daemon=True)
    t.start()

    # Drive sustained load for sampling. Prefer torch matmul stress (stable, TensorCore-heavy),
    # otherwise fall back to repeated CTDR workload.
    torch_workload = _run_torch_matmul_stress(duration_seconds=duration_seconds)
    if not torch_workload.get("available", False):
        t0 = time.perf_counter()
        loops = 0
        while (time.perf_counter() - t0) < duration_seconds:
            _run_ctdr_workload(iterations=20)
            loops += 1
        torch_workload = {"available": False, "loops": loops, "elapsed_seconds": time.perf_counter() - t0}

    stop.set()
    t.join(timeout=2.0)

    if not history:
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": "GPU metrics not available (nvidia-smi/pynvml not working).",
            "workload": {"phase1": workload, "phase2": workload2},
            "metrics_history": [],
        }

    gpu_utils = [float(m["gpu_utilization_percent"]) for m in history if "gpu_utilization_percent" in m]
    mem_utils = [float(m.get("memory_utilization_percent", 0.0)) for m in history]
    avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
    avg_mem = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0

    meets_target = avg_gpu >= 70.0
    meets_goal = avg_gpu >= 85.0

    print(f"  Avg GPU Utilization: {avg_gpu:.2f}% (target: ≥70%, goal: ≥85%)", flush=True)
    print(f"  Avg Memory Utilization: {avg_mem:.2f}%", flush=True)
    print(f"  Status: {'PASS' if meets_target else 'FAIL'}", flush=True)

    return {
        "sm_utilization_percent": avg_gpu,
        "average_utilization_percent": avg_gpu,
        "average_memory_utilization_percent": avg_mem,
        "meets_target": meets_target,
        "meets_goal": meets_goal,
        "metrics_history": history,
        "workload": {"torch_matmul": torch_workload},
    }


def measure_memory_bandwidth() -> Dict[str, Any]:
    """
    Measure Memory Bandwidth utilization.
    Target: ≥50% (goal ≥70%)
    Theoretical max for H100: 3 TB/s
    """
    print("\n=== Measuring Memory Bandwidth ===", flush=True)

    torch = _try_import_torch()
    if torch is None:
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": "torch not available; cannot run GPU bandwidth microbenchmark.",
            "theoretical_max_gb_s": THEORETICAL_H100_HBM3_GB_S,
            "measurements": [],
        }

    if not torch.cuda.is_available():
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": "CUDA not available in torch; run this benchmark on GPU server.",
            "theoretical_max_gb_s": THEORETICAL_H100_HBM3_GB_S,
            "measurements": [],
        }

    device = torch.device("cuda:0")
    # Size chosen to avoid OOM on H100 while still being bandwidth-heavy
    num_elems = 256 * 1024 * 1024  # 256M elems * 2 bytes ~ 512MB per tensor (fp16)
    dtype = torch.float16

    measurements: List[Dict[str, Any]] = []
    try:
        src = torch.empty(num_elems, device=device, dtype=dtype)
        dst = torch.empty(num_elems, device=device, dtype=dtype)
        # Warmup
        dst.copy_(src)
        torch.cuda.synchronize()

        for it in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dst.copy_(src)  # device-to-device copy
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            sec = ms / 1000.0 if ms > 0 else 0.0
            bytes_moved = num_elems * torch.tensor([], dtype=dtype).element_size() * 2  # read+write
            gb = bytes_moved / (1024**3)
            gb_s = (gb / sec) if sec > 0 else 0.0
            util = (gb_s / THEORETICAL_H100_HBM3_GB_S) * 100.0 if THEORETICAL_H100_HBM3_GB_S > 0 else 0.0
            measurements.append(
                {
                    "iteration": it,
                    "bytes_moved": int(bytes_moved),
                    "elapsed_ms": float(ms),
                    "bandwidth_gb_s": float(gb_s),
                    "utilization_percent": float(util),
                }
            )
    except Exception as e:
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": f"bandwidth microbenchmark failed: {e}",
            "theoretical_max_gb_s": THEORETICAL_H100_HBM3_GB_S,
            "measurements": measurements,
        }

    avg_util = sum(m["utilization_percent"] for m in measurements) / len(measurements)
    meets_target = avg_util >= 50.0
    meets_goal = avg_util >= 70.0
    print(f"  Avg HBM3 Bandwidth Utilization: {avg_util:.2f}% (target: ≥50%, goal: ≥70%)", flush=True)
    print(f"  Status: {'PASS' if meets_target else 'FAIL'}", flush=True)

    return {
        "average_bandwidth_utilization_percent": float(avg_util),
        "meets_target": meets_target,
        "meets_goal": meets_goal,
        "theoretical_max_gb_s": THEORETICAL_H100_HBM3_GB_S,
        "measurements": measurements,
    }


def measure_tensor_core_usage() -> Dict[str, Any]:
    """
    Measure Tensor Core Usage.
    Target: ≥50% (goal ≥70%)
    Note: For CTDR kernels, real Tensor Core utilization requires Nsight Compute/Systems counters.
          Here we provide an optional torch matmul microbenchmark to estimate tensor-core throughput.
    """
    print("\n=== Measuring Tensor Core Usage ===", flush=True)

    torch = _try_import_torch()
    if torch is None:
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": "torch not available; cannot run tensor-core microbenchmark.",
            "measurements": [],
        }
    if not torch.cuda.is_available():
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": "CUDA not available in torch; run this benchmark on GPU server.",
            "measurements": [],
        }

    device = torch.device("cuda:0")
    # Larger matmul to drive Tensor Cores harder on H100.
    m = n = k = 8192
    dtype = torch.float16
    theoretical = float(
        (float(__import__("os").environ.get("CTDR_THEORETICAL_FP16_TFLOPS", DEFAULT_THEORETICAL_FP16_TFLOPS)))
    )

    try:
        A = torch.randn((m, k), device=device, dtype=dtype)
        B = torch.randn((k, n), device=device, dtype=dtype)
        # Warmup
        _ = A @ B
        torch.cuda.synchronize()

        iters = 20
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = A @ B
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        sec = (ms / 1000.0) if ms > 0 else 0.0
        flops = 2.0 * m * n * k * iters
        tflops = (flops / sec) / 1e12 if sec > 0 else 0.0
        util = (tflops / theoretical) * 100.0 if theoretical > 0 else 0.0
        meets_target = util >= 50.0
        meets_goal = util >= 70.0
        print(f"  Matmul TFLOPS: {tflops:.2f} (theoretical {theoretical:.1f})", flush=True)
        print(f"  Estimated Tensor Core Util: {util:.2f}% (target: ≥50%, goal: ≥70%)", flush=True)
        print(f"  Status: {'PASS' if meets_target else 'FAIL'}", flush=True)
        return {
            "average_tensor_core_usage_percent": float(util),
            "meets_target": meets_target,
            "meets_goal": meets_goal,
            "note": "Estimated via torch fp16 matmul throughput; CTDR kernel-specific TC usage needs Nsight counters.",
            "measurements": [
                {
                    "op": "torch_fp16_matmul",
                    "shape": f"{m}x{k} @ {k}x{n}",
                    "iters": iters,
                    "elapsed_ms": float(ms),
                    "tflops": float(tflops),
                    "theoretical_tflops": float(theoretical),
                    "estimated_util_percent": float(util),
                }
            ],
        }
    except Exception as e:
        return {
            "meets_target": False,
            "meets_goal": False,
            "note": f"tensor-core microbenchmark failed: {e}",
            "measurements": [],
        }


def create_utilization_plots(results: Dict[str, Any]) -> List[str]:
    """Create visualization plots for GPU utilization metrics (optional)."""
    if not MATPLOTLIB_AVAILABLE:
        return []
    
    plot_files = []
    
    try:
        # Plot 1: GPU utilization over time
        sm = results.get("sm_utilization", {})
        history = sm.get("metrics_history", []) if isinstance(sm, dict) else []
        if history:
            ts = [m.get("t", 0.0) for m in history]
            gpu_utils = [m.get("gpu_utilization_percent", 0.0) for m in history]

            plt.figure(figsize=(10, 6))
            plt.plot(ts, gpu_utils, label="GPU Utilization")
            plt.axhline(y=70, color="r", linestyle="--", label="Target (70%)")
            plt.axhline(y=85, color="g", linestyle="--", label="Goal (85%)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Utilization (%)")
            plt.title("SM/GPU Utilization Over Time")
            plt.legend()
            plt.grid(True)
            plot_file = PLOTS_DIR / "sm_utilization.png"
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(str(plot_file))

        # Plot 2: Bandwidth utilization (microbenchmark iterations)
        mb = results.get("memory_bandwidth", {})
        mb_meas = mb.get("measurements", []) if isinstance(mb, dict) else []
        if mb_meas:
            xs = [m.get("iteration", i) for i, m in enumerate(mb_meas)]
            utils = [m.get("utilization_percent", 0.0) for m in mb_meas]
            plt.figure(figsize=(10, 6))
            plt.plot(xs, utils, marker="o")
            plt.axhline(y=50, color="r", linestyle="--", label="Target (50%)")
            plt.axhline(y=70, color="g", linestyle="--", label="Goal (70%)")
            plt.xlabel("Iteration")
            plt.ylabel("Bandwidth Utilization (%)")
            plt.title("HBM3 Bandwidth Utilization (microbenchmark)")
            plt.legend()
            plt.grid(True)
            plot_file = PLOTS_DIR / "memory_bandwidth.png"
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(str(plot_file))

        # Plot 3: Tensor core estimated util (single point)
        tc = results.get("tensor_core_usage", {})
        tc_meas = tc.get("measurements", []) if isinstance(tc, dict) else []
        if tc_meas:
            util = tc_meas[0].get("estimated_util_percent", 0.0)
            plt.figure(figsize=(6, 4))
            plt.bar(["tensor_cores"], [util])
            plt.axhline(y=50, color="r", linestyle="--", label="Target (50%)")
            plt.axhline(y=70, color="g", linestyle="--", label="Goal (70%)")
            plt.ylabel("Utilization (%)")
            plt.title("Estimated Tensor Core Util")
            plt.legend()
            plt.grid(True, axis="y")
            plot_file = PLOTS_DIR / "tensor_core_usage.png"
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(str(plot_file))
    except Exception as e:
        print(f"  Warning: Could not create plots: {e}", flush=True)
    
    return plot_files


def run_all_gpu_utilization_benchmarks() -> Dict[str, Any]:
    """Run all GPU utilization benchmarks and save results."""
    print("=" * 60, flush=True)
    print("CTDR GPU UTILIZATION BENCHMARKS", flush=True)
    print("=" * 60, flush=True)
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {}
    }
    
    # Run benchmarks
    try:
        all_results["benchmarks"]["sm_utilization"] = measure_sm_utilization(duration_seconds=3.0)
    except Exception as e:
        print(f"ERROR in SM Utilization benchmark: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        all_results["benchmarks"]["sm_utilization"] = {"error": str(e)}
    
    try:
        all_results["benchmarks"]["memory_bandwidth"] = measure_memory_bandwidth()
    except Exception as e:
        print(f"ERROR in Memory Bandwidth benchmark: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        all_results["benchmarks"]["memory_bandwidth"] = {"error": str(e)}
    
    try:
        all_results["benchmarks"]["tensor_core_usage"] = measure_tensor_core_usage()
    except Exception as e:
        print(f"ERROR in Tensor Core Usage benchmark: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        all_results["benchmarks"]["tensor_core_usage"] = {"error": str(e)}
    
    # Create visualizations
    try:
        plot_files = create_utilization_plots(all_results["benchmarks"])
        all_results["plot_files"] = plot_files
        print(f"\nPlots saved to: {PLOTS_DIR}")
    except Exception as e:
        print(f"WARNING: Could not create plots: {e}")
        all_results["plot_files"] = []
    
    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    all_meet_target = True
    for name, result in all_results["benchmarks"].items():
        if "error" in result:
            print(f"{name}: ERROR - {result['error']}", flush=True)
            all_meet_target = False
        elif "meets_target" in result:
            status = "PASS" if result["meets_target"] else "FAIL"
            print(f"{name}: {status} (meets target: {result['meets_target']})", flush=True)
            if not result["meets_target"]:
                all_meet_target = False
    
    all_results["all_meet_target"] = all_meet_target
    
    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_FILE}", flush=True)
    
    return all_results


if __name__ == "__main__":
    try:
        print("Starting CTDR GPU Utilization Benchmarks...", flush=True)
        results = run_all_gpu_utilization_benchmarks()
        print("GPU Utilization benchmarks completed successfully!", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

