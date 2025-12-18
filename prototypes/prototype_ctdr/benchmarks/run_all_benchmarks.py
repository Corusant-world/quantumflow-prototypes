"""
CTDR Comprehensive Benchmark Runner

Runs all benchmark scripts and consolidates results into:
- benchmarks/results/comprehensive_report.json
- benchmarks/results/latest.json (updated)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
COMPREHENSIVE_FILE = RESULTS_DIR / "comprehensive_report.json"
LATEST_FILE = RESULTS_DIR / "latest.json"


def _safe_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as e:
        return e


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"error": f"missing_file: {path}"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"read_error: {e}", "path": str(path)}


def run_all() -> Dict[str, Any]:
    print("=" * 60, flush=True)
    print("CTDR RUN ALL BENCHMARKS", flush=True)
    print("=" * 60, flush=True)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {},
        "files": {},
    }

    # 1) Performance
    perf_mod = _safe_import("benchmarks.benchmark_performance")
    if isinstance(perf_mod, Exception):
        report["results"]["performance"] = {"error": str(perf_mod)}
    else:
        try:
            perf_mod.run_all_performance_benchmarks()
            report["files"]["performance"] = str(RESULTS_DIR / "performance.json")
            report["results"]["performance"] = _read_json(RESULTS_DIR / "performance.json")
        except Exception as e:
            report["results"]["performance"] = {"error": str(e)}

    # 2) GPU utilization
    gpu_mod = _safe_import("benchmarks.benchmark_gpu_utilization")
    if isinstance(gpu_mod, Exception):
        report["results"]["gpu_utilization"] = {"error": str(gpu_mod)}
    else:
        try:
            gpu_mod.run_all_gpu_utilization_benchmarks()
            report["files"]["gpu_utilization"] = str(RESULTS_DIR / "gpu_utilization.json")
            report["results"]["gpu_utilization"] = _read_json(RESULTS_DIR / "gpu_utilization.json")
        except Exception as e:
            report["results"]["gpu_utilization"] = {"error": str(e)}

    # 3) Energy efficiency (new schema: windows/ratio)
    energy_mod = _safe_import("benchmarks.benchmark_energy_efficiency")
    if isinstance(energy_mod, Exception):
        report["results"]["energy_efficiency"] = {"error": str(energy_mod)}
    else:
        try:
            import os
            duration = float(os.environ.get("CTDR_ENERGY_DURATION_SECONDS", "10"))
            energy_mod.run_energy_efficiency_benchmark(duration_seconds=duration)
            report["files"]["energy_efficiency"] = str(RESULTS_DIR / "energy_efficiency.json")
            energy_data = _read_json(RESULTS_DIR / "energy_efficiency.json")
            report["results"]["energy_efficiency"] = energy_data
            # Extract key metrics for summary
            if "error" not in energy_data and "delta" in energy_data:
                report["summary"] = report.get("summary", {})
                report["summary"]["energy_ratio"] = energy_data.get("delta", {}).get("energy_per_query_ratio_baseline_over_ctdr", 0.0)
        except Exception as e:
            report["results"]["energy_efficiency"] = {"error": str(e)}

    # 4) Reliability (new schema: FSM precision, semantic errors, token reduction)
    rel_mod = _safe_import("benchmarks.benchmark_reliability")
    if isinstance(rel_mod, Exception):
        report["results"]["reliability"] = {"error": str(rel_mod)}
    else:
        try:
            rel_mod.run_reliability_benchmark()
            report["files"]["reliability"] = str(RESULTS_DIR / "reliability.json")
            rel_data = _read_json(RESULTS_DIR / "reliability.json")
            report["results"]["reliability"] = rel_data
            # Extract key metrics for summary
            if "error" not in rel_data:
                report["summary"] = report.get("summary", {})
                if "fsm_precision_percent" in rel_data:
                    report["summary"]["fsm_precision"] = rel_data["fsm_precision_percent"]
                if "semantic_error_rate_percent" in rel_data:
                    report["summary"]["semantic_error_rate"] = rel_data["semantic_error_rate_percent"]
                if "token_reduction_percent" in rel_data:
                    report["summary"]["token_reduction"] = rel_data["token_reduction_percent"]
        except Exception as e:
            report["results"]["reliability"] = {"error": str(e)}

    # 5) Entropy (new schema: baseline_stats, rla_stats, comparison)
    ent_mod = _safe_import("benchmarks.benchmark_entropy")
    if isinstance(ent_mod, Exception):
        report["results"]["entropy"] = {"error": str(ent_mod)}
    else:
        try:
            ent_mod.run_entropy_benchmark()
            report["files"]["entropy"] = str(RESULTS_DIR / "entropy.json")
            ent_data = _read_json(RESULTS_DIR / "entropy.json")
            report["results"]["entropy"] = ent_data
            # Extract key metrics for summary
            if "error" not in ent_data and "comparison" in ent_data:
                report["summary"] = report.get("summary", {})
                comp = ent_data.get("comparison", {})
                report["summary"]["write_reduction"] = comp.get("write_reduction_factor", 0.0)
                report["summary"]["energy_reduction"] = comp.get("energy_reduction_factor", 0.0)
                report["summary"]["read_efficiency"] = comp.get("read_efficiency", 0.0)
                if "rla_stats" in ent_data:
                    report["summary"]["cache_hit_rate"] = ent_data["rla_stats"].get("cache_hit_rate", 0.0)
        except Exception as e:
            report["results"]["entropy"] = {"error": str(e)}

    # 6) Existing KV cache benchmark (kept for continuity)
    kv_path = RESULTS_DIR / "latest.json"
    report["files"]["kv_cache_latest"] = str(kv_path)
    report["results"]["kv_cache_latest"] = _read_json(kv_path)

    # Save comprehensive report
    COMPREHENSIVE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comprehensive report saved to: {COMPREHENSIVE_FILE}", flush=True)

    # Update latest.json with key metrics and pointer to comprehensive report
    latest = {
        "timestamp": report["timestamp"],
        "comprehensive_report": {
            "path": str(COMPREHENSIVE_FILE),
            "timestamp": report["timestamp"],
        },
        "summary": report.get("summary", {}),
        "status": {
            "performance": "error" not in report["results"].get("performance", {}),
            "gpu_utilization": "error" not in report["results"].get("gpu_utilization", {}),
            "energy_efficiency": "error" not in report["results"].get("energy_efficiency", {}),
            "reliability": "error" not in report["results"].get("reliability", {}),
            "entropy": "error" not in report["results"].get("entropy", {}),
        },
    }
    
    LATEST_FILE.write_text(json.dumps(latest, indent=2), encoding="utf-8")
    print(f"Updated latest.json: {LATEST_FILE}", flush=True)
    
    # Print summary
    if "summary" in latest:
        print("\n" + "=" * 60, flush=True)
        print("SUMMARY", flush=True)
        print("=" * 60, flush=True)
        for key, value in latest["summary"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}", flush=True)
            else:
                print(f"  {key}: {value}", flush=True)

    return report


if __name__ == "__main__":
    run_all()


