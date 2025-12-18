"""
CTDR Benchmarks (legacy entrypoint)

This script is kept for backwards compatibility with README commands.
It now delegates to `benchmarks/run_all_benchmarks.py` which produces:
- benchmarks/results/comprehensive_report.json
- benchmarks/results/latest.json (updated)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import benchmarks package when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.run_all_benchmarks import run_all


def main() -> None:
    run_all()


if __name__ == "__main__":
    main()


