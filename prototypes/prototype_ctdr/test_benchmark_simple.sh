#!/bin/bash
# Simple test script to verify benchmark can run on GPU server

cd /root/tech-eldorado-infrastructure/prototypes/prototype_ctdr

echo "=== Testing benchmark_performance.py ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Python path: $(python3 -c 'import sys; print(sys.path[:3])')"

echo ""
echo "=== Running benchmark ==="
python3 benchmarks/benchmark_performance.py

echo ""
echo "=== Checking results ==="
if [ -f "benchmarks/results/performance.json" ]; then
    echo "Results file exists!"
    python3 -c "import json; f=open('benchmarks/results/performance.json'); d=json.load(f); print('Timestamp:', d.get('timestamp', 'N/A')); print('Benchmarks:', list(d.get('benchmarks', {}).keys()))"
else
    echo "ERROR: Results file not found!"
    exit 1
fi


