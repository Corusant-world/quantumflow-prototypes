#!/bin/bash
# Update four-layer versioning with git hash

cd "$(dirname "$0")/.."

# Get git hash (if in git repo)
if git rev-parse --git-dir > /dev/null 2>&1; then
    GIT_HASH=$(git rev-parse HEAD)
    echo "Git hash: $GIT_HASH"
    
    # Update versioning.json with git hash
    if [ -f "benchmarks/results/versioning.json" ]; then
        # Use Python to update JSON (more reliable than sed)
        python3 <<EOF
import json
with open('benchmarks/results/versioning.json', 'r') as f:
    data = json.load(f)
data['four_layer_versioning']['layer_1_code']['git_hash'] = '$GIT_HASH'
with open('benchmarks/results/versioning.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Updated versioning.json with git hash: $GIT_HASH")
EOF
    fi
else
    echo "Not a git repository, skipping git hash update"
fi

