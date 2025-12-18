"""
CLI entrypoint for QuantumFlow prototypes.
"""

import sys
from pathlib import Path


def smoke():
    """Run ecosystem smoke test."""
    # Add prototypes to path
    prototypes_dir = Path(__file__).parent.parent / "prototypes"
    if str(prototypes_dir) not in sys.path:
        sys.path.insert(0, str(prototypes_dir))
    
    # Import and run smoke test
    try:
        from ecosystem_smoke import main
        main()
    except ImportError as e:
        print(f"Error: Could not import ecosystem_smoke: {e}")
        print(f"Make sure prototypes/ directory is available at: {prototypes_dir}")
        sys.exit(1)


def help():
    """Print help message."""
    print("""
QuantumFlow — GPU-Accelerated Prototypes Ecosystem

Commands:
  python -m quantumflow smoke    Run ecosystem smoke test

For full documentation, see:
  https://github.com/<ORG>/<REPO>

Quick Start:
  pip install -r prototypes/requirements.gpu-cu12.txt
  python prototypes/ecosystem_smoke.py
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        help()

