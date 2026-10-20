"""Python entrypoint for the experimental DAVIS-only demo."""

import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    return subprocess.call([str(root / "run_davis_only_demo.sh")], cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())

