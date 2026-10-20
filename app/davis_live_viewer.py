"""Deprecated DAVIS live-viewer compatibility entry point.

The normal DAVIS operator interface is the main Flask dashboard:
`./scripts/run_davis_motion_demo.sh` then open `http://127.0.0.1:8080/?tab=davis`.

The legacy implementation was archived at:
`archive/legacy_scripts/davis_guis/davis_live_viewer.py`.
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deprecated DAVIS live viewer wrapper.")
    parser.parse_args(argv)
    print("The standalone DAVIS live viewer is archived and no longer opens the camera.")
    print("Use the main web dashboard for DAVIS Motion.")
    print("Run: ./scripts/run_davis_motion_demo.sh")
    print("Then open http://127.0.0.1:8080/?tab=davis")
    return 2


def run_gui(*_args, **_kwargs):
    """Compatibility stub that refuses to open a competing DAVIS GUI."""
    return main([])


if __name__ == "__main__":
    raise SystemExit(main())
