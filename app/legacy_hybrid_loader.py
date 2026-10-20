"""Load tested functions from the current hybrid perception script without running its main pipeline.

This is a transition shim for the module split. It lets the new package layout expose
the existing implementation while the old monolithic script remains available and
unchanged for behavior parity.
"""

from pathlib import Path


_CACHE = None


def load_symbols():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    root = Path(__file__).resolve().parent
    source_path = root / "scripts" / "hybrid_opencv_sam.py"
    source = source_path.read_text()
    definitions = source.split("# ---------------- MAIN ----------------", 1)[0]
    namespace = {
        "__file__": str(source_path),
        "__name__": "legacy_hybrid_definitions",
    }
    exec(compile(definitions, str(source_path), "exec"), namespace)
    _CACHE = namespace
    return namespace


def get_symbol(name):
    return load_symbols()[name]

