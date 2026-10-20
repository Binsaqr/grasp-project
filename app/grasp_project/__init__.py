"""Source-layout compatibility facade for ``python3 -m grasp_project...``.

The implementation package lives in ``src/grasp_project``. This shim lets
module commands run from the repository root without requiring an editable
install first.
"""

from pathlib import Path
from pkgutil import extend_path

_ROOT = Path(__file__).resolve().parents[1]
_SRC_PACKAGE = _ROOT / "src" / "grasp_project"

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
if _SRC_PACKAGE.exists():
    src_path = str(_SRC_PACKAGE)
    if src_path not in __path__:
        __path__.append(src_path)

