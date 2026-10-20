"""PyQt dashboard entry point."""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gui_viewer import GraspGUI


def main():
    app = QApplication(sys.argv)
    window = GraspGUI()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
