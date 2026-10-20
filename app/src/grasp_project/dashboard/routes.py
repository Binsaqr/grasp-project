"""Dashboard route compatibility exports.

This wrapper intentionally exposes the existing Flask app without starting a
server, so import tests and collaborator tooling can load it without cameras.
"""

from grasp_project.dashboard.web_dashboard import app

__all__ = ["app"]
