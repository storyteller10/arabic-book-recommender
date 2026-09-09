"""Shared project-relative paths; environment settings are optional."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def configured_path(variable, default):
    return project_path(os.environ.get(variable, default))
