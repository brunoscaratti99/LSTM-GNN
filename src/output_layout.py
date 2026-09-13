"""Canonical on-disk layout for generated experiment artifacts.

New runs keep machine-readable artifacts in ``logs/`` and training-history
figures in ``train_history/``.  Readers fall back to the former flat layout so
existing experiments remain usable.
"""

from __future__ import annotations

from pathlib import Path


LOGS_DIRECTORY = "logs"
TRAIN_HISTORY_DIRECTORY = "train_history"


def logs_directory(run_dir: Path | str, *, create: bool = False) -> Path:
    """Return the run's log directory, optionally creating it."""
    directory = Path(run_dir) / LOGS_DIRECTORY
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def train_history_directory(run_dir: Path | str, *, create: bool = False) -> Path:
    """Return the directory reserved for training-history figures."""
    directory = Path(run_dir) / TRAIN_HISTORY_DIRECTORY
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_artifact_path(run_dir: Path | str, filename: str) -> Path:
    """Return the canonical destination for a machine-readable run artifact."""
    return logs_directory(run_dir) / filename


def resolve_run_artifact(run_dir: Path | str, filename: str) -> Path:
    """Resolve a new-layout artifact, falling back to a legacy flat run."""
    run_dir = Path(run_dir)
    logged = log_artifact_path(run_dir, filename)
    legacy = run_dir / filename
    if logged.is_file() or not legacy.is_file():
        return logged
    return legacy
