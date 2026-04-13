"""Shared task state model."""

from __future__ import annotations

from enum import StrEnum


class TaskState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERROR = "error"
