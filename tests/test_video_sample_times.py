# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Video sample time positions (no ffmpeg required)."""

from pathlib import Path

from app.video_frames import diagnose_media_decode
from app.video_frames import video_sample_times_sec


def test_three_points_full_duration() -> None:
    t = video_sample_times_sec(100.0, 3)
    assert len(t) == 3
    assert t[0] == 0.0
    assert abs(t[1] - 50.0) < 0.02
    assert t[2] < 100.0


def test_single_point() -> None:
    assert video_sample_times_sec(10.0, 1) == [0.0]


def test_zero_duration() -> None:
    assert video_sample_times_sec(0.0, 3) == [0.0, 0.0, 0.0]


def test_diagnose_missing_video_does_not_raise(tmp_path: Path) -> None:
    report = diagnose_media_decode(tmp_path / "missing.mp4", 3)
    assert report["exists"] is False
    assert report["decoded_frames"] == 0
    assert report["wanted_frames"] == 3
