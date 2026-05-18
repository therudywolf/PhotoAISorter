# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""CLIP device selection and CUDA install hints."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

TORCH_CUDA_INSTALL_CMD = (
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
)


def _nvidia_gpu_detected() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_clip_device(
    pref: str,
    torch: Any,
    *,
    on_log: Callable[[str], None] | None = None,
) -> Any:
    """Pick cpu/cuda/mps and log why (helps when PyTorch is CPU-only but GPU exists)."""
    p = (pref or "auto").strip().lower()
    cuda_ok = bool(torch.cuda.is_available())

    if p == "cpu":
        if on_log:
            on_log("CLIP: устройство CPU (fast_classify.device=cpu в настройках).")
        return torch.device("cpu")

    if p == "cuda":
        if cuda_ok:
            if on_log:
                on_log(f"CLIP: GPU {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        if on_log:
            on_log("CLIP: cuda запрошена, но PyTorch CUDA недоступен — используется CPU.")
        _log_cpu_torch_hint(torch, on_log)
        return torch.device("cpu")

    if p == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if on_log:
                on_log("CLIP: Apple MPS")
            return torch.device("mps")
        if on_log:
            on_log("CLIP: MPS недоступен — CPU.")
        return torch.device("cpu")

    # auto
    if cuda_ok:
        if on_log:
            on_log(f"CLIP: GPU {torch.cuda.get_device_name(0)} (device=auto)")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if on_log:
            on_log("CLIP: Apple MPS (device=auto)")
        return torch.device("mps")
    if on_log:
        on_log("CLIP: GPU не используется — CPU (device=auto).")
    _log_cpu_torch_hint(torch, on_log)
    return torch.device("cpu")


def _log_cpu_torch_hint(torch: Any, on_log: Callable[[str], None] | None) -> None:
    if on_log is None:
        return
    ver = str(getattr(torch, "__version__", ""))
    if "+cpu" in ver or not _nvidia_gpu_detected():
        if _nvidia_gpu_detected():
            on_log(
                f"CLIP: PyTorch без CUDA ({ver}). Для RTX и др.: {TORCH_CUDA_INSTALL_CMD}"
            )
        return
    if _nvidia_gpu_detected():
        on_log(
            f"CLIP: NVIDIA GPU найден, но torch.cuda недоступен ({ver}). "
            f"Переустановите PyTorch: {TORCH_CUDA_INSTALL_CMD}"
        )

