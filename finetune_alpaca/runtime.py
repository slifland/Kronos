from __future__ import annotations

import os


def is_distributed_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def detect_best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

