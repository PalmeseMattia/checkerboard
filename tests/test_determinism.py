"""Determinism smoke test: same seed + same device => identical results.

Reproducibility guarantee documented in the README. RNG streams differ
between CPU and CUDA, so this is asserted per device (CPU here).
"""

import torch

from src.theory import zipf_importances
from src.train import TrainConfig, train_models


def _short_run(seed: int) -> torch.Tensor:
    cfg = TrainConfig(n=8, alpha=0.8, steps=120, batch_size=128, eval_every=120,
                      eval_batch=2048, eval_chunk=2048, seed=seed, device="cpu")
    I = torch.tensor(zipf_importances(cfg.n), dtype=torch.float32)
    return train_models([2, 4], cfg, I, I)["eval_loss"]


def test_same_seed_same_device_is_bit_identical():
    assert torch.equal(_short_run(0), _short_run(0))


def test_different_seed_differs():
    assert not torch.equal(_short_run(0), _short_run(1))
