import torch

from src.theory import zipf_importances
from src.train import TrainConfig, train_models


def test_short_training_decreases_loss():
    cfg = TrainConfig(
        n=8, alpha=0.8, steps=400, batch_size=256, eval_every=100,
        eval_batch=4096, eval_chunk=4096, device="cpu",
    )
    I = torch.tensor(zipf_importances(cfg.n), dtype=torch.float32)
    res = train_models([2, 4], cfg, I, I)
    first = res["history"][0]["eval_loss"]
    assert torch.all(res["eval_loss"] < first)
    assert torch.all(torch.isfinite(res["eval_loss"]))


def test_output_shapes():
    cfg = TrainConfig(
        n=8, alpha=0.8, steps=50, batch_size=128, eval_every=50,
        eval_batch=1024, eval_chunk=1024, device="cpu",
    )
    I = torch.tensor(zipf_importances(cfg.n), dtype=torch.float32)
    res = train_models([1, 2, 3], cfg, I, I)
    assert res["W"].shape == (3, 3, 8)
    assert res["b"].shape == (3, 8)
    assert res["eval_loss"].shape == (3,)
    assert res["per_feature_mse"].shape == (3, 8)
    # Masked rows of the returned weights are exactly zero.
    assert torch.all(res["W"][0, 1:, :] == 0)
