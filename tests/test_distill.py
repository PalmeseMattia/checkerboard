import torch

from src.theory import zipf_importances
from src.train import TrainConfig, make_distillation_teacher, train_models


def _toy_teachers(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two hand-built width-2 teachers with distinct, recognizable outputs."""
    W = torch.zeros(2, 2, n)
    W[0, 0, 0] = 1.0  # teacher 0 represents feature 0
    W[1, 0, 1] = 1.0  # teacher 1 represents feature 1
    b = torch.zeros(2, n)
    return W, b


def test_teacher_callable_maps_models_to_their_teacher():
    n = 4
    W, b = _toy_teachers(n)
    # Models 0/1 distill from teachers 0/1; model 2 is a direct control.
    teacher = make_distillation_teacher(
        W, b, teacher_of_model=[0, 1, 0], distill_flags=[True, True, False],
        device=torch.device("cpu"),
    )
    x = torch.ones(3, 5, n)
    out = teacher(x)
    assert out.shape == (3, 5, n)
    # Teacher 0 reconstructs only feature 0, teacher 1 only feature 1.
    assert torch.allclose(out[0], torch.eye(n)[0].expand(5, n))
    assert torch.allclose(out[1], torch.eye(n)[1].expand(5, n))
    # Direct control receives the true x.
    assert torch.allclose(out[2], x[2])


def test_distillation_training_runs_and_tracks_teacher():
    cfg = TrainConfig(
        n=8, alpha=0.8, steps=400, batch_size=256, eval_every=200,
        eval_batch=4096, eval_chunk=4096, device="cpu",
    )
    I = torch.tensor(zipf_importances(cfg.n), dtype=torch.float32)
    t_res = train_models([4, 4], cfg, I, I, data_groups=[0, 1])
    teacher = make_distillation_teacher(
        t_res["W"], t_res["b"], teacher_of_model=[0, 1], distill_flags=[True, True],
        device=torch.device("cpu"),
    )
    res = train_models([4, 4], cfg, I, I, teacher=teacher, data_groups=[0, 1])
    # Distilled students approach their teacher's true-task loss.
    assert torch.all(torch.isfinite(res["eval_loss"]))
    assert torch.all(res["eval_loss"] < 2 * t_res["eval_loss"] + 0.05)
