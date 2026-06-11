"""Vectorized training: all (width, seed) models as one batched computation."""

from dataclasses import asdict, dataclass, field

import torch

from src.data import sample_features
from src.model import BatchedToyModels


@dataclass
class TrainConfig:
    n: int = 40
    d_T: int = 10
    alpha: float = 0.90
    n_seeds: int = 5
    steps: int = 10_000
    batch_size: int = 1024
    lr: float = 3e-3
    eval_every: int = 500
    eval_batch: int = 65_536
    eval_chunk: int = 8_192
    seed: int = 0
    eval_seed: int = 7_777
    device: str = "auto"

    def asdict(self) -> dict:
        return asdict(self)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def importance_weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, importances: torch.Tensor
) -> torch.Tensor:
    """L_m = E_batch[ sum_i I_i (pred_i - target_i)^2 ] for each model m.

    pred/target: (M, B, n); importances: (n,) shared or (M, n) per-model.
    Returns (M,).
    """
    I = importances if importances.dim() == 2 else importances.unsqueeze(0)
    return ((pred - target) ** 2 * I.unsqueeze(1)).sum(-1).mean(-1)


@torch.no_grad()
def evaluate(
    model: BatchedToyModels,
    cfg: TrainConfig,
    importances: torch.Tensor,
    sampler=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """True-task loss (M,) and per-feature MSE (M, n) on a fixed eval set.

    Always evaluates against the TRUE features x with the TRUE
    importances, regardless of what the model was trained on. The eval
    set is shared across models and fixed by cfg.eval_seed; chunked to
    bound memory at (M, eval_chunk, n).
    """
    device = model.W.device
    gen = torch.Generator(device=device).manual_seed(cfg.eval_seed)
    draw = sampler or sample_features
    M = len(model.widths)
    loss = torch.zeros(M, device=device)
    per_feature = torch.zeros(M, cfg.n, device=device)
    done = 0
    while done < cfg.eval_batch:
        b = min(cfg.eval_chunk, cfg.eval_batch - done)
        x = draw((b,), cfg.n, cfg.alpha, gen, device)
        sq = (model(x) - x) ** 2  # (M, b, n)
        per_feature += sq.sum(dim=1)
        loss += (sq * importances).sum(dim=-1).sum(dim=-1)
        done += b
    return loss / cfg.eval_batch, per_feature / cfg.eval_batch


def make_distillation_teacher(
    W_T: torch.Tensor,
    b_T: torch.Tensor,
    teacher_of_model: list[int],
    distill_flags: list[bool],
    device: torch.device,
):
    """Teacher callable for `train_models(..., teacher=...)`.

    W_T: (T, d_T, n), b_T: (T, n) — trained teacher weights (one teacher
        per seed, as returned by a previous train_models call).
    teacher_of_model: index into the teacher dim per student model
        (the caller maps each student to its seed's teacher).
    distill_flags: per student model; False means the model is a
        direct-training control and receives the true x as target.

    Returns f(x: (M, B, n)) -> (M, B, n) targets.
    """
    Wg = W_T.to(device)[torch.tensor(teacher_of_model, device=device)]
    bg = b_T.to(device)[torch.tensor(teacher_of_model, device=device)]
    flags = torch.tensor(distill_flags, device=device).view(-1, 1, 1)

    def teacher(x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("mdn,mbn->mbd", Wg, x)
        out = torch.relu(torch.einsum("mdn,mbd->mbn", Wg, h) + bg.unsqueeze(1))
        return torch.where(flags, out, x)

    return teacher


def train_models(
    widths_grid: list[int],
    cfg: TrainConfig,
    train_importances: torch.Tensor,
    eval_importances: torch.Tensor,
    teacher=None,
    data_groups: list[int] | None = None,
    sampler=None,
    regularizer=None,
) -> dict:
    """Train len(widths_grid) independent models in parallel.

    widths_grid: hidden width of each model (the widths x seeds grid,
        flattened by the caller).
    train_importances: (n,) or (M, n) weights for the training loss
        (Exp B uses per-model boosted weights I-tilde).
    eval_importances: (n,) true importances, used for evaluation only.
    teacher: optional callable (M, B, n) -> (M, B, n); when given, the
        training target is teacher(x) instead of x (distillation).
    data_groups: group id (0..G-1) per model; models in the same group
        see the same training batches. The caller groups by seed so the
        per-step RNG cost is G batches instead of M (RNG dominates the
        step time otherwise); widths within a seed are paired on data.
        Default: one shared stream for all models.
    sampler: optional drop-in for sample_features (same signature);
        used for BOTH training batches and the eval set (Exp C trains
        and evaluates on block-correlated data).
    regularizer: optional callable model -> (M,) penalty added to each
        model's training loss (Exp C group-alignment placement loss).

    Returns a dict of CPU tensors: final masked weights, biases, eval
    loss/per-feature MSE on the true task, and the eval-loss history.
    """
    device = resolve_device(cfg.device)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    model = BatchedToyModels(widths_grid, cfg.n, generator=gen, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    I_train = train_importances.to(device)
    I_eval = eval_importances.to(device)
    group_idx = None
    n_groups = 1
    if data_groups is not None:
        group_idx = torch.tensor(data_groups, device=device)
        n_groups = max(data_groups) + 1

    draw = sampler or sample_features
    history = []
    for step in range(cfg.steps):
        x = draw((n_groups, cfg.batch_size), cfg.n, cfg.alpha, gen, device)
        x = x.expand(len(widths_grid), -1, -1) if group_idx is None else x[group_idx]
        with torch.no_grad():
            target = x if teacher is None else teacher(x)
        losses = importance_weighted_mse(model(x), target, I_train)
        if regularizer is not None:
            losses = losses + regularizer(model)
        opt.zero_grad()
        losses.sum().backward()
        opt.step()
        sched.step()
        if (step + 1) % cfg.eval_every == 0:
            eval_loss, _ = evaluate(model, cfg, I_eval, sampler=sampler)
            history.append({"step": step + 1, "eval_loss": eval_loss.cpu()})

    eval_loss, per_feature_mse = evaluate(model, cfg, I_eval, sampler=sampler)
    return {
        "widths_grid": list(widths_grid),
        "W": model.W_eff.detach().cpu(),
        "b": model.b.detach().cpu(),
        "eval_loss": eval_loss.cpu(),
        "per_feature_mse": per_feature_mse.cpu(),
        "history": history,
    }
