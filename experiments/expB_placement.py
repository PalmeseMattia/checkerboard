"""Exp B — controlled placement: boost training weights of a critical set C.

Students train with I-tilde (true I boosted by beta on the critical set C)
and are always evaluated with the true I. Two targets per (d_S, beta, seed):
'teacher' distills against a width-d_T teacher's output (the Exp B of the
project spec); 'task' trains against the true x with the same boosted
weights — the control that isolates the placement cost predicted by Eq. 2
from the teacher-menu constraint found in Exp A (distilled students only
ever keep features their teacher represents).

    python experiments/expB_placement.py --config configs/expB_full.yaml
"""

import time

import numpy as np
import torch

from common import dump_json, load_config, tag
from src.metrics import column_norms_sq, survived_mask
from src.plotting import save_fig, set_style
from src.theory import F_kept, d_star, g_alpha, predicted_floor, zipf_importances
from src.train import TrainConfig, make_distillation_teacher, resolve_device, train_models

import matplotlib.pyplot as plt

TARGETS = ["teacher", "task"]  # distillation vs placement-only control


def critical_ranks(n: int) -> list[int]:
    """Low-importance critical set C (1-indexed ranks): {20, 28, 36} at n=40."""
    return [n // 2, int(0.7 * n), int(0.9 * n)]


def run_config(cfg: TrainConfig, outdir, widths: list[int], betas: list[int]) -> dict:
    I = zipf_importances(cfg.n)
    I_t = torch.tensor(I, dtype=torch.float32)
    device = resolve_device(cfg.device)
    C_ranks = critical_ranks(cfg.n)
    C = [r - 1 for r in C_ranks]  # 0-indexed feature ids
    t0 = time.time()

    t_res = train_models([cfg.d_T] * cfg.n_seeds, cfg, I_t, I_t,
                         data_groups=list(range(cfg.n_seeds)))
    t_survived = survived_mask(column_norms_sq(t_res["W"])).numpy()  # (seeds, n)

    grid, groups, flags, I_rows = [], [], [], []
    for d in widths:
        for target in TARGETS:
            for beta in betas:
                for s in range(cfg.n_seeds):
                    grid.append(d)
                    groups.append(s)
                    flags.append(target == "teacher")
                    I_b = I.copy()
                    I_b[C] *= beta
                    I_rows.append(I_b)
    teacher = make_distillation_teacher(t_res["W"], t_res["b"], groups, flags, device)
    s_cfg = TrainConfig(**{**cfg.asdict(), "seed": cfg.seed + 1000})
    I_train = torch.tensor(np.stack(I_rows), dtype=torch.float32)
    res = train_models(grid, s_cfg, I_train, I_t, teacher=teacher, data_groups=groups)
    runtime = time.time() - t0

    shape = (len(widths), len(TARGETS), len(betas), cfg.n_seeds)
    col_norms = column_norms_sq(res["W"]).numpy().reshape(*shape, cfg.n)
    survived = survived_mask(torch.tensor(col_norms)).numpy()  # (..., n)
    loss = res["eval_loss"].numpy().reshape(shape)
    per_feature = res["per_feature_mse"].numpy().reshape(*shape, cfg.n)

    crit_survival = survived[..., C].mean(-1)  # (W, T, B, S)
    delta_L = loss - loss[:, :, :1]  # paired vs beta=1 (same width/target/seed)
    # Evicted to make room: survived under beta=1 but dead under beta.
    evicted = (survived[:, :, :1] & ~survived).sum(-1)  # (W, T, B, S)

    # Theory: Eq. 2 with the ordering induced by I-tilde (charges TRUE I).
    pred_floor, pred_surv = [], []
    for d in widths:
        row_f, row_s = [], []
        for beta in betas:
            I_b = I.copy()
            I_b[C] *= beta
            row_f.append(predicted_floor(d, cfg.n, cfg.alpha, I, order_by=I_b))
            kept = np.argsort(-I_b, kind="stable")[:F_kept(d, cfg.alpha, cfg.n)]
            row_s.append(float(np.isin(C, kept).mean()))
        pred_floor.append(row_f)
        pred_surv.append(row_s)
    pred_floor = np.array(pred_floor)  # (W, B)
    pred_delta = pred_floor - pred_floor[:, :1]

    out = {
        "config": cfg.asdict(),
        "widths": list(widths),
        "betas": list(betas),
        "targets": TARGETS,
        "critical_ranks": C_ranks,
        "critical_width": d_star(cfg.n, cfg.alpha),
        "capacity_g": g_alpha(cfg.alpha),
        "teacher": {
            "eval_loss": t_res["eval_loss"].tolist(),
            "n_survived": t_survived.sum(-1).tolist(),
            "critical_in_teacher": t_survived[:, C].mean(-1).tolist(),  # per seed
        },
        # Achieved, per (width, target, beta, seeds):
        "critical_survival": crit_survival.tolist(),
        "achieved_floor": loss.tolist(),
        "delta_floor": delta_L.tolist(),
        "n_evicted": evicted.tolist(),
        "n_survived": survived.sum(-1).tolist(),
        # Predicted by Eq. 2 under the I-tilde ordering, per (width, beta):
        "predicted_floor_placement": pred_floor.tolist(),
        "predicted_delta_floor": pred_delta.tolist(),
        "predicted_critical_survival": pred_surv,
        "per_feature_mse": per_feature.mean(-2).tolist(),  # (W, T, B, n) seed-mean
        "runtime_s": runtime,
    }
    t = tag(cfg.n, cfg.alpha)
    dump_json(outdir / f"expB_{t}.json", out)

    plot_pareto(out, outdir / f"fig4_pareto_{t}.png")
    plot_per_feature(out, outdir / f"fig5_perfeature_{t}.png")
    surv_t = crit_survival.mean(-1)  # (W, T, B)
    print(f"[expB {t}] C={C_ranks}  max C-survival (distill): "
          f"{surv_t[:, 0].max():.2f}  (task control: {surv_t[:, 1].max():.2f})  "
          f"teacher carries C: {np.mean(out['teacher']['critical_in_teacher']):.2f}  "
          f"({runtime:.0f}s)", flush=True)
    return out


def plot_pareto(out: dict, path) -> None:
    """Fig 4 (Exp B headline): critical survival vs true floor cost."""
    widths = out["widths"]
    betas = out["betas"]
    cfg = out["config"]
    surv = np.array(out["critical_survival"])  # (W, T, B, S)
    dL = np.array(out["delta_floor"])

    fig, axes = plt.subplots(1, len(out["targets"]), figsize=(11, 4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for t_i, (target, ax) in enumerate(zip(out["targets"], axes)):
        for w_i, d in enumerate(widths):
            color = cmap(w_i / max(len(widths) - 1, 1))
            x = dL[w_i, t_i].mean(-1)
            y = surv[w_i, t_i].mean(-1)
            ax.plot(x, y, "o-", color=color, label=f"$d_S$={d}")
            for b_i, beta in enumerate(betas):
                ax.annotate(f"β={beta}", (x[b_i], y[b_i]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
            ax.plot(np.array(out["predicted_delta_floor"][w_i]),
                    np.array(out["predicted_critical_survival"][w_i]),
                    "s--", color=color, alpha=0.4, markerfacecolor="none",
                    label=f"Eq. 2 pred. $d_S$={d}" if t_i == 1 else None)
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("floor cost ΔL (true-importance MSE)")
        ax.set_title(f"target: {target}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"survival of C = ranks {out['critical_ranks']}")
    fig.suptitle(
        f"Exp B — controlled placement Pareto (n={cfg['n']}, α={cfg['alpha']}, "
        f"d_T={cfg['d_T']}, {cfg['n_seeds']} seeds; teacher carries C: "
        f"{np.mean(out['teacher']['critical_in_teacher']):.2f})")
    save_fig(fig, path)


def plot_per_feature(out: dict, path) -> None:
    """Fig 5: per-feature MSE, vanilla vs strongest boost, C highlighted."""
    cfg = out["config"]
    widths = out["widths"]
    w_i = len(widths) // 2  # middle width
    b_hi = len(out["betas"]) - 1
    mse = np.array(out["per_feature_mse"])  # (W, T, B, n)
    ranks = np.arange(1, cfg["n"] + 1)

    fig, axes = plt.subplots(1, len(out["targets"]), figsize=(11, 4), sharey=True)
    for t_i, (target, ax) in enumerate(zip(out["targets"], axes)):
        ax.plot(ranks, mse[w_i, t_i, 0], "o-", label="vanilla (β=1)")
        ax.plot(ranks, mse[w_i, t_i, b_hi], "o-",
                label=f"placement (β={out['betas'][b_hi]})")
        for r in out["critical_ranks"]:
            ax.axvline(r, color="#D55E00", alpha=0.3, linewidth=6)
        ax.set_yscale("log")
        ax.set_xlabel("feature rank")
        ax.set_title(f"target: {target} ($d_S$={widths[w_i]})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("per-feature MSE (true task)")
    fig.suptitle(f"Exp B — per-feature cost of placement (n={cfg['n']}, "
                 f"α={cfg['alpha']}; bands = critical set C)")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg = load_config("expB_full.yaml", __doc__)
    tc = TrainConfig(n=cfg.n, alpha=cfg.alpha, d_T=cfg.d_T, n_seeds=cfg.seeds,
                     steps=cfg.steps, seed=cfg.seed, device=cfg.device)
    run_config(tc, cfg.outdir, widths=list(cfg.widths), betas=list(cfg.betas))


if __name__ == "__main__":
    main()
