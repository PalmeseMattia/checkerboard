"""Exp A — allocation audit: does vanilla distillation respect assumption A2?

One teacher of width d_T per seed is trained on the task; students of
widths 1..d_T are distilled against their seed's teacher output.
Direct-training controls (same widths, seeds, and data streams) run in the
same batched computation. Reported per student: overlap@k (A2 ordering),
|S| vs F_kept (A2 count), and teacher containment |S_s ∩ S_T| / |S_s|.

    python experiments/expA_audit.py --config configs/expA_full.yaml
"""

import time

import numpy as np
import torch

from common import dump_json, load_config, tag
from exp0_replication import plot_survival_heatmap
from src.metrics import column_norms_sq, overlap_at_k, survived_mask
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import F_kept, d_star, g_alpha, predicted_floor, zipf_importances
from src.train import TrainConfig, make_distillation_teacher, resolve_device, train_models

import matplotlib.pyplot as plt

SETTINGS = ["distill", "direct"]


def run_config(cfg: TrainConfig, outdir) -> dict:
    I = zipf_importances(cfg.n)
    I_t = torch.tensor(I, dtype=torch.float32)
    device = resolve_device(cfg.device)
    t0 = time.time()

    # Teachers: one width-d_T model per seed, trained directly on the task.
    t_res = train_models([cfg.d_T] * cfg.n_seeds, cfg, I_t, I_t,
                         data_groups=list(range(cfg.n_seeds)))
    t_survived = survived_mask(column_norms_sq(t_res["W"])).numpy()  # (seeds, n)

    # Students: (width x setting x seed), distilled and direct in one run.
    widths = list(range(1, cfg.d_T + 1))
    grid, groups, flags = [], [], []
    for d in widths:
        for setting in SETTINGS:
            for s in range(cfg.n_seeds):
                grid.append(d)
                groups.append(s)  # same data stream per seed across settings
                flags.append(setting == "distill")
    teacher = make_distillation_teacher(t_res["W"], t_res["b"], groups, flags, device)
    # Fresh data stream relative to the teachers' training draws.
    s_cfg = TrainConfig(**{**cfg.asdict(), "seed": cfg.seed + 1000})
    res = train_models(grid, s_cfg, I_t, I_t, teacher=teacher, data_groups=groups)
    runtime = time.time() - t0

    shape = (len(widths), len(SETTINGS), cfg.n_seeds)
    col_norms = column_norms_sq(res["W"]).numpy().reshape(*shape, cfg.n)
    survived = survived_mask(torch.tensor(col_norms)).numpy()  # (W, 2, S, n)
    loss = res["eval_loss"].numpy().reshape(shape)
    per_feature = res["per_feature_mse"].numpy().reshape(*shape, cfg.n)

    overlap = np.array([
        [[overlap_at_k(survived[w, k, s], I) for s in range(cfg.n_seeds)]
         for k in range(len(SETTINGS))]
        for w in range(len(widths))
    ])  # (W, 2, S)
    n_surv = survived.sum(-1)  # (W, 2, S)
    inter = (survived & t_survived[None, None]).sum(-1)
    containment = np.where(n_surv > 0, inter / np.maximum(n_surv, 1), np.nan)

    out = {
        "config": cfg.asdict(),
        "widths": widths,
        "settings": SETTINGS,
        "critical_width": d_star(cfg.n, cfg.alpha),
        "capacity_g": g_alpha(cfg.alpha),
        "theory_F": [F_kept(d, cfg.alpha, cfg.n) for d in widths],
        "predicted_floor": [predicted_floor(d, cfg.n, cfg.alpha, I) for d in widths],
        "teacher": {
            "eval_loss": t_res["eval_loss"].tolist(),
            "n_survived": t_survived.sum(-1).tolist(),
            "survived": [np.flatnonzero(m).tolist() for m in t_survived],
        },
        # Each of these: {setting: (widths, seeds) nested list}.
        "overlap_at_k": {k: overlap[:, i].tolist() for i, k in enumerate(SETTINGS)},
        "n_survived": {k: n_surv[:, i].tolist() for i, k in enumerate(SETTINGS)},
        "teacher_containment": {
            k: containment[:, i].tolist() for i, k in enumerate(SETTINGS)
        },
        "achieved_floor": {k: loss[:, i].tolist() for i, k in enumerate(SETTINGS)},
        "mean_col_norms_sq": {
            k: col_norms[:, i].mean(axis=1).tolist() for i, k in enumerate(SETTINGS)
        },
        "per_feature_mse": {
            k: per_feature[:, i].mean(axis=1).tolist() for i, k in enumerate(SETTINGS)
        },
        "runtime_s": runtime,
    }
    t = tag(cfg.n, cfg.alpha)
    dump_json(outdir / f"expA_{t}.json", out)

    plot_overlap(out, outdir / f"fig3_overlap_{t}.png")
    heat = {"mean_col_norms_sq": out["mean_col_norms_sq"]["distill"],
            "widths": widths, "critical_width": out["critical_width"],
            "config": out["config"]}
    plot_survival_heatmap(heat, outdir / f"fig2_survival_distill_{t}.png",
                          kind="distilled")
    ov_d = np.nanmean(overlap[:, 0], axis=1)
    print(f"[expA {t}] overlap@k distill: min={np.nanmin(ov_d):.3f} at "
          f"d_S={widths[int(np.nanargmin(ov_d))]}  teacher |S_T|="
          f"{np.mean(out['teacher']['n_survived']):.1f}  ({runtime:.0f}s)",
          flush=True)
    return out


def plot_overlap(out: dict, path) -> None:
    """Fig 3 (Exp A headline): overlap@k and |S| vs d_S, distill vs direct."""
    widths = np.array(out["widths"])
    cfg = out["config"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for setting, color in zip(out["settings"],
                              (OKABE_ITO["blue"], OKABE_ITO["orange"])):
        ov = np.array(out["overlap_at_k"][setting])  # (W, seeds)
        axes[0].errorbar(widths, np.nanmean(ov, axis=1), yerr=np.nanstd(ov, axis=1),
                         fmt="o-", color=color, capsize=3, label=setting)
        ns = np.array(out["n_survived"][setting])
        axes[1].errorbar(widths, ns.mean(axis=1), yerr=ns.std(axis=1),
                         fmt="o-", color=color, capsize=3, label=f"|S| {setting}")
    axes[0].axhline(1.0, color="gray", linestyle=":", label="A2 (exact)")
    axes[0].set_xlabel("student width $d_S$")
    axes[0].set_ylabel("overlap@k")
    axes[0].set_ylim(None, 1.05)
    axes[0].legend()
    axes[1].plot(widths, out["theory_F"], "s--", color=OKABE_ITO["vermillion"],
                 label="$F_{kept}$ = ⌊d·g(α)⌋")
    axes[1].axhline(np.mean(out["teacher"]["n_survived"]), color="k",
                    linestyle=":", label=f"teacher |S_T| (d_T={cfg['d_T']})")
    axes[1].set_xlabel("student width $d_S$")
    axes[1].set_ylabel("survived features |S|")
    axes[1].legend()
    fig.suptitle(f"Exp A — allocation audit under distillation "
                 f"(n={cfg['n']}, α={cfg['alpha']}, {cfg['n_seeds']} seeds)")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg = load_config("expA_full.yaml", __doc__)
    tc = TrainConfig(n=cfg.n, alpha=cfg.alpha, d_T=cfg.d_T, n_seeds=cfg.seeds,
                     steps=cfg.steps, seed=cfg.seed, device=cfg.device)
    run_config(tc, cfg.outdir)


if __name__ == "__main__":
    main()
