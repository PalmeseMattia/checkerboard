"""Exp 0 — replication of the Eq. 2 loss floor (Sarkar & Deka 2026).

Trains models of width d_S = 1..d_T directly on the task and compares
achieved floors to the predicted L*(d_S). Per (n, alpha) config it writes
`exp0_<tag>.json` (with per-seed column norms, fractional capacities, and
per-feature MSE — consumed by experiments/predictors.py), the floor-vs-width
figure (fig1) and the survival staircase (fig2).

    python experiments/exp0_replication.py --config configs/exp0_full.yaml
"""

import time

import numpy as np
import torch

from common import dump_json, load_config, tag
from src.metrics import column_norms_sq, feature_capacity, survived_mask
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import d_star, g_alpha, predicted_floor, zipf_importances
from src.train import TrainConfig, train_models

import matplotlib.pyplot as plt


def sweep_d_T(n: int, alpha: float) -> int:
    """Largest width to train: just past the critical width d*."""
    return int(np.ceil(d_star(n, alpha))) + 1


def run_config(cfg: TrainConfig, outdir) -> dict:
    widths = list(range(1, cfg.d_T + 1))
    grid = [d for d in widths for _ in range(cfg.n_seeds)]
    groups = [s for _ in widths for s in range(cfg.n_seeds)]  # data stream per seed
    I = zipf_importances(cfg.n)
    I_t = torch.tensor(I, dtype=torch.float32)

    t0 = time.time()
    res = train_models(grid, cfg, I_t, I_t, data_groups=groups)
    runtime = time.time() - t0

    W, S = len(widths), cfg.n_seeds
    loss = res["eval_loss"].numpy().reshape(W, S)
    predicted = np.array([predicted_floor(d, cfg.n, cfg.alpha, I) for d in widths])
    col_norms = column_norms_sq(res["W"]).numpy().reshape(W, S, cfg.n)
    caps = feature_capacity(res["W"]).numpy().reshape(W, S, cfg.n)
    per_feat = res["per_feature_mse"].numpy().reshape(W, S, cfg.n)
    survived = survived_mask(torch.tensor(col_norms)).sum(-1).numpy()  # (W, S)
    # Degenerate configs (predicted floor identically 0 at every width, e.g.
    # n=20, alpha=0.99) yield r = NaN; the report footnotes them explicitly.
    pearson_r = float(np.corrcoef(np.repeat(predicted, S), loss.flatten())[0, 1])

    out = {
        "config": cfg.asdict(),
        "widths": widths,
        "predicted_floor": predicted.tolist(),
        "achieved_floor": loss.tolist(),  # (widths, seeds)
        "critical_width": d_star(cfg.n, cfg.alpha),
        "capacity_g": g_alpha(cfg.alpha),
        "pearson_r": pearson_r,
        "survived_count": survived.tolist(),  # (widths, seeds)
        "mean_col_norms_sq": col_norms.mean(axis=1).tolist(),  # (widths, n)
        # Per-seed raw arrays consumed by experiments/predictors.py:
        "col_norms_sq": col_norms.tolist(),  # (widths, seeds, n)
        "feature_capacity": caps.tolist(),  # (widths, seeds, n) fractional C_i
        "per_feature_mse": per_feat.tolist(),  # (widths, seeds, n)
        "history": [
            {"step": h["step"], "eval_loss": h["eval_loss"].tolist()}
            for h in res["history"]
        ],
        "runtime_s": runtime,
    }
    t = tag(cfg.n, cfg.alpha)
    dump_json(outdir / f"exp0_{t}.json", out)
    plot_floor_vs_width(out, outdir / f"fig1_floor_{t}.png")
    plot_survival_heatmap(out, outdir / f"fig2_survival_{t}.png")
    print(f"[exp0 {t}] r={pearson_r:.4f}  d*={out['critical_width']:.2f}  "
          f"({runtime:.0f}s)", flush=True)
    return out


def plot_floor_vs_width(out: dict, path) -> None:
    """Fig 1: achieved floor (mean +/- std over seeds) vs predicted, log y."""
    widths = np.array(out["widths"])
    loss = np.array(out["achieved_floor"])
    predicted = np.array(out["predicted_floor"])
    cfg = out["config"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(widths, loss.mean(axis=1), yerr=loss.std(axis=1),
                fmt="o-", capsize=3, color=OKABE_ITO["blue"],
                label="achieved (mean ± std)")
    pos = predicted > 0
    ax.plot(widths[pos], predicted[pos], "s--", color=OKABE_ITO["vermillion"],
            label="predicted $L^*$ (Eq. 2)")
    ax.axvline(out["critical_width"], color="gray", linestyle=":",
               label=f"$d^* = {out['critical_width']:.1f}$")
    ax.set_yscale("log")
    ax.set_xlabel("student width $d_S$")
    ax.set_ylabel("importance-weighted MSE")
    ax.set_title(f"Loss floor vs width  (n={cfg['n']}, α={cfg['alpha']}, "
                 f"{cfg['n_seeds']} seeds, r={out['pearson_r']:.3f})")
    ax.legend()
    save_fig(fig, path)


def plot_survival_heatmap(out: dict, path, kind: str = "direct") -> None:
    """Fig 2: mean ||W_i||^2 per (feature rank, width) — the staircase."""
    cns = np.array(out["mean_col_norms_sq"])  # (widths, n)
    cfg = out["config"]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        cns, aspect="auto", origin="lower", cmap="viridis",
        extent=(0.5, cfg["n"] + 0.5,
                min(out["widths"]) - 0.5, max(out["widths"]) + 0.5),
    )
    ax.axhline(out["critical_width"], color="white", linestyle=":", linewidth=1)
    ax.grid(False)
    ax.set_xlabel("feature rank (1 = most important)")
    ax.set_ylabel("width $d_S$")
    ax.set_title(f"Feature survival ‖W_i‖², {kind} (n={cfg['n']}, α={cfg['alpha']})")
    fig.colorbar(im, ax=ax, label="mean ‖W_i‖² over seeds")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg = load_config("exp0_full.yaml", __doc__)
    for n, alpha in [tuple(c) for c in cfg.configs]:
        d_T = cfg.d_T if getattr(cfg, "d_T", None) else sweep_d_T(n, alpha)
        tc = TrainConfig(n=n, alpha=alpha, d_T=d_T, n_seeds=cfg.seeds,
                         steps=cfg.steps, seed=cfg.seed, device=cfg.device)
        run_config(tc, cfg.outdir)


if __name__ == "__main__":
    main()
