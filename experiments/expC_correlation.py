"""Exp C — correlation-aware packing on block-anticorrelated data.

Groups are consecutive importance ranks (group 0 = the top `group_size`
features), mutually exclusive within a group. Settings, each trained as one
batched (widths x seeds) call: 'iid' baseline; 'blocked' (train+eval on
mutually exclusive groups — does emergent superposition exploit
anti-correlation? claim a); 'blocked λ=...' adds the group-alignment
placement loss (claim b). The Eq. 2 floor L* is unchanged by blocking
(identical per-feature marginals), so beating it on blocked data
quantifies the iid assumption's slack.

    python experiments/expC_correlation.py --config configs/expC_full.yaml
"""

import time
import warnings

import numpy as np
import torch

from common import dump_json, load_config, tag
from src.data import sample_features_blocked
from src.metrics import column_norms_sq, feature_capacity, survived_mask
from src.plotting import CYCLE, save_fig, set_style
from src.theory import d_star, predicted_floor, zipf_importances
from src.train import TrainConfig, resolve_device, train_models

import matplotlib.pyplot as plt


def make_group_alignment_regularizer(n: int, group_size: int, lam: float, device):
    """Placement loss packing mutually exclusive features into shared directions.

    Penalty per model: lam * mean over same-group pairs (i != j) of
    ||W_i||^2 ||W_j||^2 - (W_i . W_j)^2  (>= 0 by Cauchy-Schwarz; 0 iff
    the pair is parallel/antiparallel or one column is dead).
    """
    gid = torch.arange(n, device=device) // group_size
    mask = (gid[:, None] == gid[None, :]) & ~torch.eye(n, dtype=torch.bool, device=device)
    n_pairs = mask.sum()

    def regularizer(model) -> torch.Tensor:
        W = model.W_eff
        G = torch.einsum("mdi,mdj->mij", W, W)
        Gd = torch.diagonal(G, dim1=-2, dim2=-1)
        pen = (Gd.unsqueeze(-1) * Gd.unsqueeze(-2) - G**2) * mask
        return lam * pen.sum(dim=(-2, -1)) / n_pairs

    return regularizer


def group_alignment_stats(W: np.ndarray, survived: np.ndarray, group_size: int):
    """Mean |cos| between surviving column pairs, within vs across groups."""
    n = W.shape[-1]
    G = np.einsum("di,dj->ij", W, W)
    norms = np.sqrt(np.diag(G))
    ok = survived & (norms > 1e-6)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    cos = np.abs(G / np.outer(np.maximum(norms, 1e-12), np.maximum(norms, 1e-12)))
    gid = np.arange(n) // group_size
    pair_ok = np.outer(ok, ok) & ~np.eye(n, dtype=bool)
    within = pair_ok & (gid[:, None] == gid[None, :])
    across = pair_ok & (gid[:, None] != gid[None, :])
    w = float(cos[within].mean()) if within.any() else float("nan")
    a = float(cos[across].mean()) if across.any() else float("nan")
    return w, a


def run_config(cfg: TrainConfig, outdir, group_size: int, lambdas: list[float]) -> dict:
    I = zipf_importances(cfg.n)
    I_t = torch.tensor(I, dtype=torch.float32)
    device = resolve_device(cfg.device)
    widths = list(range(1, cfg.d_T + 1))
    grid = [d for d in widths for _ in range(cfg.n_seeds)]
    groups = [s for _ in widths for s in range(cfg.n_seeds)]

    def blocked(shape, n, alpha, gen, dev):
        return sample_features_blocked(shape, n, alpha, group_size, gen, dev)

    settings = [("iid", None, None), ("blocked", blocked, None)] + [
        (f"blocked λ={lam}", blocked,
         make_group_alignment_regularizer(cfg.n, group_size, lam, device))
        for lam in lambdas
    ]

    t0 = time.time()
    results = {}
    for name, sampler, reg in settings:
        res = train_models(grid, cfg, I_t, I_t, data_groups=groups,
                           sampler=sampler, regularizer=reg)
        shape = (len(widths), cfg.n_seeds)
        col = column_norms_sq(res["W"]).numpy().reshape(*shape, cfg.n)
        surv = survived_mask(torch.tensor(col)).numpy()
        W = res["W"].numpy().reshape(*shape, -1, cfg.n)
        align = np.array([
            [group_alignment_stats(W[w, s], surv[w, s], group_size)
             for s in range(cfg.n_seeds)]
            for w in range(len(widths))
        ])  # (W, S, 2): within, across
        results[name] = {
            "floor": res["eval_loss"].numpy().reshape(shape).tolist(),
            "n_survived": surv.sum(-1).tolist(),
            "capacity_sum": feature_capacity(res["W"]).sum(-1)
                .numpy().reshape(shape).tolist(),
            "cos_within": align[..., 0].tolist(),
            "cos_across": align[..., 1].tolist(),
        }
    runtime = time.time() - t0

    out = {
        "config": cfg.asdict(),
        "group_size": group_size,
        "widths": widths,
        "settings": [s[0] for s in settings],
        "critical_width": d_star(cfg.n, cfg.alpha),
        "predicted_floor_iid": [predicted_floor(d, cfg.n, cfg.alpha, I) for d in widths],
        "results": results,
        "runtime_s": runtime,
    }
    t = tag(cfg.n, cfg.alpha)
    dump_json(outdir / f"expC_{t}.json", out)

    plot_blocked(out, outdir / f"fig6_blocked_{t}.png")
    mid = len(widths) // 2
    floors = {k: np.array(v["floor"])[mid].mean() for k, v in results.items()}
    print(f"[expC {t}] k={group_size}  floors at d_S={widths[mid]}: "
          + "  ".join(f"{k}={v:.4f}" for k, v in floors.items())
          + f"  L*={out['predicted_floor_iid'][mid]:.4f}  ({runtime:.0f}s)",
          flush=True)
    return out


def plot_blocked(out: dict, path) -> None:
    """Fig 6: floor, survival, and group alignment on blocked data."""
    widths = np.array(out["widths"])
    cfg = out["config"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pred = np.array(out["predicted_floor_iid"])
    pos = pred > 0
    axes[0].plot(widths[pos], pred[pos], "k--", label="$L^*$ (Eq. 2, iid)")
    for i, name in enumerate(out["settings"]):
        color = CYCLE[i % len(CYCLE)]
        r = out["results"][name]
        floor = np.array(r["floor"])
        axes[0].errorbar(widths, floor.mean(1), yerr=floor.std(1), fmt="o-",
                         capsize=2, color=color, label=name)
        ns = np.array(r["n_survived"])
        axes[1].errorbar(widths, ns.mean(1), yerr=ns.std(1), fmt="o-",
                         capsize=2, color=color, label=name)
        if name != "iid":
            # Small widths can have no surviving cross-group pair -> all-NaN
            # slice; np.nanmean warns on those, so suppress and keep the NaN.
            cw = np.array(r["cos_within"])
            ca = np.array(r["cos_across"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_cw, mean_ca = np.nanmean(cw, 1), np.nanmean(ca, 1)
            axes[2].plot(widths, mean_cw, "o-", color=color,
                         label=f"{name}: within-group")
            axes[2].plot(widths, mean_ca, "o--", color=color,
                         alpha=0.5, label=f"{name}: cross-group")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("width $d_S$")
    axes[0].set_ylabel("importance-weighted MSE")
    axes[0].set_title("Floor vs iid prediction")
    axes[0].legend(fontsize=7)
    axes[1].axvline(out["critical_width"], color="gray", linestyle=":")
    axes[1].set_xlabel("width $d_S$")
    axes[1].set_ylabel("survived |S|")
    axes[1].set_title("Feature survival")
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel("width $d_S$")
    axes[2].set_ylabel("mean |cos| (surviving pairs)")
    axes[2].set_title("Packing alignment")
    axes[2].legend(fontsize=6)
    fig.suptitle(
        f"Exp C — block-anticorrelated packing (n={cfg['n']}, α={cfg['alpha']}, "
        f"groups of {out['group_size']} consecutive ranks, {cfg['n_seeds']} seeds)")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg = load_config("expC_full.yaml", __doc__)
    tc = TrainConfig(n=cfg.n, alpha=cfg.alpha, d_T=cfg.d_T, n_seeds=cfg.seeds,
                     steps=cfg.steps, seed=cfg.seed, device=cfg.device)
    run_config(tc, cfg.outdir, group_size=cfg.group_size,
               lambdas=list(cfg.lambdas))


if __name__ == "__main__":
    main()
