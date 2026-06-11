"""Capacity-gap probe: compressed-sensing bound vs achieved ReLU-decoder packing.

Measures the gap between the capacity bound g(alpha) of Sarkar & Deka and
the packing actually achieved by trained ReLU decoders, as a function of
(a) the importance distribution (Zipf vs uniform) and (b) sparsity alpha.
Discriminators:

1. Uniform importances: if packing rises toward g(alpha)/dim, the
   shortfall under Zipf is the importance equilibrium (tail features not
   worth their interference cost), not decoder capacity.
2. Total fractional capacity sum_i C_i (Scherlis et al.): if
   sum_i C_i ~ d while the surviving-feature count is low, the gap is a
   loss-equilibrium allocation choice, not unused capacity.

The canonical packing-law FIT lives in experiments/probe_slope_law.py
(see src/theory.py:fit_packing_law for the protocol); this probe reports
the gap table and capacity-utilization figure.

    python experiments/probe_capacity.py --config configs/probe_capacity_full.yaml
"""

import time

import numpy as np
import torch

from common import dump_json, equalized_schedule, load_config
from src.metrics import column_norms_sq, feature_capacity, survived_mask
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import g_alpha, zipf_importances
from src.train import TrainConfig, train_models

import matplotlib.pyplot as plt

IMPORTANCE_KINDS = ["zipf", "uniform"]


def run_alpha(cfg_ns, alpha: float) -> dict:
    """One batched run: (widths x importance-kinds x seeds) models at this alpha."""
    if cfg_ns.equalize_active:
        batch, steps = equalized_schedule(alpha, cfg_ns.equalize_active)
    else:
        batch, steps = 1024, cfg_ns.steps
    cfg = TrainConfig(n=cfg_ns.n, alpha=alpha, steps=steps, batch_size=batch,
                      seed=cfg_ns.seed, eval_every=max(steps // 10, 500),
                      device=cfg_ns.device)
    I_by_kind = {"zipf": zipf_importances(cfg_ns.n), "uniform": np.ones(cfg_ns.n)}

    grid, groups, I_rows, labels = [], [], [], []
    for d in cfg_ns.widths:
        for kind in IMPORTANCE_KINDS:
            for s in range(cfg_ns.seeds):
                grid.append(d)
                groups.append(s)  # same data stream per seed across settings
                I_rows.append(I_by_kind[kind])
                labels.append({"d": d, "importance": kind, "seed": s})
    I_train = torch.tensor(np.stack(I_rows), dtype=torch.float32)

    t0 = time.time()
    res = train_models(grid, cfg, I_train, I_train[0], data_groups=groups)
    n_survived = survived_mask(column_norms_sq(res["W"])).sum(-1)  # (M,)
    cap_sum = feature_capacity(res["W"]).sum(-1)  # (M,)

    records = [
        {**lab, "alpha": alpha, "n_survived": int(n_survived[m]),
         "capacity_sum": float(cap_sum[m]), "eval_loss": float(res["eval_loss"][m])}
        for m, lab in enumerate(labels)
    ]
    out = {
        "n": cfg_ns.n, "alpha": alpha, "widths": list(cfg_ns.widths),
        "n_seeds": cfg_ns.seeds, "batch_size": batch, "steps": steps,
        "active_samples_per_feature": batch * steps * (1 - alpha),
        "config": cfg.asdict(), "runtime_s": time.time() - t0,
        "records": records,
    }
    path = cfg_ns.outdir / f"probe_capacity_n{cfg_ns.n}_a{alpha:.2f}.json"
    dump_json(path, out)
    print(f"[probe_capacity] n={cfg_ns.n} alpha={alpha} batch={batch} "
          f"steps={steps} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def aggregate(cfg_ns) -> None:
    """Combine per-alpha JSONs for this n: gap table + utilization figure."""
    import json
    runs = [json.loads(p.read_text()) for p in
            sorted(cfg_ns.outdir.glob(f"probe_capacity_n{cfg_ns.n}_a*.json"))
            if not p.stem.endswith("_summary")]
    d = max(cfg_ns.widths)
    alphas = [r["alpha"] for r in runs]
    g = np.array([g_alpha(a) for a in alphas])

    stats = {}  # f"{kind}_{key}" -> per-alpha means at width d
    for kind in IMPORTANCE_KINDS:
        for key in ("n_survived", "capacity_sum"):
            stats[f"{kind}_{key}"] = [
                float(np.mean([rec[key] for rec in r["records"]
                               if rec["d"] == d and rec["importance"] == kind]))
                for r in runs
            ]

    print(f"\nat d={d}, n={cfg_ns.n} (mean over seeds):")
    print("alpha  g(α)   | zipf: surv/d ΣC/d | uniform: surv/d ΣC/d | bound min(g, n/d)")
    for i, a in enumerate(alphas):
        print(f"{a:.2f}  {g[i]:6.2f} |"
              f"  {stats['zipf_n_survived'][i] / d:5.2f}  "
              f"{stats['zipf_capacity_sum'][i] / d:.3f}  |"
              f"   {stats['uniform_n_survived'][i] / d:5.2f}  "
              f"{stats['uniform_capacity_sum'][i] / d:.3f}  |"
              f"  {min(g_alpha(a), cfg_ns.n / d):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, color in zip(IMPORTANCE_KINDS,
                           (OKABE_ITO["blue"], OKABE_ITO["orange"])):
        g_hat = np.array(stats[f"{kind}_n_survived"]) / d
        axes[0].plot(g, g_hat, "o-", color=color, label=f"{kind} importances")
        axes[1].plot(alphas, np.array(stats[f"{kind}_capacity_sum"]) / d, "o-",
                     color=color, label=f"{kind} importances")
    axes[0].plot(g, np.minimum(g, cfg_ns.n / d), "k--",
                 label="bound: min(g(α), n/d)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("g(α)")
    axes[0].set_ylabel(f"achieved features/dim at d={d}")
    axes[0].set_title("Achieved packing vs capacity bound")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="k", linestyle="--", label="saturation Σ$C_i$ = d")
    axes[1].set_xlabel("sparsity α")
    axes[1].set_ylabel(f"Σ$C_i$ / d at d={d}")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Decoder capacity utilization (fractional $C_i$)")
    axes[1].legend()
    fig.suptitle(f"Capacity-bound gap probe (n={cfg_ns.n})")
    save_fig(fig, cfg_ns.outdir / f"fig_probe_capacity_n{cfg_ns.n}.png")

    dump_json(cfg_ns.outdir / f"probe_capacity_n{cfg_ns.n}_summary.json",
              {"d": d, "n": cfg_ns.n, "alphas": alphas, "g": g.tolist(),
               "stats": stats})


def main() -> None:
    set_style()
    cfg_ns = load_config("probe_capacity_full.yaml", __doc__)
    for alpha in cfg_ns.alphas:
        run_alpha(cfg_ns, alpha)
    aggregate(cfg_ns)


if __name__ == "__main__":
    main()
