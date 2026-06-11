"""Slope-law probe: packing law ghat(alpha, s) and its threshold robustness.

THE canonical instrument for every packing-law number in this repository
(fit protocol documented in src/theory.py:fit_packing_law). Trains packing
models at fixed (n, d) over a grid of importance power laws I_i ∝ i^(−s)
(s=0 ≡ uniform, s=1 ≡ Zipf) and sparsities alpha, saving per-feature
column norms ‖W_i‖² and fractional capacities C_i so kept counts can be
recomputed at any threshold after the fact. Outputs:

- per-alpha JSONs with per-feature arrays;
- `probe_slope_law_n<n>_fit.json`: canonical fits ghat = a·g^b per slope at
  tau=0.5, plus the threshold-robustness table (tau in {0.3..0.7}, norm²
  and C_i variants);
- fig8: packing law vs slope (left) and fitted exponent b vs slope (right).

The n=400 convergence-check configs (1x and 3x training at alpha=0.99,
uniform) reuse this script; their comparison table is assembled by
scripts/make_report.py.

    python experiments/probe_slope_law.py --config configs/probe_slope_law_full.yaml
"""

import json
import time

import numpy as np
import torch

from common import dump_json, equalized_schedule, load_config
from src.metrics import SURVIVAL_THRESHOLD, column_norms_sq, feature_capacity
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import fit_packing_law, g_alpha, power_law_importances
from src.train import TrainConfig, train_models

import matplotlib.pyplot as plt

NORM_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
CAP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def run_alpha(cfg_ns, alpha: float) -> dict:
    if cfg_ns.equalize_active:
        batch, steps = equalized_schedule(alpha, cfg_ns.equalize_active)
    else:
        batch, steps = 1024, cfg_ns.steps
    cfg = TrainConfig(n=cfg_ns.n, alpha=alpha, steps=steps, batch_size=batch,
                      seed=cfg_ns.seed, eval_every=max(steps // 5, 500),
                      device=cfg_ns.device)

    grid, groups, I_rows, labels = [], [], [], []
    for s in cfg_ns.slopes:
        for seed in range(cfg_ns.seeds):
            grid.append(cfg_ns.width)
            groups.append(seed)
            I_rows.append(power_law_importances(cfg_ns.n, s))
            labels.append({"slope": s, "seed": seed})
    I_train = torch.tensor(np.stack(I_rows), dtype=torch.float32)

    t0 = time.time()
    res = train_models(grid, cfg, I_train, I_train[0], data_groups=groups)
    norms = column_norms_sq(res["W"]).numpy()  # (M, n)
    caps = feature_capacity(res["W"]).numpy()  # (M, n)

    records = [
        {**lab, "alpha": alpha,
         "col_norms_sq": norms[m].tolist(),
         "feature_capacity": caps[m].tolist(),
         "capacity_sum": float(caps[m].sum()),
         "eval_loss": float(res["eval_loss"][m])}
        for m, lab in enumerate(labels)
    ]
    out = {"n": cfg_ns.n, "width": cfg_ns.width, "alpha": alpha,
           "slopes": list(cfg_ns.slopes), "n_seeds": cfg_ns.seeds,
           "batch_size": batch, "steps": steps,
           "active_samples_per_feature": batch * steps * (1 - alpha),
           "config": cfg.asdict(), "runtime_s": time.time() - t0,
           "records": records}
    path = cfg_ns.outdir / f"probe_slope_law_n{cfg_ns.n}_a{alpha:.2f}.json"
    path.write_text(json.dumps(out))
    print(f"[slope_law] n={cfg_ns.n} α={alpha} batch={batch} steps={steps} "
          f"({out['runtime_s']:.0f}s)", flush=True)
    return out


def aggregate(cfg_ns) -> None:
    runs = [json.loads(p.read_text()) for p in
            sorted(cfg_ns.outdir.glob(f"probe_slope_law_n{cfg_ns.n}_a*.json"))
            if not p.stem.endswith("_fit")]
    d = cfg_ns.width
    alphas = sorted({r["alpha"] for r in runs})
    g = np.array([g_alpha(a) for a in alphas])

    def mean_survived(slope: float, by: str, thr: float) -> np.ndarray:
        """Mean kept count per alpha for one slope, thresholding `by`."""
        key = "col_norms_sq" if by == "norm" else "feature_capacity"
        out = []
        for a in alphas:
            run = next(r for r in runs if r["alpha"] == a)
            counts = [int((np.array(rec[key]) > thr).sum())
                      for rec in run["records"] if rec["slope"] == slope]
            out.append(np.mean(counts) if counts else np.nan)
        return np.array(out)

    # Threshold robustness of the uniform / Zipf exponents.
    thr_table = {"norm": {}, "cap": {}}
    for by, thrs in (("norm", NORM_THRESHOLDS), ("cap", CAP_THRESHOLDS)):
        for thr in thrs:
            row = {}
            for name, slope in (("uniform", 0.0), ("zipf", 1.0)):
                if slope not in cfg_ns.slopes:
                    continue
                row[name] = fit_packing_law(g, mean_survived(slope, by, thr) / d)
            thr_table[by][f"{thr:.1f}"] = row

    # Canonical fit ghat(alpha, s) at the survival threshold (norm² > 0.5),
    # with seed-bootstrap standard errors on the prefactor and exponent
    # (resample seeds with replacement per alpha, refit; B=1000, fixed RNG).
    def per_seed_counts(slope: float) -> dict[float, np.ndarray]:
        out = {}
        for a in alphas:
            run = next(r for r in runs if r["alpha"] == a)
            out[a] = np.array(
                [int((np.array(rec["col_norms_sq"]) > SURVIVAL_THRESHOLD).sum())
                 for rec in run["records"] if rec["slope"] == slope], dtype=float)
        return out

    rng = np.random.default_rng(0)
    B = 1000
    slope_fit, surv_by_slope = {}, {}
    for slope in cfg_ns.slopes:
        counts_by_alpha = per_seed_counts(slope)
        counts = np.array([counts_by_alpha[a].mean() for a in alphas])
        surv_by_slope[f"{slope}"] = counts.tolist()
        fit = fit_packing_law(g, counts / d)
        boot_a, boot_b = [], []
        if len(alphas) >= 2:
            for _ in range(B):
                ghat = np.array([
                    rng.choice(c, size=len(c), replace=True).mean()
                    for c in (counts_by_alpha[a] for a in alphas)
                ]) / d
                bf = fit_packing_law(g, ghat)
                boot_a.append(bf["a"])
                boot_b.append(bf["b"])
        fit["a_se"] = float(np.nanstd(boot_a)) if boot_a else float("nan")
        fit["b_se"] = float(np.nanstd(boot_b)) if boot_b else float("nan")
        slope_fit[f"{slope}"] = fit

    fit = {"n": cfg_ns.n, "width": d, "alphas": alphas, "g": g.tolist(),
           "slopes": list(cfg_ns.slopes),
           "threshold_robustness": thr_table,
           "slope_fit": slope_fit,
           "survived_by_slope_tau0.5": surv_by_slope}
    dump_json(cfg_ns.outdir / f"probe_slope_law_n{cfg_ns.n}_fit.json", fit)

    print(f"\ncanonical fits at τ={SURVIVAL_THRESHOLD} (n={cfg_ns.n}, d={d}, "
          f"seed-bootstrap SE, B={B}):")
    for slope in cfg_ns.slopes:
        f_ = slope_fit[f"{slope}"]
        print(f"  s={slope}: ĝ = {f_['a']:.2f}(±{f_['a_se']:.2f})·"
              f"g^{f_['b']:.2f}(±{f_['b_se']:.2f})")

    if len(alphas) >= 2:
        plot_slope_law(fit, cfg_ns.outdir / f"fig8_slope_law_n{cfg_ns.n}.png")
        plot_norm_hist(runs, alphas,
                       cfg_ns.outdir / f"fig9_norm_hist_n{cfg_ns.n}.png")


def plot_slope_law(fit: dict, path) -> None:
    alphas = fit["alphas"]
    g = np.array(fit["g"])
    d = fit["width"]
    slopes = fit["slopes"]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, slope in enumerate(slopes):
        color = cmap(i / max(len(slopes) - 1, 1))
        surv = np.array(fit["survived_by_slope_tau0.5"][f"{slope}"]) / d
        f_ = fit["slope_fit"][f"{slope}"]
        axes[0].plot(g, surv, "o", color=color,
                     label=f"s={slope}: ĝ={f_['a']:.2f}·g^{f_['b']:.2f}")
        if np.isfinite(f_["b"]):
            axes[0].plot(g, f_["a"] * g ** f_["b"], "-", color=color, alpha=0.4)
    axes[0].plot(g, g, "--", color=OKABE_ITO["black"], alpha=0.5,
                 label="g(α) (bound)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("g(α)")
    axes[0].set_ylabel(f"achieved features/dim (d={d})")
    axes[0].set_title("Packing law vs importance slope s")
    axes[0].legend(fontsize=7)

    b_vals = [fit["slope_fit"][f"{s}"]["b"] for s in slopes]
    axes[1].plot(slopes, b_vals, "o-", color=OKABE_ITO["purple"])
    axes[1].axhline(1.0, color="gray", linestyle=":",
                    label="b=1 (proportional to g)")
    axes[1].set_xlabel("importance slope s (I ∝ i^−s)")
    axes[1].set_ylabel("fitted exponent b")
    axes[1].set_title("Capacity-scaling exponent vs slope")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Importance-slope packing law (n={fit['n']}, τ=0.5, "
                 f"α grid {alphas})")
    save_fig(fig, path)


def plot_norm_hist(runs: list, alphas: list, path) -> None:
    """Fig 9: distribution of ‖W_i‖² per alpha (all slopes & seeds pooled).

    Shows the bimodality that makes the kept-count exponents flat across
    tau in [0.3, 0.7]: column norms concentrate near 0 (dropped) and near
    or above 1 (kept), leaving the threshold window nearly empty.
    """
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.2 * len(alphas), 3.4),
                             sharey=True)
    if len(alphas) == 1:
        axes = [axes]
    for ax, a in zip(axes, alphas):
        run = next(r for r in runs if r["alpha"] == a)
        norms = np.concatenate(
            [np.array(rec["col_norms_sq"]) for rec in run["records"]])
        clipped = np.minimum(norms, 2.0)  # >2 folded into the last bin
        in_band = float(((norms > 0.3) & (norms < 0.7)).mean())
        ax.hist(clipped, bins=40, range=(0, 2), color="#0072B2", alpha=0.85)
        ax.axvspan(0.3, 0.7, color="#D55E00", alpha=0.18,
                   label=f"τ band [0.3, 0.7]: {in_band:.1%} of mass")
        ax.set_yscale("log")
        ax.set_xlabel("‖W_i‖²  (clipped at 2)")
        ax.set_title(f"α = {a}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count (log)")
    fig.suptitle("Column-norm bimodality: why kept counts are τ-stable "
                 "(all slopes & seeds pooled)")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg_ns = load_config("probe_slope_law_full.yaml", __doc__)
    if not cfg_ns.aggregate_only:
        for alpha in cfg_ns.alphas:
            run_alpha(cfg_ns, alpha)
    aggregate(cfg_ns)


if __name__ == "__main__":
    main()
