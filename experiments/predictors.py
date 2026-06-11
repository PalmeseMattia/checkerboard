"""Absolute-floor predictors: does the equilibrium correction fix Eq. 2?

Three predictors of the ABSOLUTE loss floor are evaluated on every
(config, width, seed) point of the Exp 0 sweep:

  (a) original Eq. 2:      F_kept = floor(d * g(alpha)), drop by importance;
  (b) equilibrium count:   F = round(d * ghat(alpha)), with ghat = a*g^b the
      canonical Zipf packing-law fit read from the slope-law probe output;
  (c) per-feature form:    sum_i I_i * E[x_i^2] * (1 - C_i), with measured
      fractional capacities C_i (diagnostic: tests whether count + partial
      representation jointly explain absolute values).

Predictor (a) underestimates floors (achieved > predicted everywhere,
worst near/above d*) exactly as the refined Sarkar-Deka formula does on
Pythia (their calibration needs C = 8.97 > 1). Reports R^2 on absolute
values, overall and stratified by distance from d*, plus the d_S = d_T
geometric/residual split. Produces the headline figure fig7
(predicted vs observed, log-log, three series, y = x line).

Requires: exp0_*.json (from exp0_replication.py, with per-seed
feature_capacity arrays) and probe_slope_law_n*_fit.json (from
probe_slope_law.py) in the output directory.

    python experiments/predictors.py --config configs/predictors_full.yaml
"""

import json
import sys

import numpy as np

from common import dump_json, load_config
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import expected_x_sq, g_alpha, zipf_importances

import matplotlib.pyplot as plt


def load_zipf_fit(outdir, fit_n: int) -> dict:
    """Canonical Zipf packing-law (a, b) from the slope-law probe output."""
    path = outdir / f"probe_slope_law_n{fit_n}_fit.json"
    if not path.exists():
        sys.exit(f"{path} not found — run probe_slope_law.py first.")
    fit = json.loads(path.read_text())
    # Slope keys are stringified YAML values ("1" or "1.0"); match by float.
    for key, val in fit["slope_fit"].items():
        if float(key) == 1.0:
            return val
    sys.exit(f"{path} has no s=1 (Zipf) fit.")


def floor_with_count(I: np.ndarray, F: int, alpha: float) -> float:
    """Eq. 2 floor keeping the top-F features by true importance."""
    n = len(I)
    if F >= n:
        return 0.0
    dropped = np.argsort(-I, kind="stable")[max(F, 0):]
    return float(I[dropped].sum() * expected_x_sq(alpha))


def floor_per_feature(I: np.ndarray, C: np.ndarray, alpha: float) -> float:
    """Per-feature form: each feature charged I_i E[x^2] (1 - C_i)."""
    C = np.clip(C, 0.0, 1.0)
    return float((I * expected_x_sq(alpha) * (1.0 - C)).sum())


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination on absolute values (can go negative)."""
    obs, pred = np.asarray(obs), np.asarray(pred)
    if obs.size < 2:
        return float("nan")
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def collect(outdir, zipf_fit: dict) -> list[dict]:
    """One row per (config, width, seed) across all Exp 0 JSONs."""
    a_fit, b_fit = zipf_fit["a"], zipf_fit["b"]
    rows = []
    for f in sorted(outdir.glob("exp0_*.json")):
        out = json.loads(f.read_text())
        cfg = out["config"]
        n, alpha = cfg["n"], cfg["alpha"]
        I = zipf_importances(n)
        g = g_alpha(alpha)
        ghat = a_fit * g ** b_fit
        if "feature_capacity" not in out:
            sys.exit(f"{f.name} lacks per-seed feature_capacity — re-run "
                     "exp0_replication.py first.")
        capC = np.array(out["feature_capacity"])  # (W, S, n)
        obs = np.array(out["achieved_floor"])  # (W, S)
        for w_i, d in enumerate(out["widths"]):
            pa = floor_with_count(I, int(np.floor(d * g)), alpha)
            pb = floor_with_count(I, int(round(d * ghat)), alpha)
            for s in range(cfg["n_seeds"]):
                rows.append({
                    "config": f"n{n}_a{alpha:.2f}", "n": n, "alpha": alpha,
                    "d": d, "seed": s, "d_star": out["critical_width"],
                    "dist": d - out["critical_width"],
                    "obs": float(obs[w_i, s]),
                    "pred_a": pa, "pred_b": pb,
                    "pred_c": floor_per_feature(I, capC[w_i, s], alpha),
                })
    return rows


def stratum(dist: float) -> str:
    if dist < -1.0:
        return "below"
    if dist > 1.0:
        return "above"
    return "near"


def analyze(cfg_ns) -> dict:
    outdir = cfg_ns.outdir
    zipf_fit = load_zipf_fit(outdir, cfg_ns.fit_n)
    rows = collect(outdir, zipf_fit)
    obs = np.array([r["obs"] for r in rows])
    preds = {k: np.array([r[f"pred_{k}"] for r in rows]) for k in "abc"}
    strata = np.array([stratum(r["dist"]) for r in rows])

    summary = {"overall": {k: r2(obs, preds[k]) for k in "abc"}}
    for s in ("below", "near", "above"):
        m = strata == s
        summary[s] = {"n_points": int(m.sum()),
                      **{k: r2(obs[m], preds[k][m]) for k in "abc"}}

    # d_S = d_T geometric/residual split. Predictor (b) is the validated
    # geometric estimate; (c) overestimates (a partially represented feature
    # plus an optimal bias recovers most of its variance), so its fraction
    # is only a loose upper bound on the geometric share.
    split = []
    for f in sorted(outdir.glob("exp0_*.json")):
        out = json.loads(f.read_text())
        cfg = out["config"]
        tag_c = f"n{cfg['n']}_a{cfg['alpha']:.2f}"
        crows = [r for r in rows
                 if r["config"] == tag_c and r["d"] == max(out["widths"])]
        L_obs = float(np.mean([r["obs"] for r in crows]))
        L_b = float(np.mean([r["pred_b"] for r in crows]))
        L_c = float(np.mean([r["pred_c"] for r in crows]))
        split.append({
            "config": tag_c, "d_T": max(out["widths"]),
            "d_star": out["critical_width"], "L_obs": L_obs,
            "L_geometric_b": L_b, "L_residual_b": L_obs - L_b,
            "geometric_fraction_b": L_b / L_obs if L_obs > 0 else float("nan"),
            "L_geometric_c": L_c,
            "geometric_fraction_c": L_c / L_obs if L_obs > 0 else float("nan"),
        })

    result = {"zipf_fit": zipf_fit, "n_points": len(rows),
              "r2_summary": summary, "dt_decomposition": split, "rows": rows}
    dump_json(outdir / "floor_predictors.json", result)
    plot_scatter(rows, outdir / "fig7_predicted_vs_observed.png")

    print(f"\n{len(rows)} points across {len({r['config'] for r in rows})} configs"
          f"  (ĝ = {zipf_fit['a']:.2f}·g^{zipf_fit['b']:.2f})")
    print("\nR^2 on ABSOLUTE floors:")
    print(f"{'stratum':<8} {'n':>4} {'(a) Eq.2':>10} {'(b) equil':>10} {'(c) per-feat':>12}")
    for s in ("overall", "below", "near", "above"):
        d = summary[s]
        npts = d.get("n_points", len(rows))
        print(f"{s:<8} {npts:>4} {d['a']:>10.3f} {d['b']:>10.3f} {d['c']:>12.3f}")
    print("\nd_S=d_T geometric/residual split (mean over seeds):")
    print(f"{'config':<12} {'L_obs':>9} {'L_geom(b)':>10} {'resid(b)':>9} "
          f"{'gfrac(b)':>9} {'gfrac(c)':>9}")
    for s in split:
        print(f"{s['config']:<12} {s['L_obs']:>9.4f} {s['L_geometric_b']:>10.4f} "
              f"{s['L_residual_b']:>9.4f} {s['geometric_fraction_b']:>9.2f} "
              f"{s['geometric_fraction_c']:>9.2f}")
    return result


def plot_scatter(rows: list[dict], path) -> None:
    """Headline fig 7: predicted vs observed absolute floor, log-log, y=x."""
    obs = np.array([r["obs"] for r in rows])
    series = [("pred_a", "(a) Eq. 2  $F_{kept}$=⌊d·g⌋", OKABE_ITO["vermillion"], "o"),
              ("pred_b", "(b) equilibrium count", OKABE_ITO["blue"], "s"),
              ("pred_c", "(c) per-feature $C_i$", OKABE_ITO["orange"], "^")]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    lo = max(obs[obs > 0].min(), 1e-5)
    hi = obs.max() * 1.5
    for key, label, color, marker in series:
        pred = np.array([r[key] for r in rows])
        m = (pred > 0) & (obs > 0)
        ax.scatter(obs[m], pred[m], s=16, alpha=0.55, color=color,
                   marker=marker, linewidths=0,
                   label=f"{label}  (R²={r2(obs, pred):.2f})")
        if m.any():
            lo = min(lo, pred[m].min())
    lim = (max(lo * 0.7, 1e-5), hi)
    ax.plot(lim, lim, "--", color=OKABE_ITO["black"], linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("observed floor (importance-weighted MSE)")
    ax.set_ylabel("predicted floor")
    ax.set_title("Absolute-floor predictors vs observed\n"
                 "(Exp 0 sweep, all configs × widths × seeds)")
    ax.legend(fontsize=8, loc="upper left")
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg_ns = load_config("predictors_full.yaml", __doc__)
    analyze(cfg_ns)


if __name__ == "__main__":
    main()
