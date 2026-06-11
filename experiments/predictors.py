"""Absolute-floor predictors: does the equilibrium correction fix Eq. 2?

Predictors of the ABSOLUTE loss floor, evaluated on every
(config, width, seed) point of the Exp 0 sweep:

  (a)  original Eq. 2:     F_kept = floor(d * g(alpha)), drop by importance;
  (b)  equilibrium count:  F = round(d * ghat(alpha)), ghat = a*g^b the
       canonical Zipf packing-law fit from the slope-law probe (1 fitted law);
       also reported with floor() instead of round() to quantify the
       convention's effect near d*;
  (b') measured kept set (ZERO fitted parameters): per run, Eq. 2 charged on
       the run's own observed survived set S_obs:
           floor' = sum_{i not in S_obs} I_i * E[x_i^2].
       This is the decisive audit: if (b') fits, the floor is exactly the
       dropped importance and the only modelling left is predicting |S|;
  (c)  per-feature form:   sum_i I_i * E[x_i^2] * (1 - C_i), measured C_i
       (diagnostic; overshoots — kept in tables, not in the headline figure).

Gap decomposition per run (separates dropped-count error from interference):
  achieved = sum_{i not in S} I_i*mse_i  (dropped part)
           + sum_{i in S}     I_i*mse_i  (kept residual = interference).
The dropped part is compared with the Eq. 2 charge E[x^2] per dropped
feature; the kept residual is what no dropped-importance formula can see.

Strata are PRE-REGISTERED in the config as fixed d/d* bands
(below: d/d* < 0.8, near: 0.8 <= d/d* <= 1.2, above: d/d* > 1.2).

Reports R^2 and MAE per stratum for each predictor, pooled and per-config
(12 units), plus the decomposition shares. Produces the headline figure
fig7: predicted-vs-observed scatter for (a), (b), (b') and the gap
decomposition vs d/d*.

Requires: exp0_*.json (with per-seed col_norms_sq / feature_capacity /
per_feature_mse) and probe_slope_law_n*_fit.json in the output directory.

    python experiments/predictors.py --config configs/predictors_full.yaml
"""

import json
import sys

import numpy as np

from common import dump_json, load_config
from src.metrics import SURVIVAL_THRESHOLD
from src.plotting import OKABE_ITO, save_fig, set_style
from src.theory import expected_x_sq, g_alpha, zipf_importances

import matplotlib.pyplot as plt

PREDICTORS = ["a", "b", "b_floor", "bp", "c"]


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


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination on absolute values (can go negative)."""
    obs, pred = np.asarray(obs), np.asarray(pred)
    if obs.size < 2:
        return float("nan")
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(obs) - np.asarray(pred))))


def collect(outdir, zipf_fit: dict) -> list[dict]:
    """One row per (config, width, seed) across all Exp 0 JSONs."""
    a_fit, b_fit = zipf_fit["a"], zipf_fit["b"]
    rows = []
    for f in sorted(outdir.glob("exp0_*.json")):
        out = json.loads(f.read_text())
        cfg = out["config"]
        n, alpha = cfg["n"], cfg["alpha"]
        I = zipf_importances(n)
        Ex2 = expected_x_sq(alpha)
        g = g_alpha(alpha)
        ghat = a_fit * g ** b_fit
        for key in ("feature_capacity", "col_norms_sq", "per_feature_mse"):
            if key not in out:
                sys.exit(f"{f.name} lacks per-seed {key} — re-run "
                         "exp0_replication.py first.")
        capC = np.array(out["feature_capacity"])    # (W, S, n)
        norms = np.array(out["col_norms_sq"])       # (W, S, n)
        mse = np.array(out["per_feature_mse"])      # (W, S, n)
        obs = np.array(out["achieved_floor"])       # (W, S)
        for w_i, d in enumerate(out["widths"]):
            pa = floor_with_count(I, int(np.floor(d * g)), alpha)
            pb = floor_with_count(I, int(round(d * ghat)), alpha)
            pbf = floor_with_count(I, int(np.floor(d * ghat)), alpha)
            for s in range(cfg["n_seeds"]):
                S_obs = norms[w_i, s] > SURVIVAL_THRESHOLD
                wmse = I * mse[w_i, s]
                rows.append({
                    "config": f"n{n}_a{alpha:.2f}", "n": n, "alpha": alpha,
                    "d": d, "seed": s, "d_star": out["critical_width"],
                    "ratio": d / out["critical_width"],
                    "obs": float(obs[w_i, s]),
                    "pred_a": pa, "pred_b": pb, "pred_b_floor": pbf,
                    # (b'): zero-parameter, this run's own kept set.
                    "pred_bp": float(I[~S_obs].sum() * Ex2),
                    "pred_c": float((I * Ex2 * (1 - np.clip(capC[w_i, s], 0, 1))).sum()),
                    # Gap decomposition (importance-weighted, from measured MSE).
                    "dropped_part": float(wmse[~S_obs].sum()),
                    "kept_part": float(wmse[S_obs].sum()),
                    "n_kept": int(S_obs.sum()),
                })
    return rows


def analyze(cfg_ns) -> dict:
    outdir = cfg_ns.outdir
    zipf_fit = load_zipf_fit(outdir, cfg_ns.fit_n)
    rows = collect(outdir, zipf_fit)
    below_max = float(cfg_ns.strata["below_max"])
    near_max = float(cfg_ns.strata["near_max"])

    def stratum(ratio: float) -> str:
        if ratio < below_max:
            return "below"
        if ratio > near_max:
            return "above"
        return "near"

    obs = np.array([r["obs"] for r in rows])
    preds = {k: np.array([r[f"pred_{k}"] for r in rows]) for k in PREDICTORS}
    strata = np.array([stratum(r["ratio"]) for r in rows])
    configs = np.array([r["config"] for r in rows])

    summary = {"overall": {**{k: r2(obs, preds[k]) for k in PREDICTORS},
                           **{f"mae_{k}": mae(obs, preds[k]) for k in PREDICTORS}}}
    for s in ("below", "near", "above"):
        m = strata == s
        summary[s] = {"n_points": int(m.sum()),
                      **{k: r2(obs[m], preds[k][m]) for k in PREDICTORS},
                      **{f"mae_{k}": mae(obs[m], preds[k][m]) for k in PREDICTORS}}

    # Per-config aggregated R^2 (12 units) alongside the pooled values.
    per_config = {}
    for c in sorted(set(configs)):
        m = configs == c
        per_config[c] = {k: r2(obs[m], preds[k][m]) for k in PREDICTORS}
    config_agg = {k: {"mean": float(np.nanmean([v[k] for v in per_config.values()])),
                      "median": float(np.nanmedian([v[k] for v in per_config.values()]))}
                  for k in PREDICTORS}

    # Gap decomposition per stratum: shares of the achieved floor, and the
    # measured dropped cost vs the Eq. 2 per-feature charge.
    decomposition = {}
    for s in ("overall", "below", "near", "above"):
        m = np.ones(len(rows), bool) if s == "overall" else strata == s
        dropped = np.array([r["dropped_part"] for r in rows])[m]
        kept = np.array([r["kept_part"] for r in rows])[m]
        pbp = preds["bp"][m]
        tot = np.maximum(dropped + kept, 1e-12)
        ok = pbp > 1e-9
        decomposition[s] = {
            "dropped_share": float(np.mean(dropped / tot)),
            "kept_residual_share": float(np.mean(kept / tot)),
            "dropped_vs_eq2_charge": float(np.mean(dropped[ok] / pbp[ok]))
            if ok.any() else float("nan"),
        }

    # d_S = d_T geometric/residual split, by the zero-parameter (b').
    split = []
    for f in sorted(outdir.glob("exp0_*.json")):
        out = json.loads(f.read_text())
        cfg = out["config"]
        tag_c = f"n{cfg['n']}_a{cfg['alpha']:.2f}"
        crows = [r for r in rows
                 if r["config"] == tag_c and r["d"] == max(out["widths"])]
        L_obs = float(np.mean([r["obs"] for r in crows]))
        L_b = float(np.mean([r["pred_b"] for r in crows]))
        L_bp = float(np.mean([r["pred_bp"] for r in crows]))
        L_c = float(np.mean([r["pred_c"] for r in crows]))
        split.append({
            "config": tag_c, "d_T": max(out["widths"]),
            "d_star": out["critical_width"], "L_obs": L_obs,
            "L_geometric_b": L_b, "L_residual_b": L_obs - L_b,
            "geometric_fraction_b": L_b / L_obs if L_obs > 0 else float("nan"),
            "L_geometric_bp": L_bp,
            "geometric_fraction_bp": L_bp / L_obs if L_obs > 0 else float("nan"),
            "L_geometric_c": L_c,
            "geometric_fraction_c": L_c / L_obs if L_obs > 0 else float("nan"),
        })

    result = {"zipf_fit": zipf_fit, "n_points": len(rows),
              "strata_definition": {"below": f"d/d* < {below_max}",
                                    "near": f"{below_max} <= d/d* <= {near_max}",
                                    "above": f"d/d* > {near_max}"},
              "r2_summary": summary, "per_config_r2": per_config,
              "per_config_aggregate": config_agg,
              "decomposition": decomposition,
              "dt_decomposition": split, "rows": rows}
    dump_json(outdir / "floor_predictors.json", result)
    plot_headline(rows, summary, outdir / "fig7_predicted_vs_observed.png")

    print(f"\n{len(rows)} points across {len(per_config)} configs "
          f"(ĝ = {zipf_fit['a']:.2f}·g^{zipf_fit['b']:.2f}; strata: "
          f"below<{below_max}, near {below_max}-{near_max}, above>{near_max})")
    print("\nR^2 (MAE) on ABSOLUTE floors:")
    hdr = f"{'stratum':<8} {'n':>4}"
    for k in PREDICTORS:
        hdr += f" {('(' + k + ')'):>16}"
    print(hdr)
    for s in ("overall", "below", "near", "above"):
        d = summary[s]
        npts = d.get("n_points", len(rows))
        line = f"{s:<8} {npts:>4}"
        for k in PREDICTORS:
            line += f" {d[k]:>7.3f} ({d[f'mae_{k}']:.4f})"
        print(line)
    print("\nper-config aggregated R^2 (12 units):")
    for k in PREDICTORS:
        print(f"  ({k}): mean={config_agg[k]['mean']:.3f} "
              f"median={config_agg[k]['median']:.3f}")
    print("\ngap decomposition (mean shares of achieved floor):")
    for s in ("overall", "below", "near", "above"):
        dd = decomposition[s]
        print(f"  {s:<8} dropped={dd['dropped_share']:.2f} "
              f"kept-residual={dd['kept_residual_share']:.2f} "
              f"dropped/Eq2-charge={dd['dropped_vs_eq2_charge']:.2f}")
    return result


def plot_headline(rows: list[dict], summary: dict, path) -> None:
    """Fig 7 (headline): (a)/(b)/(b') scatter + gap decomposition vs d/d*."""
    obs = np.array([r["obs"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    series = [("pred_a", "(a) Eq. 2  F=⌊d·g⌋", OKABE_ITO["vermillion"], "o"),
              ("pred_b", "(b) equilibrium count", OKABE_ITO["blue"], "s"),
              ("pred_bp", "(b′) measured kept set (0 params)", OKABE_ITO["green"], "D")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    lo = max(obs[obs > 0].min(), 1e-5)
    hi = obs.max() * 1.5
    for key, label, color, marker in series:
        pred = np.array([r[key] for r in rows])
        m = (pred > 0) & (obs > 0)
        ax.scatter(obs[m], pred[m], s=14, alpha=0.5, color=color,
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

    ax = axes[1]
    tot = np.array([r["dropped_part"] + r["kept_part"] for r in rows])
    tot = np.maximum(tot, 1e-12)
    kept_share = np.array([r["kept_part"] for r in rows]) / tot
    ax.scatter(ratio, kept_share, s=12, alpha=0.45, linewidths=0,
               color=OKABE_ITO["orange"], label="kept-feature residual (interference)")
    ax.scatter(ratio, 1 - kept_share, s=12, alpha=0.45, linewidths=0,
               color=OKABE_ITO["purple"], label="dropped-feature importance")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="d = d*")
    ax.set_xlabel("d / d*")
    ax.set_ylabel("share of achieved floor")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Where the loss lives:\ndropped importance vs kept-feature residual")
    ax.legend(fontsize=8)
    save_fig(fig, path)


def main() -> None:
    set_style()
    cfg_ns = load_config("predictors_full.yaml", __doc__)
    analyze(cfg_ns)


if __name__ == "__main__":
    main()
