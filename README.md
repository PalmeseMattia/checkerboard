# Controlled Feature Placement in Distillation

A small, fully reproducible PyTorch testbed that audits the capacity
allocation of **superposition** and the minimum-width loss-floor theorem of
**Sarkar & Deka 2026** (*Geometric Limits of Knowledge Distillation*,
arXiv:2604.04037). Built on the **Elhage et al. 2022** toy model of
superposition and the **Scherlis et al. 2022** notion of fractional
capacity.

The repository is designed so that **every number and figure in the
findings regenerates from a clean clone** with documented commands —
reproducibility is the product, not a side effect.

## Contents

- [Summary of findings](#summary-of-findings)
- [Background: the model and the theory we audit](#background-the-model-and-the-theory-we-audit)
- [Claims → evidence map](#claims--evidence-map)
- [Results in detail](#results-in-detail)
- [Quickstart](#quickstart)
- [Full reproduction](#full-reproduction)
- [Repository layout](#repository-layout)
- [Relationship to prior work](#relationship-to-prior-work)
- [Limitations](#limitations)
- [AI-assistance disclosure](#ai-assistance-disclosure)
- [License & citation](#license--citation)

---

## Summary of findings

The refined Eq. 2 loss floor of Sarkar & Deka tracks achieved floors well
*below* the critical width d* (Pearson r ≈ 0.98 across a 12-config sweep)
but **systematically underestimates absolute floors near and above d*** —
exactly the regime their Pythia calibration patches with a constant
C = 8.97 > 1. The cause is their **assumption A2** (a width-d student keeps
exactly the top ⌊d·g(α)⌋ features by importance): trained students instead
settle at a *superposition equilibrium* that keeps **fewer, cleaner**
features, so more importance is dropped than Eq. 2 charges.

Replacing the kept count with an empirically fitted equilibrium count,
**F = round(d·ĝ(α)) with ĝ(α) = 1.03·g(α)^0.53**, flips the absolute-floor
R² from negative to positive in the near/above-d* strata (near:
−3.2 → 0.70; above: −2.9 → 0.58). The packing exponent is
importance-slope dependent (b falls from 0.93 for uniform importances to
0.34 for steep power laws), so any recalibration of their C/d*/B must be
slope-specific. Vanilla distillation respects A2's *ordering* about as well
as direct training but adds a hard *menu* constraint (students keep only
features their teacher represents); controlled placement can override
emergent allocation at a cost predictable from the theory, until the
teacher menu binds. On block-anticorrelated data, emergent superposition
already beats the iid floor — the theory's iid assumption is conservative.

> **Scope.** This is a toy-model study (single ReLU decoder family,
> synthetic sparse data). It is an instrument for auditing the theory's
> *assumptions*, not a claim about trained LLMs. See [Limitations](#limitations).

---

## Background: the model and the theory we audit

**Data.** Each input x ∈ ℝⁿ has sparse features: independently per
coordinate, xᵢ ~ U[0,1] with probability (1−α) and xᵢ = 0 otherwise. α is
the **sparsity**. Feature importances are a power law Iᵢ ∝ i^(−s); s = 1 is
the Zipf default, s = 0 is uniform.

**Model** (Elhage toy model, tied weights). For W ∈ ℝ^{d×n} and bias b ∈ ℝⁿ:

```
h = W x                         # project n features into d < n dimensions
x̂ = ReLU(Wᵀ h + b) = ReLU(WᵀW x + b)
```

The width d is the hidden bottleneck; when d < n the model must place
multiple features in shared directions (superposition).

**Loss.** Importance-weighted MSE, L = E[ Σᵢ Iᵢ (x̂ᵢ − xᵢ)² ].

**The theory we audit** (Sarkar & Deka 2026):

| quantity | formula | code |
|---|---|---|
| capacity | g(α) = 1 / ((1−α)·ln(1/(1−α))) | `theory.g_alpha` |
| critical width | d* = n / g(α) | `theory.d_star` |
| kept count | F = ⌊d·g(α)⌋ | `theory.F_kept` |
| loss floor (Eq. 2) | L*(d) = Σ_{i>F} I_(i)·E[xᵢ²], E[xᵢ²] = (1−α)/3 | `theory.predicted_floor` |
| **assumption A2** | the student keeps exactly the top-F features by importance | — (audited) |

**Fractional capacity** (Scherlis et al. 2022), a threshold-free measure of
how much of the decoder each feature occupies, with Σᵢ Cᵢ ≤ d:

```
Cᵢ = ‖Wᵢ‖⁴ / Σⱼ (Wᵢ·Wⱼ)²            #  1 = own direction, 1/2 = antipodal pair, 0 = dead
```

`theory.fit_packing_law` documents the single protocol used for every
empirical packing-law number ĝ(α) = a·g(α)^b in this repo.

---

## Claims → evidence map

Each claim regenerates from a clean clone with the listed command
(`results/report.md` collects every number and figure).

| # | Claim | Experiment | Figure | Command |
|---|-------|-----------|--------|---------|
| 1 | Eq. 2 floor tracks achieved floors (r > 0.9) but underestimates magnitude | Exp 0 | `fig1_floor_*`, `fig2_survival_*` | `exp0_replication.py --config configs/exp0_full.yaml` |
| 2 | Eq. 2 R² goes **negative** near/above d*; the equilibrium count restores it | predictors | **`fig7_predicted_vs_observed.png`** | `predictors.py --config configs/predictors_full.yaml` |
| 3 | At d_S=d_T ~63% of the floor is geometric → baseline B is contaminated | predictors (task 2) | `fig7` | same as #2 |
| 4 | g(α) bound is reachable with uniform importances; Zipf equilibrates below it | capacity probe | `fig_probe_capacity_n200.png` | `probe_capacity.py --config configs/probe_capacity_full.yaml` |
| 5 | Packing exponent b is threshold-stable and slope-dependent (0.93 → 0.34) | slope-law probe | `fig8_slope_law_n200.png` | `probe_slope_law.py --config configs/probe_slope_law_full.yaml` |
| 6 | α=0.99 uniform sub-g packing is a real equilibrium, not under-training | convergence | (table in report) | `probe_slope_law.py --config configs/convergence_n400_{1x,3x}.yaml` |
| 7 | Distillation respects A2 ordering ~ as well as direct, but adds a teacher menu | Exp A | `fig3_overlap_*` | `expA_audit.py --config configs/expA_full.yaml` |
| 8 | Placement overrides emergent allocation at a theory-predictable cost | Exp B | `fig4_pareto_*`, `fig5_perfeature_*` | `expB_placement.py --config configs/expB_full.yaml` |
| 9 | Emergent superposition already exploits anti-correlation (beats iid L*) | Exp C | `fig6_blocked_*` | `expC_correlation.py --config configs/expC_full.yaml` |

The headline figure is **`results/fig7_predicted_vs_observed.png`** (claim 2).

---

## Results in detail

Numbers below are at the default configs (`results/report.md` has the full
tables, error bars, and every (n, α)).

**Exp 0 — replication.** Across the 12-config sweep, mean Pearson r between
predicted and achieved floors is **0.98** (n=20/α=0.99 is degenerate —
d* < 1 so L* ≡ 0 — and is footnoted out). But achieved > predicted
everywhere, and the gap widens toward d*.

**Floor predictors (the core result).** Absolute-floor R² on 600 points
(12 configs × widths × seeds), stratified by distance from d*:

| stratum | n | (a) Eq. 2 | (b) equilibrium count | (c) per-feature Cᵢ |
|---|--:|--:|--:|--:|
| overall | 600 | 0.52 | **0.97** | −0.53 |
| below d* | 425 | 0.49 | 0.97 | −0.79 |
| near d* | 115 | **−3.23** | **0.70** | −7.52 |
| above d* | 60 | **−2.94** | **0.58** | −8.65 |

Eq. 2 (a) is adequate below d* (where 425/600 points sit — this is what
carries the paper's >93% median-accuracy claim) and collapses near/above;
the equilibrium count (b) restores a positive fit. The per-feature form (c)
*overshoots* — a partially represented feature plus an optimal bias recovers
most of its variance, so fractional capacity does **not** map linearly to
loss (a useful negative result). At d_S = d_T the geometric (superposition)
share of the observed floor is **0.40–0.73** (mean ≈ 0.63), so the
"architectural baseline" B read at d_S = d_T is contaminated by
superposition cost.

**Packing law (capacity & slope probes).** The g(α) bound is nearly
reachable with uniform importances (ĝ = 0.93·g^0.93) but Zipf equilibrates
below it (ĝ = 1.03·g^0.53), with Σ Cᵢ/d ≈ 1.0 in both cases — capacity is
*fully used*, just allocated to fewer, cleaner features. The exponent b is
flat across survival thresholds τ ∈ [0.3, 0.7] but falls monotonically with
importance steepness (0.93, 0.69, 0.53, 0.44, 0.34, 0.34 for
s = 0, 0.5, 1, 1.5, 2, 3). At α=0.99, tripling the training budget at n=400
does not raise the uniform packing fraction (16.40 → 15.63 features/dim,
−3%), so the sub-g packing is a real equilibrium, not slow symmetry-breaking.

**Exp A — distillation allocation audit.** Distilled and direct students
violate A2's ordering by the same small margin (overlap@k ∈ [0.80, 0.97],
never 1.0), so distillation is not the culprit. But **teacher containment is
1.000 at every width** — a distilled student keeps only features its teacher
represents (the teacher's |S_T| ≈ 22/40 menu).

**Exp B — controlled placement.** Boosting the training weight of a
low-importance critical set C = ranks {20, 28, 36} by β makes those features
survive. Under the task-target control, survival reaches 100% (β=10 at
d_S=3,5; β=3 at d_S=7). The floor cost is **small in absolute terms** —
ΔL ≈ 0.002–0.010 importance-weighted MSE at the smallest β reaching ≥90%
survival — and ordered as the theory predicts (monotone in both β and
survival). We report it as an absolute cost, not a ratio: the ratio to the
Eq. 2 placement prediction is the same order of magnitude at moderate β
(d_S=3, β=10: achieved 0.010 vs predicted 0.005, ≈2×) but **degrades to
10–50× at larger d_S where the Eq. 2 baseline itself approaches zero**, so
the ratio is uninformative there — see the full ach/Eq.2 table in
`results/report.md`. Under true distillation, survival **saturates at 0.33**
for any β — the fraction of C the teacher itself carries — confirming the
Exp A menu constraint is the binding failure mode.

**Exp C — correlation-aware packing.** On block-anticorrelated data
(mutually exclusive feature groups, identical per-feature marginals so L* is
unchanged), the achieved floor is **below the iid L\* at 10/10 widths** (mean
ratio 0.88) and within-group |cos| rises emergently — superposition already
exploits anti-correlation. An explicit group-alignment placement loss
increases alignment further but does not beat the emergent floor.

---

## Quickstart

```bash
git clone https://github.com/PalmeseMattia/checkerboard && cd checkerboard
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # CPU torch is fine for the smoke run

pytest                                    # 35 tests: theory closed-forms, shapes, determinism
bash scripts/reproduce_smoke.sh          # < 5 min on CPU, exercises every script end-to-end
```

The smoke run writes a reduced `results/smoke/report.md` so you can confirm
the whole pipeline works without a GPU.

---

## Full reproduction

```bash
bash scripts/reproduce_all.sh             # regenerates results/ end-to-end
```

This deletes `results/`, retrains everything, and rebuilds every figure and
`results/report.md`. Per-experiment runtimes (RTX 3060 vs one core of a
12-core CPU; the scripts train all widths×seeds of a config as **one batched
tensor computation**, so wall-clock is far below the model count):

| Experiment | Config | GPU | CPU |
|---|---|--:|--:|
| Exp 0 full sweep | 12 configs, widths 1..≈27, 5 seeds | ~6 min | ~2.5 h |
| Capacity probe | n=200, 4 α, 3 seeds | ~6 min | ~25 min |
| Slope-law probe | n=200, 6 slopes × 4 α, 3 seeds | ~5 min | ~30 min |
| Convergence (n=400, 1×+3×) | α=0.99 uniform | ~6 min | ~30 min |
| Predictors | analysis only (no training) | < 5 s | < 5 s |
| Exp A / B / C | n=40, α=0.90, 5 seeds | ~1 / 1.5 / 2 min | ~15 / 20 / 25 min |

GPU is selected automatically (`device: auto`); pass `--device cpu` to force
CPU. **Determinism:** a run is fully determined by its seed *on a fixed
device*. CPU and CUDA use different RNG streams, so results are reproducible
per device, not bit-identical across devices (`tests/test_determinism.py`
asserts the per-device guarantee). Every output JSON embeds its full config;
the only field that varies between identical re-runs is `runtime_s`.

Every experiment script takes `--config <yaml>` plus optional `--device` and
`--outdir` overrides, e.g.:

```bash
python experiments/exp0_replication.py --config configs/exp0_full.yaml --device cpu
```

---

## Repository layout

```
README.md  LICENSE (MIT)  CITATION.cff  requirements.txt
SPEC.md       the original experiment specification (verbatim project brief)
src/          model.py data.py theory.py train.py metrics.py plotting.py
experiments/  exp0_replication.py expA_audit.py expB_placement.py
              expC_correlation.py probe_capacity.py probe_slope_law.py
              predictors.py        (common.py = shared YAML/IO plumbing)
configs/      <experiment>_full.yaml and _smoke.yaml per experiment
scripts/      reproduce_all.sh  reproduce_smoke.sh  make_report.py
results/      generated: per-config JSONs, 200-dpi PNGs, report.md
tests/        pytest (theory, metrics, model, train, distill, determinism)
```

`make_report.py` rebuilds `results/report.md` (executive summary + one
section per experiment) purely from the JSONs — it never trains anything.

---

## Relationship to prior work

We build on three works; here is precisely what is theirs vs. ours.

**Elhage et al. 2022, *Toy Models of Superposition*** (theirs). The model
x̂ = ReLU(WᵀWx + b), the sparse-feature data distribution
(xᵢ ~ U[0,1] w.p. 1−α), Zipf importances, and the qualitative
feature-survival picture. Reused verbatim (`src/model.py`, `src/data.py`).

**Scherlis et al. 2022, *Polysemanticity and Capacity in Neural Networks***
(theirs). The fractional capacity Cᵢ = ‖Wᵢ‖⁴ / Σⱼ (Wᵢ·Wⱼ)², satisfying
Σᵢ Cᵢ ≤ d (`src.metrics.feature_capacity`). Used as a threshold-free measure
of decoder occupancy.

**Sarkar & Deka 2026, *Geometric Limits of Knowledge Distillation***
(theirs — the work we audit). The capacity function g(α), the critical width
d* = n/g(α), the Eq. 2 loss floor with F = ⌊d·g(α)⌋, and **assumption A2**.
All in `src/theory.py` with their notation (`g_alpha`, `d_star`, `F_kept`).

**Ours.** (1) An empirical audit showing A2 fails in two ways near d* — the
kept *count* is below F (a superposition equilibrium), and the *ordering*
has margin violations. (2) The equilibrium-corrected floor predictor that
flips absolute-floor R² positive where Eq. 2 fails, and the d_S=d_T
geometric/residual split showing baseline B is contaminated (claims 2–3).
(3) The packing law ĝ(α, s) and its slope dependence (claims 4–6). (4) The
distillation menu constraint (Exp A) and the placement-cost Pareto (Exp B).
(5) The anti-correlation result (Exp C).

---

## Limitations

- **Toy-model regime only.** Synthetic sparse data and a single ReLU
  decoder family; no claim transfers directly to trained transformers. The
  value is in stress-testing the *assumptions* of the width theorem.
- **Exp A/B/C are demonstrated at a single (n, α) = (40, 0.90).** Only Exp 0
  and the packing probes sweep multiple configs; multi-config replication of
  the distillation, placement, and correlation results is future work.
- **Single decoder architecture.** x̂ = ReLU(WᵀWx + b) with tied
  encoder/decoder, as in Elhage et al. A different head (untied weights,
  layernorm, deeper decoder) could allocate capacity differently.
- **"Kept" is operationalized by a norm² threshold** (‖Wᵢ‖² > 0.5). This is
  a choice; we report the threshold-free Σ Cᵢ alongside it and verify the
  packing-law exponents are stable across τ ∈ [0.3, 0.7] (robustness table
  in the report and `probe_slope_law` output). Cᵢ thresholds above 0.5
  degenerate (an antipodal pair already has C = 0.5).
- **Equalization is partial.** `--equalize-active` equalizes per-feature
  gradient counts across α; pairwise co-activation still scales (1−α)² and
  is not equalized, so extreme-α comparisons carry that caveat.
- **One packing-law fit protocol.** A single documented protocol
  (`src.theory.fit_packing_law`) produces every packing-law number.

### A note on the unified packing-law fit

Two earlier runs of the same protocol gave the Zipf exponent as 0.57
(capacity-probe fit) and 0.53 (threshold-robustness fit) — within
seed-to-seed noise. We standardized on **one** instrument and protocol: the
slope-law probe (`experiments/probe_slope_law.py`), which stores per-feature
norms (enabling the threshold-robustness check), at n=200, d=10, 3 seeds,
update-equalized to 1e6 active samples/feature, kept count by ‖Wᵢ‖² > 0.5,
log-log least squares over α ∈ {0.80, 0.90, 0.95, 0.99}. All numbers and
figures in the release are regenerated with this single fit, which gives the
Zipf law ĝ(α) = 1.03·g(α)^0.53. The predictors read these coefficients
directly from `probe_slope_law_n200_fit.json`, so the report and the README
cannot drift apart.

---

## AI-assistance disclosure

Research planning and analysis design were done in collaboration with Claude
(Anthropic); the implementation was written with Claude Code. **All results
were verified by the author through reproduction** — every number and figure
in `results/report.md` regenerates from a clean clone via
`scripts/reproduce_all.sh`, and the claims map above ties each finding to the
command that produces it.

---

## License & citation

MIT (see [`LICENSE`](LICENSE)). If you use this software or its findings,
please cite via [`CITATION.cff`](CITATION.cff).
