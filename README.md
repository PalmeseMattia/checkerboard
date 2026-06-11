# Controlled Feature Placement in Distillation

[![DOI](https://zenodo.org/badge/1265708873.svg)](https://doi.org/10.5281/zenodo.20640978)

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

This is an **audit** of the assumptions behind the Sarkar & Deka loss-floor
theorem, in the toy-model regime where those assumptions can be measured
directly. The refined Eq. 2 floor tracks achieved floors well *below* the
critical width d* (Pearson r ≈ 0.98 across a 12-config sweep) but
**systematically underestimates absolute floors near and above d*** —
the regime where their affine Pythia calibration (C = 8.97 > 1) absorbs
the discrepancy. The audit localizes the failure precisely: the
**zero-fitted-parameter predictor (b′)** — Eq. 2 charged on each run's own
*measured* kept set — is the best absolute predictor in every stratum
(R² = 0.97 overall, 0.80 near d*, 0.64 above; per-config mean 0.91), and
the measured per-dropped-feature cost is ≈0.9× the Eq. 2 charge. So Eq. 2's
*charging* is essentially correct; what fails is **assumption A2's kept
count**: trained students settle at a superposition equilibrium that keeps
fewer, cleaner features than F = ⌊d·g(α)⌋.

Replacing the kept count with a one-fitted-law equilibrium count,
**F = round(d·ĝ(α)) with ĝ(α) = 1.03·g(α)^0.53** (seed-bootstrap SE ±0.03 /
±0.01), recovers most of (b′)'s accuracy *in the pooled metrics*: the
near/above-d* strata go from R² = −3.8 / −2.0 (Eq. 2) to 0.67 / 0.49.
Per-config it is uneven (mean −0.04 / median 0.83; see Results). The
packing exponent is importance-slope dependent (b falls from 0.93 ± 0.01
for uniform importances to 0.34 ± 0.02 at s = 3), so any recalibration of
their C/d*/B must be slope-specific. Vanilla distillation respects A2's
*ordering* about as well as direct training but adds a hard *menu*
constraint (students keep only features their teacher represents);
controlled placement can override emergent allocation at a small absolute
cost, until the teacher menu binds. On block-anticorrelated data, emergent
superposition already beats the iid floor — the theory's iid assumption is
conservative.

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
| 2 | Eq. 2 R² goes **negative** near/above d*; the zero-parameter (b′) and the equilibrium count (b) restore it — the failure is the kept count, not the charging | predictors | **`fig7_predicted_vs_observed.png`** | `predictors.py --config configs/predictors_full.yaml` |
| 3 | At d_S=d_T the geometric share of the floor is 0.61–0.96 (mean 0.75) via the measured kept set (b′), 0.40–0.73 (mean 0.63) via the fitted law (b) → baseline B is contaminated | predictors | `fig7` | same as #2 |
| 4 | g(α) bound is approached under uniform importances (see Limitations for the n-dependent prefactor); Zipf equilibrates below it | capacity probe | `fig_probe_capacity_n200.png` | `probe_capacity.py --config configs/probe_capacity_full.yaml` |
| 5 | Packing exponent b is threshold-stable and slope-dependent (0.93 → 0.34) | slope-law probe | `fig8_slope_law_n200.png` | `probe_slope_law.py --config configs/probe_slope_law_full.yaml` |
| 6 | α=0.99 uniform sub-g packing is a real equilibrium, not under-training | convergence | (table in report) | `probe_slope_law.py --config configs/convergence_n400_{1x,3x}.yaml` |
| 7 | Distillation respects A2 ordering ~ as well as direct, but adds a teacher menu | Exp A | `fig3_overlap_*` | `expA_audit.py --config configs/expA_full.yaml` |
| 8 | Placement overrides emergent allocation at a small absolute cost, ordered as the theory predicts (menu-bound under distillation) | Exp B | `fig4_pareto_*`, `fig5_perfeature_*` | `expB_placement.py --config configs/expB_full.yaml` |
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

**Floor predictors (the core result).** Absolute-floor R² (MAE in
parentheses) on 600 points (12 configs × widths × seeds). Strata are
**fixed in config** (`configs/predictors_full.yaml`) as d/d* bands:
below d/d* < 0.8, near 0.8–1.2, above > 1.2. Honesty note: these bands
were declared after v1.0 (which used ±1 absolute-width bands); re-banding
moved the fitted-law predictor's near/above R² from 0.70/0.58 to
0.67/0.49 — i.e. *against* the then-headline predictor — while (b′) is
insensitive to the choice. The above-d* stratum has n=30 points but only
**~6 independent (config, width) cells** (5 seeds each), so its
conclusions are indicative, not load-bearing.

| stratum | n | (a) Eq. 2 | (b) equilibrium, round() | (b) w/ floor() | (b′) measured kept set | (c) per-feature Cᵢ |
|---|--:|--:|--:|--:|--:|--:|
| overall | 600 | 0.52 (.026) | 0.97 (.006) | 0.93 (.007) | **0.97 (.005)** | −0.53 (.045) |
| below d* | 385 | 0.57 (.025) | 0.97 (.006) | 0.92 (.007) | 0.97 (.005) | −0.65 (.048) |
| near d* | 185 | **−3.76** (.028) | 0.67 (.008) | 0.72 (.007) | **0.80 (.006)** | −8.59 (.041) |
| above d* | 30 | **−2.03** (.014) | 0.49 (.006) | 0.56 (.005) | **0.64 (.004)** | −10.24 (.025) |

Per-config aggregated R² (12 units): Eq. 2 mean **−5.4** / median −1.2 (the
pooled 0.52 is held up by cross-config variance); **(b) mean −0.04 /
median 0.83** — the negative mean is driven by the three α=0.99 configs
(R² = −5.8 / −1.7 / −0.5), where floors are tiny and a ±1-feature count
error explodes R²; (b′) stays ≥ 0.85 on those same configs and is mean
**0.91** / median 0.91 overall. So (b) matches (b′) in pooled metrics but
not per-config; only (b′) is uniformly good. The **zero-fitted-parameter
(b′)** — Eq. 2 charged on each run's own measured kept set — is the best
predictor in every stratum, and the gap decomposition shows dropped
importance carries 72–87% of the achieved floor with the measured
per-dropped-feature cost at ≈0.9× the Eq. 2 charge. (b′)'s residual near
d* is itself predictable: it overcharges dropped features by ~1/0.9 and
omits the ~27% interference share on kept features, netting ~20%
under-prediction — "charging essentially correct" holds to first order.
Together: **Eq. 2's charging is right to first order; A2's kept count is
what fails.** The rounding convention in (b) (floor vs round of d·ĝ) is
worth ±0.03–0.07 R² near d* — both are reported. The per-feature form (c)
*overshoots* — a partially represented feature plus an optimal bias recovers
most of its variance, so fractional capacity does **not** map linearly to
loss (a useful negative result). At d_S = d_T the geometric (superposition)
share of the observed floor is **0.61–0.96 (mean 0.75)** via the measured
kept set (b′), 0.40–0.73 (mean 0.63) via the fitted law (b) — either way the
"architectural baseline" B read at d_S = d_T is contaminated by
superposition cost.

**Packing law (capacity & slope probes).** With uniform importances the
achieved packing scales near-proportionally to the bound
(ĝ = 0.93·g^0.93, seed-bootstrap SE ±0.01/±0.01) — though with a prefactor
below 1 and a measurable n-dependence (see Limitations) — while Zipf
equilibrates well below it (ĝ = 1.03·g^0.53, SE ±0.03/±0.01), with
Σ Cᵢ/d ≈ 1.0 in both cases: capacity is *fully used*, just allocated to
fewer, cleaner features. The exponent b is flat across survival thresholds
τ ∈ [0.3, 0.7] but falls monotonically with importance steepness
(0.93, 0.69, 0.53, 0.44, 0.34, 0.34 for s = 0, 0.5, 1, 1.5, 2, 3). At
α=0.99 the uniform packing fraction is convergence-stable: at fixed n=400,
tripling the training budget moves survived/d by −2.9% (16.10 → 15.63
features/dim) — a plateau, so the sub-g packing is a real equilibrium, not
slow symmetry-breaking. The n=400 (3×) value sits −4.7% below the n=200
baseline (16.40), an n-dependence of the prefactor listed in Limitations.

**Exp A — distillation allocation audit.** Distilled and direct students
violate A2's ordering by the same small margin (overlap@k ∈ [0.80, 0.97],
never 1.0), so distillation is not the culprit. But **teacher containment is
1.000 at every width** — a distilled student keeps only features its teacher
represents (the teacher's |S_T| = 22/40 menu). The teacher count is itself
an out-of-experiment check of the equilibrium law: d_T·ĝ(0.90) = 22.5, and
all five teachers keep exactly ⌊22.5⌋ = 22 features. One further
observation, single-config and unreplicated: distilled students keep
slightly *more* features than direct ones at mid widths (e.g. 14.6 vs 13.6
at d_S=6), consistent with the teacher output being a *denoised target* —
features the teacher dropped contribute no target variance, freeing
capacity — but we flag it as a hypothesis, not a finding.

**Exp B — controlled placement.** Boosting the training weight of a
low-importance critical set C = ranks {20, 28, 36} by β makes those features
survive. Under the task-target control, survival crosses 90% at β=10 for
d_S=3 (0.93; 100% from β=30) and reaches 100% at β=10 for d_S=5 and at β=3
for d_S=7. The floor cost is **small in absolute terms** —
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
Their *correlated and anticorrelated features* section already shows
qualitatively that anticorrelated features prefer shared (antipodal)
directions — Exp C quantifies that effect against the iid Eq. 2 floor in
the distillation context.

**Liu, Liu & Gore 2025 (arXiv:2505.10465)** (theirs). A scaling analysis of
loss and interference regimes in this same toy model. Complementary to us:
they characterize how loss scales with width/sparsity within the model
family; we audit the *kept-count assumption* of the compressed-sensing
capacity bound in the distillation-floor context, and test predictors of
the absolute floor.

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
  a choice; we report the threshold-free Σ Cᵢ alongside it. The τ-stability
  claim is scoped to **this norm² operationalization**: the exponents are
  flat across τ ∈ [0.3, 0.7] because the column-norm distribution is
  strongly bimodal — norms pile up near 0 (dropped) and near/above 1
  (kept), leaving the threshold window nearly empty (histogram:
  `results/fig9_norm_hist_n200.png`). The Cᵢ-threshold columns in the
  robustness table degenerate above C = 0.5 by construction (an antipodal
  pair already has C = 0.5), so they test a different, coarser
  operationalization, not the same claim. The operationalization's
  strongest defense is **(b′) itself**: the kept set it defines yields the
  best absolute-floor predictions with zero fitted parameters.
- **Equalization is partial.** `--equalize-active` equalizes per-feature
  gradient counts across α; pairwise co-activation still scales (1−α)² and
  is not equalized, so extreme-α comparisons carry that caveat.
- **The packing-law prefactor is n-dependent.** At α=0.99 the uniform
  fraction of the bound drifts from 0.76 (n=200) to 0.72 (n=400, 3×
  training budget; −4.7%), and the historical n=40 probe sat higher still.
  The exponents are stable; the prefactors should be read as
  n≈200-specific. "Reachable" claims about the g(α) bound are therefore
  qualified, not exact.
- **One packing-law fit protocol.** A single documented protocol
  (`src.theory.fit_packing_law`) produces every packing-law number, now
  with seed-bootstrap standard errors on every prefactor and exponent.
- **Width grid / point count.** Exp 0 trains widths 1..⌈d*⌉+1 per config
  (per-config counts: n=20 → 8/6/4/2 widths for α = 0.80/0.90/0.95/0.99;
  n=40 → 14/11/7/3; n=80 → 27/20/13/5; total 120 widths × 5 seeds =
  **600 points**).

### A note on the unified packing-law fit

Two earlier independent runs of the same protocol gave the Zipf exponent as
0.57 (capacity-probe fit) and 0.53 (threshold-robustness fit). With the
seed-bootstrap SE now quantified (±0.01), that gap is ≈2 SE — consistent
with run-to-run variation but not trivially "noise"; we flag it rather than
average it. We standardized on **one** instrument and protocol: the
slope-law probe (`experiments/probe_slope_law.py`), which stores per-feature
norms (enabling the threshold-robustness check and the bootstrap), at n=200,
d=10, 3 seeds, update-equalized to 1e6 active samples/feature, kept count by
‖Wᵢ‖² > 0.5, log-log least squares over α ∈ {0.80, 0.90, 0.95, 0.99}. All
numbers and figures in the release are regenerated with this single fit,
which gives the Zipf law ĝ(α) = 1.03·g(α)^0.53 (SE ±0.03/±0.01). The
predictors read these coefficients directly from
`probe_slope_law_n200_fit.json`, so the report and the README cannot drift
apart.

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
please cite via [`CITATION.cff`](CITATION.cff) or the archived release —
DOI [10.5281/zenodo.20640978](https://doi.org/10.5281/zenodo.20640978).
