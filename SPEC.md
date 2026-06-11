# Original specification — Controlled Feature Placement in Distillation

(Verbatim project brief; source of truth for experiment definitions.)

## Goal

Build a small, clean PyTorch research codebase to test whether **controlled
"feature placement" during knowledge distillation** can override the emergent
capacity allocation that arises from superposition — and at what cost.

This extends two works:
- Elhage et al. 2022, *Toy Models of Superposition* (the model and setup)
- Sarkar & Deka 2026, *Geometric Limits of Knowledge Distillation: A
  Minimum-Width Theorem via Superposition Theory* (arXiv:2604.04037 — the
  capacity formula and the assumption we audit)

## Theoretical background encoded in `src/theory.py`

- **Data**: x ∈ R^n with sparse features: x_i ~ U[0,1] with probability
  (1−α), else 0. α is the sparsity.
- **Model (Elhage toy model)**: W ∈ R^{d×n}, b ∈ R^n. Forward: h = Wx,
  x̂ = ReLU(WᵀWx + b) = ReLU(Wᵀh + b).
- **Loss**: importance-weighted MSE, L = E[ Σ_i I_i (x̂_i − x_i)² ], with
  Zipf importances I_i ∝ 1/i.
- **Capacity function**: g(α) = 1 / ((1−α) · ln(1/(1−α))). A width-d model
  encodes ≈ d·g(α) features. Critical width d* = n / g(α).
- **Predicted loss floor (Eq. 2 of the paper)**: with F_S = floor(d_S · g(α))
  and features sorted by importance, L*(d_S) = Σ_{i > F_S} I_i · E[x_i²],
  where E[x_i²] = (1−α)/3 for this data distribution.
- **Assumption A2 (the one we audit)**: the student allocates capacity
  optimally by importance order, keeping exactly the top-F_S features.

## Experiments

### Exp 0 — Replication (sanity)
Train models of width d_S = 1..d_T directly on the task. Compare achieved
floors to L*(d_S). Default config: n=40, d_T=10, α=0.90, Zipf importances,
≥5 seeds. Full sweep flag: n ∈ {20, 40, 80}, α ∈ {0.80, 0.90, 0.95, 0.99}.

### Exp A — Allocation audit (does vanilla distillation respect A2?)
1. Train a teacher of width d_T on the task.
2. Distill students of every width d_S: student is trained to reconstruct the
   **teacher's output** x̂_T (importance-weighted MSE against x̂_T).
3. Per student, measure per-feature representation: column norms ‖W_S[:,i]‖²
   and per-feature MSE on the true task. Define the **survived set**
   S = {i : ‖W_S[:,i]‖² > 0.5}.
4. Metric: **overlap@k** = |S ∩ top-|S|-by-importance| / |S|. Report vs d_S
   across seeds. Hypothesis: overlap < 1 near the cutoff (A2 violated at the
   margin); quantify how much and where.
5. Also produce the feature-survival "staircase" plot (features × widths,
   colored by ‖W_i‖²).

### Exp B — Controlled placement (the core contribution)
1. Pick a set C of "critical" features with low emergent importance
   (e.g., ranks {20, 28, 36} of n=40).
2. Distill with **training weights Ĩ** where Ĩ_i = I_i for i ∉ C and
   Ĩ_i = β·I_i for i ∈ C (boost factor β). Evaluation always uses the true
   importances I.
3. Measure, vs vanilla distillation at the same d_S (use d_S well below d*):
   - survival rate of C (fraction of critical features in S),
   - true floor cost ΔL (placement floor − vanilla floor),
   - which features got evicted to make room.
4. Sweep β ∈ {1, 3, 10, 30, 100} → **Pareto curve**: critical survival vs
   floor cost.
5. Compare achieved placement floor against the *predicted* floor obtained by
   re-running Eq. 2 with the importance ordering induced by Ĩ. Claim to test:
   the cost of placement is predictable from the theory.

### Exp C — Correlation-aware packing (stretch goal, behind a flag)
Generate features with block correlation structure (mutually exclusive
groups: at most one feature per group active). Test whether (a) emergent
superposition already exploits anti-correlation, and (b) a placement loss
that explicitly packs mutually exclusive features into shared directions
beats the iid prediction L*(d_S). This probes a known limitation of the g(α)
formula (it assumes iid features).

## Engineering requirements

- PyTorch with autograd (no manual gradients). Vectorize aggressively: train
  all (widths × seeds) models in parallel as one batched tensor computation
  (einsum or vmap over a leading "model" dimension).
- CPU-friendly by default (full default run ≤ ~10 min on CPU); use CUDA/MPS
  if available.
- Repo layout: `src/` (model.py, data.py, theory.py, train.py, metrics.py),
  `experiments/run.py` (CLI: --exp {0,A,B,C} --full-sweep --seeds N),
  `tests/` (pytest), `results/` (auto-created: JSON + PNG + report.md).
- Reproducibility: explicit seeds everywhere, config as dataclass, dump full
  config into every results JSON.
- Figures (matplotlib, saved to results/):
  1. Floor vs d_S, actual (mean ± std) vs predicted, log y, dotted vertical
     at d* — per config.
  2. Feature survival heatmap (feature rank × d_S).
  3. overlap@k vs d_S with error bars (Exp A headline figure).
  4. Pareto: critical survival vs floor cost across β (Exp B headline figure).
  5. Per-feature loss, vanilla vs placement, highlighting set C.
- Auto-generate `results/report.md` summarizing config, headline numbers,
  and embedding the figures.
- Code and comments in English (paper-ready). README with install + run.

## Success criteria

1. Exp 0 reproduces the qualitative floor-vs-width behavior and the d* phase
   transition (Pearson r > 0.9 between predicted and achieved floors across
   the sweep).
2. Exp A produces a quantified answer to "is A2 exactly true?" — either
   result is interesting; report it honestly.
3. Exp B shows critical-feature survival ≥ 90% at d_S « d* for some β, with
   measured floor cost compared against the theory-predicted cost.
4. All tests pass; a single command reproduces every figure.
