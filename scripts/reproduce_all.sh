#!/usr/bin/env bash
# Regenerate every result, figure, and report.md from scratch.
# Deletes results/ first so the run is hermetic. Runtimes (RTX 3060 / 12-core CPU):
#   GPU ~25 min total; CPU ~3-4 h (Exp 0 full sweep dominates).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

rm -rf results
mkdir -p results

echo "== Exp 0 (full sweep, 12 configs) =="
$PY experiments/exp0_replication.py --config configs/exp0_full.yaml

echo "== Capacity-gap probe (n=200) =="
$PY experiments/probe_capacity.py   --config configs/probe_capacity_full.yaml

echo "== Slope-law probe (canonical packing-law fit) =="
$PY experiments/probe_slope_law.py  --config configs/probe_slope_law_full.yaml

echo "== alpha=0.99 convergence checks (n=400, 1x and 3x) =="
$PY experiments/probe_slope_law.py  --config configs/convergence_n400_1x.yaml
$PY experiments/probe_slope_law.py  --config configs/convergence_n400_3x.yaml

echo "== Absolute-floor predictors (reads Exp 0 + slope-law fit) =="
$PY experiments/predictors.py       --config configs/predictors_full.yaml

echo "== Exp A / B / C =="
$PY experiments/expA_audit.py       --config configs/expA_full.yaml
$PY experiments/expB_placement.py   --config configs/expB_full.yaml
$PY experiments/expC_correlation.py --config configs/expC_full.yaml

echo "== Report =="
$PY scripts/make_report.py results
echo "Done. See results/report.md"
