#!/usr/bin/env bash
# Reduced end-to-end run: exercises every script in < 5 min on CPU so a
# reviewer can verify the pipeline without a GPU. Writes to results/smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

rm -rf results/smoke
mkdir -p results/smoke

$PY experiments/exp0_replication.py --config configs/exp0_smoke.yaml          --device cpu
$PY experiments/probe_capacity.py   --config configs/probe_capacity_smoke.yaml --device cpu
$PY experiments/probe_slope_law.py  --config configs/probe_slope_law_smoke.yaml --device cpu
$PY experiments/predictors.py       --config configs/predictors_smoke.yaml
$PY experiments/expA_audit.py       --config configs/expA_smoke.yaml          --device cpu
$PY experiments/expB_placement.py   --config configs/expB_smoke.yaml          --device cpu
$PY experiments/expC_correlation.py --config configs/expC_smoke.yaml          --device cpu
$PY scripts/make_report.py results/smoke
echo "Smoke OK. See results/smoke/report.md"
