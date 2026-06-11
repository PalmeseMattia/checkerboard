"""Shared experiment plumbing: YAML configs, output paths, JSON dumps.

Every experiment script accepts `--config configs/<name>.yaml` plus
`--device` and `--outdir` overrides. The full config (YAML values merged
over defaults, plus the resolved TrainConfig) is dumped into every output
JSON so results are self-describing.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_config(default_yaml: str, description: str) -> SimpleNamespace:
    """Parse --config/--device/--outdir and return the merged config."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / default_yaml),
                        help="YAML config (see configs/)")
    parser.add_argument("--device", default=None, help="override device (auto/cpu/cuda)")
    parser.add_argument("--outdir", default=None, help="override output directory")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip training; re-aggregate existing JSONs "
                             "(probes only)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.device is not None:
        cfg["device"] = args.device
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    cfg.setdefault("device", "auto")
    cfg.setdefault("outdir", "results")
    cfg["aggregate_only"] = args.aggregate_only
    ns = SimpleNamespace(**cfg)
    ns.outdir = Path(ns.outdir)
    ns.outdir.mkdir(parents=True, exist_ok=True)
    return ns


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def tag(n: int, alpha: float) -> str:
    return f"n{n}_a{alpha:.2f}"


def equalized_schedule(alpha: float, target_active: float) -> tuple[int, int]:
    """(batch, steps) giving ~target_active active samples per feature.

    Equalizes the number of nonzero-gradient updates each feature receives
    across sparsities (an active sample contributes a gradient; activation
    probability is 1-alpha). Pairwise co-activation counts still scale as
    (1-alpha)^2 and cannot be equalized this way — flagged when
    interpreting extreme-alpha results.
    """
    import numpy as np
    batch = int(min(4096, max(1024, 41 / (1 - alpha))))
    steps = int(np.ceil(target_active / (batch * (1 - alpha))))
    return batch, steps
