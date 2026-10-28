"""Pack archived predictions for a code release without redistributing masks.

`pack` reads the original per-case archives and stores only model probabilities
and available exact binary predictions. `unpack` recreates the per-case layout
expected by the figure scripts. Neither operation stores ground-truth arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUNDLES = RESULTS / "prediction_bundles"


def pack() -> None:
    BUNDLES.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for run in sorted(RESULTS.iterdir()):
        if not run.is_dir() or run == BUNDLES:
            continue
        files = sorted((run / "predictions").glob("*.npz"))
        if not files:
            continue
        ids = [p.stem for p in files]
        probs = []
        for path in files:
            with np.load(path) as z:
                probs.append(z["prob"].astype(np.float16, copy=False))
        binary_files = sorted((run / "predictions_binary").glob("*.npz"))
        binary_ids = [p.stem for p in binary_files]
        binary = []
        for path in binary_files:
            with np.load(path) as z:
                binary.append(z["pred"].astype(np.uint8, copy=False))
        output = BUNDLES / (run.name + ".npz")
        np.savez_compressed(
            output,
            ids=np.asarray(ids),
            prob=np.stack(probs),
            binary_ids=np.asarray(binary_ids),
            pred=np.stack(binary) if binary else np.empty((0, *probs[0].shape), dtype=np.uint8),
        )
        missing = sorted(set(ids) - set(binary_ids))
        manifest[run.name] = {
            "prob_count": len(ids),
            "exact_binary_count": len(binary_ids),
            "missing_exact_binary_ids": missing,
            "shape": list(probs[0].shape),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    (BUNDLES / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packed {len(manifest)} runs into {BUNDLES}")


def unpack() -> None:
    for path in sorted(BUNDLES.glob("*.npz")):
        run = RESULTS / path.stem
        prob_dir = run / "predictions"
        binary_dir = run / "predictions_binary"
        if list(prob_dir.glob("*.npz")) or list(binary_dir.glob("*.npz")):
            raise FileExistsError(f"Prediction files already exist in {run}; refusing to overwrite them")
        prob_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)
        with np.load(path) as z:
            ids = z["ids"].tolist()
            binary_ids = z["binary_ids"].tolist()
            for sample_id, prob in zip(ids, z["prob"], strict=True):
                np.savez_compressed(prob_dir / f"{sample_id}.npz", prob=prob)
            for sample_id, pred in zip(binary_ids, z["pred"], strict=True):
                np.savez_compressed(binary_dir / f"{sample_id}.npz", pred=pred)
        print(f"Unpacked {path.stem}: {len(ids)} probabilities, {len(binary_ids)} exact masks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pack", "unpack"))
    args = parser.parse_args()
    (pack if args.action == "pack" else unpack)()
