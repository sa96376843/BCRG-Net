"""Verify the public Kvasir-SEG files against this paper's fixed split."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "data" / "Kvasir-SEG"

PAIRS = [(image, base / "masks" / image.name) for image in sorted((base / "images").glob("*"))]
assert len(PAIRS) == 1000, f"Expected 1000 Kvasir-SEG pairs, found {len(PAIRS)}"
for image, mask in PAIRS:
    with Image.open(image) as f:
        f.verify()
    with Image.open(mask) as f:
        f.verify()

train, val, test = [], [], []
for image, mask in PAIRS:
    value = int(hashlib.md5(("kvasir::" + image.stem).encode()).hexdigest()[:8], 16) / 0xffffffff
    (train if value < .70 else val if value < .85 else test).append((image, mask))
assert tuple(map(len, (train, val, test))) == (713, 138, 149)
actual = {
    (split, str(image.relative_to(ROOT)).replace("\\", "/"),
     str(mask.relative_to(ROOT)).replace("\\", "/"))
    for split, pairs in (("train", train), ("validation", val), ("test", test))
    for image, mask in pairs
}
with (ROOT / "results" / "split_manifest.csv").open(newline="", encoding="utf-8") as f:
    expected = {
        (row["split"], row["image"].replace("\\", "/"), row["mask"].replace("\\", "/"))
        for row in csv.DictReader(f) if row["dataset"] == "kvasir"
    }
assert actual == expected, f"Split mismatch: {len(actual - expected)} new, {len(expected - actual)} missing"
print("Kvasir-SEG verified: 1000 pairs; train=713, validation=138, test=149")
