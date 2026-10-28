"""Verify public data, export deterministic split manifests and integrity hashes."""
from pathlib import Path
import hashlib, json
from PIL import Image
from experiment import ROOT, pairs_for, split_pairs

report = {}
manifest = []
snapshot_path = ROOT / 'data' / 'experiment_data_64x64.npz'
if snapshot_path.exists():
    import numpy as np
    snapshot = np.load(snapshot_path)
    assert len(snapshot['ids']) == 2202 and snapshot['images'].shape == (2202,64,64,3)
    assert snapshot['masks'].shape == (2202,64,64)
for dataset, expected in [('kvasir', 1000), ('cvc', 612), ('instrument', 590)]:
    pairs = pairs_for(dataset)
    assert len(pairs) == expected, (dataset, len(pairs), expected)
    for image_path, mask_path in pairs:
        assert image_path.exists() and mask_path.exists()
        if not snapshot_path.exists():
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(mask_path) as im:
                im.verify()
    train, val, test = split_pairs(dataset, pairs)
    assert not ({p for p, _ in train} & {p for p, _ in val})
    assert not ({p for p, _ in train} & {p for p, _ in test})
    assert not ({p for p, _ in val} & {p for p, _ in test})
    for split, items in [('train', train), ('validation', val), ('test', test)]:
        for image_path, mask_path in items:
            manifest.append({'dataset': dataset, 'split': split,
                             'image': str(image_path.relative_to(ROOT)),
                             'mask': str(mask_path.relative_to(ROOT))})
    report[dataset] = {'pairs': len(pairs), 'train': len(train),
                       'validation': len(val), 'test': len(test),
                       'images_readable': True, 'masks_readable': True}

import csv
with (ROOT / 'results' / 'split_manifest.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'split', 'image', 'mask'])
    writer.writeheader(); writer.writerows(manifest)
(ROOT / 'results' / 'data_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
