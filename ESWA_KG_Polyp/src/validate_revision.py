"""Integrity checks for all revised-manuscript inputs."""
from pathlib import Path
import json, re

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
SEEDS = [13,29,47]

def load(p):
    return json.loads(Path(p).read_text())

def check_run(path, size=64):
    d = load(path/'result.json')
    assert d['epochs']==12 and d['size']==size and d['test_n']==149
    assert len(d['history'])==12 and len(load(path/'per_image.json'))==149
    return d

count = 0
for v in ['dual','graph_only','relation','boundary']:
    for s in SEEDS:
        check_run(R/f'kvasir_ablation_{v}_s{s}_size64')
        count += 1
for m in ['attention_original','unetpp','deeplab']:
    for s in SEEDS:
        check_run(R/f'kvasir_{m}_s{s}_r0.05_b0.1')
        count += 1
for b in [0,0.05,0.2]:
    d = check_run(R/f'kvasir_bcskg_s13_r0.05_b{float(b)}')
    assert d['lambda_bd']==b
    count += 1
for m in ['unet','bcskg']:
    check_run(R/f'kvasir_{m}_s13_r0.05_b0.1_size128',size=128)
    count += 1
for s in SEEDS:
    d = check_run(R/f'kvasir_unet_s{s}_r0.05_b0.1_base10')
    assert d['base']==10
    count += 1
    d = check_run(R/f'kvasir_bcskg_s{s}_r0.05_b0.1_branch0.0')
    assert d['lambda_branch']==0
    count += 1
assert count==32
main_tags = [f'kvasir_{m}_s{s}_r0.05_b0.1' for m in ['unet','attention','attention_original','unetpp','deeplab','bcskg'] for s in SEEDS]
main_tags += [f'kvasir_unet_s{s}_r0.05_b0.1_base10' for s in SEEDS]
for tag in main_tags:
    d = check_run(R/tag)
    rob = load(R/tag/'robustness.json')
    assert len(rob)==9 and all(0<=v<=1 for v in rob.values())
text = (ROOT/'manuscript/references.bib').read_text()
dois = re.findall(r'\bdoi\s*=\s*\{([^}]+)\}',text,re.I)
assert len(dois)==len(set(x.lower() for x in dois))==37
assert 'https://doi.org/' not in text.lower()
for f in ['revision_statistics.json','revision_diagnostics.json','expanded_efficiency_benchmark.json']:
    assert (R/f).exists()
for f in ['fig11_protocol_diagnostics.png']:
    assert (ROOT/'figures'/f).exists()
for f in ['ablation.tex','robustness.tex','main_results.tex']:
    assert (ROOT/'tables'/f).exists()
print(json.dumps({'new_training_runs':count,'robustness_evaluations':21,
                  'unique_dois':len(dois),'status':'all revision inputs verified'},indent=2))
