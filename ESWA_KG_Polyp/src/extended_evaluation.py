"""Post-hoc robustness and cross-dataset evaluation of frozen trained weights."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from experiment import ROOT, RESULTS, CachedSeg, pairs_for, split_pairs, evaluate
from model import BCSKGNet
from baseline_models import build_additional_baseline

class Corrupted(Dataset):
    def __init__(self, base, kind, severity):
        self.base, self.kind, self.severity = base, kind, severity
        self.mean = torch.tensor([.485,.456,.406])[:,None,None]
        self.std = torch.tensor([.229,.224,.225])[:,None,None]
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y, name = self.base[i]
        if self.kind == 'noise':
            g = torch.Generator().manual_seed(1009 + i)
            x = x + torch.randn(x.shape, generator=g) * self.severity
        elif self.kind == 'brightness':
            raw = (x * self.std + self.mean).clamp(0, 1) * self.severity
            x = (raw.clamp(0, 1) - self.mean) / self.std
        elif self.kind == 'blur':
            repeats = int(self.severity)
            for _ in range(repeats):
                x = torch.nn.functional.avg_pool2d(x[None], 3, 1, 1)[0]
        return x, y, name

def load_run(tag):
    result = json.loads((RESULTS / tag / 'result.json').read_text())
    model = build_additional_baseline(result['method'], base=result['base'])
    if model is None:
        model = BCSKGNet(base=result['base'], graph=result['method'] not in ['unet','attention','attention_original'],
                         attention='original' if result['method']=='attention_original' else result['method']=='attention')
    model.load_state_dict(torch.load(RESULTS / tag / 'model.pt', map_location='cpu'))
    model.eval(); return result, model

def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    p.add_argument('--mode', choices=['robustness','cross'], required=True)
    args = p.parse_args(); torch.set_num_threads(2)
    result, model = load_run(args.tag)
    threshold = result.get('threshold', .5)
    out = RESULTS / args.tag / (args.mode + '.json')
    if args.mode == 'cross':
        target = 'cvc' if result['dataset'] == 'kvasir' else 'kvasir'
        pairs = pairs_for(target)
        loader = DataLoader(CachedSeg(pairs, result['size']), batch_size=result['batch'])
        summary, _ = evaluate(model, loader, quick=True, threshold=threshold)
        payload = {'source': result['dataset'], 'target': target,
                   'target_n': len(pairs), 'dice': summary['dice']}
    else:
        pairs = pairs_for(result['dataset']); _, _, test = split_pairs(result['dataset'], pairs)
        clean = CachedSeg(test, result['size']); payload = {}
        conditions = [('clean',0),('noise',.05),('noise',.1),('noise',.2),
                      ('brightness',.6),('brightness',.8),('brightness',1.2),
                      ('blur',1),('blur',2)]
        for kind, severity in conditions:
            ds = clean if kind == 'clean' else Corrupted(clean, kind, severity)
            loader = DataLoader(ds, batch_size=result['batch'])
            summary, _ = evaluate(model, loader, quick=True, threshold=threshold)
            payload[f'{kind}_{severity}'] = summary['dice']
    out.write_text(json.dumps(payload, indent=2)); print(args.tag, args.mode, json.dumps(payload))

if __name__ == '__main__': main()
