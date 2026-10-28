"""Frozen-model mechanism, efficiency, graph, and boundary-operator diagnostics."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from scipy.ndimage import binary_erosion
from model import BCSKGNet, boundary_map
from baseline_models import build_additional_baseline
from experiment import ROOT,RESULTS,pairs_for,split_pairs,CachedSeg,evaluate
from ablation_experiment import select_operating_point

torch.set_num_threads(2)
tr,va,te=split_pairs('kvasir',pairs_for('kvasir'))
val=DataLoader(CachedSeg(va,64),batch_size=16);test=DataLoader(CachedSeg(te,64),batch_size=16)
mechanism=[]
for seed in [13,29,47]:
    tag=f'kvasir_bcskg_s{seed}_r0.05_b0.1';result=json.loads((RESULTS/tag/'result.json').read_text())
    model=BCSKGNet(base=8,graph=True);model.load_state_dict(torch.load(RESULTS/tag/'model.pt',map_location='cpu'));model.eval()
    record={'seed':seed,'gamma':result['graph_gate'],'selected_fusion':result['fusion_weight'],'selected':result['test'],'selected_threshold':result['threshold']}
    for mode,w in [('base_only',0.),('graph_only',1.)]:
        fusion,threshold=select_operating_point(model,val,fixed_fusion=w)
        mode_dir=RESULTS/f'{tag}_frozen_{mode}'
        summary,rows=evaluate(model,test,save_dir=mode_dir/'predictions',threshold=threshold)
        (mode_dir/'per_image.json').write_text(json.dumps(rows,indent=2))
        record[mode]={'fusion':fusion,'threshold':threshold,'test':summary}
    mechanism.append(record)
    (RESULTS/f'kvasir_bcskg_s{seed}_mechanism.json').write_text(json.dumps(record,indent=2))

# Profiler-counted MACs; all methods receive the same tensor and run on CPU.
efficiency=[]
for method,base in [('unet',8),('attention',8),('attention_original',8),('unetpp',8),('deeplab',8),('unetwide',10),('bcskg',8)]:
    graph=method=='bcskg';attention='original' if method=='attention_original' else method=='attention'
    model=build_additional_baseline(method,base=base)
    if model is None:
        model=BCSKGNet(base=base,graph=graph,attention=attention)
    model=model.eval();x=torch.rand(1,3,64,64)
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:model(x)
    flops=sum(e.flops for e in prof.key_averages() if e.flops is not None)
    efficiency.append({'method':method,'parameters':sum(p.numel() for p in model.parameters()),'gmacs_profiler':flops/2e9})

# Descriptive graph sparsity and membership quality on held-out predictions.
model=BCSKGNet(base=8,graph=True);model.load_state_dict(torch.load(RESULTS/'kvasir_bcskg_s13_r0.05_b0.1'/'model.pt',map_location='cpu'));model.eval()
graph_stats={'top5_mass':[],'effective_neighbors':[],'top5_same_class':[],'membership_dice':[]}
with torch.no_grad():
    for x,y,names in test:
        logits,aux=model(x);adj=torch.softmax(aux['relation_logits'],dim=-1)
        ent=-(adj*adj.clamp_min(1e-12).log()).sum(-1)
        graph_stats['effective_neighbors'].extend(torch.exp(ent).mean(1).tolist())
        top=torch.topk(adj,5,dim=-1).values
        graph_stats['top5_mass'].extend(top.sum(-1).mean(1).tolist())
        gt=torch.nn.functional.interpolate(y,size=(8,8),mode='area').flatten(1)>.5
        idx=adj.topk(5,dim=-1).indices
        for b in range(len(x)):
            same=(gt[b][idx[b]]==gt[b,:,None]).float().mean()
            graph_stats['top5_same_class'].append(float(same))
            pm=(aux['membership'][b,:,1].reshape(8,8)>.5);gm=gt[b].reshape(8,8)
            graph_stats['membership_dice'].append(float((2*(pm&gm).sum()+1e-7)/(pm.sum()+gm.sum()+1e-7)))
graph_stats={k:float(np.mean(v)) for k,v in graph_stats.items()}

# Agreement between the differentiable pooling boundary and morphological contours on validation targets.
boundary_agreement={str(t):[] for t in [.01,.05,.1,.2,.3,.5]}
for x,y,n in val:
    omega=boundary_map(y)
    morph=np.stack([((yy[0]>.5)&~binary_erosion(yy[0]>.5)) for yy in y.numpy()])
    morph=torch.from_numpy(morph).float()[:,None]
    for t in boundary_agreement:
        pr=omega>=float(t);dims=(1,2,3)
        vals=(2*(pr*morph).sum(dims)+1e-7)/(pr.sum(dims)+morph.sum(dims)+1e-7)
        boundary_agreement[t].extend(vals.numpy().tolist())
boundary_agreement={k:float(np.mean(v)) for k,v in boundary_agreement.items()}
payload={'mechanism':mechanism,'efficiency':efficiency,'graph_stats_seed13':graph_stats,'validation_boundary_operator_dice':boundary_agreement}
(RESULTS/'revision_diagnostics.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
