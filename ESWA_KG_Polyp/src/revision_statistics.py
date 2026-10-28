"""Paired effect sizes, bootstrap intervals, and surface-distance exclusion audit."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import wilcoxon, rankdata
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
SEEDS=[13,29,47]
def tag(m,s):
    if m=='unetwide': return f'kvasir_unet_s{s}_r0.05_b0.1_base10'
    return f'kvasir_{m}_s{s}_r0.05_b0.1'
def rows(m,s):return json.loads((R/tag(m,s)/'per_image.json').read_text())
def rank_biserial(d):
    d=np.asarray(d);d=d[d!=0]
    if not len(d):return 0.
    ranks=rankdata(np.abs(d),method='average')
    return float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum())
def holm(ps):
    ps=np.asarray(ps);order=np.argsort(ps);out=np.empty(len(ps));running=0
    for rank,i in enumerate(order):
        running=max(running,(len(ps)-rank)*ps[i]);out[i]=min(1,running)
    return out
stats=[]
for base in ['unet','attention','attention_original','unetpp','deeplab','unetwide']:
    for s in SEEDS:
        f=R/tag(base,s)/'per_image.json'
        if not f.exists():continue
        a={r['id']:r['dice'] for r in rows(base,s)};b={r['id']:r['dice'] for r in rows('bcskg',s)}
        ids=sorted(a);d=np.array([b[i]-a[i] for i in ids]);offset={'unet':0,'attention':100,'attention_original':200,'unetpp':300,'deeplab':400,'unetwide':500}[base];rng=np.random.default_rng(20261024+s+offset)
        boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)
        stats.append({'baseline':base,'seed':s,'n':len(d),'n_nonzero':int(np.count_nonzero(d)),'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'ci95_mean_delta':[float(x) for x in np.percentile(boot,[2.5,97.5])],'rank_biserial':rank_biserial(d),'p_raw':float(wilcoxon(d).pvalue)})
for base in sorted(set(x['baseline'] for x in stats)):
    idx=[i for i,x in enumerate(stats) if x['baseline']==base]
    for i,p in zip(idx,holm([stats[i]['p_raw'] for i in idx])):stats[i]['p_holm_within_comparator']=float(p)
# Primary confirmatory family remains all six original comparator-seed tests.
idx=[i for i,x in enumerate(stats) if x['baseline'] in ['unet','attention']]
for i,p in zip(idx,holm([stats[i]['p_raw'] for i in idx])):stats[i]['p_holm_primary_family']=float(p)
exclusions=[]
for m in ['unet','attention','attention_original','unetpp','deeplab','unetwide','bcskg']:
    for s in SEEDS:
        f=R/tag(m,s)/'per_image.json'
        if not f.exists():continue
        rr=json.loads(f.read_text())
        exclusions.append({'method':m,'seed':s,'n':len(rr),'hd95_assd_excluded':sum(not np.isfinite(r['hd95']) for r in rr)})
sizes=[]
for p in sorted((ROOT/'data/Kvasir-SEG/images').glob('*')):
    with Image.open(p) as im:sizes.append(im.size)
a=np.asarray(sizes);scale={'median_width':float(np.median(a[:,0])),'median_height':float(np.median(a[:,1])),'two_resized_pixels_median_original_x':float(np.median(2*a[:,0]/64)),'two_resized_pixels_median_original_y':float(np.median(2*a[:,1]/64))}
(R/'revision_statistics.json').write_text(json.dumps({'paired':stats,'surface_exclusions':exclusions,'original_scale':scale,'bootstrap_replicates':10000,'primary_holm_family':'Six tests: BCRG versus U-Net and skip-attention U-Net for seeds 13, 29, and 47.'},indent=2))
print(json.dumps(stats,indent=2));print('exclusions',exclusions)
