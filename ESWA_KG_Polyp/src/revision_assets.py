"""Generate three consolidated cross-column tables and a diagnostic figure."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
R, F, T = ROOT/'results', ROOT/'figures', ROOT/'tables'
SEEDS = [13, 29, 47]
METHODS = ['unet', 'attention', 'attention_original', 'unetpp', 'deeplab', 'unetwide', 'bcskg']
ML = {'unet':'U-Net', 'attention':'Skip-attention U-Net',
      'attention_original':'Attention U-Net', 'unetpp':'UNet++',
      'deeplab':'Compact DeepLabV3+', 'unetwide':'Wide U-Net', 'bcskg':'BCRG-Net'}
SHORT = {'unet':'U-Net', 'attention':'Skip-att.', 'attention_original':'Attention',
         'unetpp':'UNet++', 'deeplab':'DLabV3+', 'unetwide':'Wide U-Net', 'bcskg':'BCRG-Net'}
VL = {'dual':'A: dual decoder', 'graph_only':'B: + graph residual',
      'relation':'C: + relation loss', 'boundary':'D: + boundary loss',
      'nobranch':'E: Full w/o branch sup.', 'full':'Full model',
      'fullzero':r'Full, inference $\alpha=0$', 'fullgraph':r'Full, inference $\alpha=1$'}

def load_json(p):
    return json.loads(Path(p).read_text())

def tag(m, s):
    if m == 'unetwide':
        return f'kvasir_unet_s{s}_r0.05_b0.1_base10'
    return f'kvasir_{m}_s{s}_r0.05_b0.1'

def get(m, s):
    return load_json(R/tag(m, s)/'result.json')

def ab(v, s):
    return load_json(R/f'kvasir_ablation_{v}_s{s}_size64'/'result.json')

def ms(v):
    return f'{np.mean(v):.3f} $\\pm$ {np.std(v,ddof=1):.3f}'

def metric_bounds(rows, key):
    bounds=[]
    for row in rows:
        value=row['test'][key]
        if value is None:
            if key not in row.get('surface_metric_bounds',{}):
                return None
            bounds.append(row['surface_metric_bounds'][key])
        else:
            bounds.append([value,value])
    return np.asarray(bounds,dtype=float)

def metric_summary(rows, key):
    bounds=metric_bounds(rows,key)
    if bounds is None:
        return '--'
    mean_bounds=bounds.mean(axis=0)
    mid=bounds.mean(axis=1)
    radius=(bounds[:,1]-bounds[:,0])/2
    # Sample SD is ||center(x)||/sqrt(n-1); this norm bound covers every
    # compatible archived mask, without choosing an ambiguous prediction.
    sd=np.std(mid,ddof=1)
    error=np.linalg.norm(radius)/np.sqrt(len(mid)-1)
    sd_bounds=[max(0.,sd-error),sd+error]
    assert f'{mean_bounds[0]:.3f}'==f'{mean_bounds[1]:.3f}', (key,mean_bounds)
    assert f'{sd_bounds[0]:.3f}'==f'{sd_bounds[1]:.3f}', (key,sd_bounds)
    return f'{mean_bounds[0]:.3f} $\\pm$ {sd_bounds[0]:.3f}'

def save(fig, name):
    fig.savefig(F/(name+'.png'), dpi=420, bbox_inches='tight', facecolor='white')
    plt.close(fig)

sns.set_theme(style='whitegrid', context='paper')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.titlesize':11, 'axes.labelsize':10,
                     'xtick.labelsize':9, 'ytick.labelsize':9,
                     'legend.fontsize':9, 'axes.spines.top':False,
                     'axes.spines.right':False})
diag = load_json(R/'revision_diagnostics.json')
stats = load_json(R/'revision_statistics.json')
eff = {x['method']:x for x in diag['efficiency']}
benchmark = {x['method']:x for x in load_json(R/'expanded_efficiency_benchmark.json')}
keys = ['dice','iou','hd95','assd','biou']
ranks={}
for k in keys:
    intervals={m:metric_bounds([get(m,s) for s in SEEDS],k).mean(axis=0) for m in METHODS}
    ordered=sorted(METHODS,key=lambda m:intervals[m][0])
    assert all(intervals[a][1]<intervals[b][0] for a,b in zip(ordered,ordered[1:])), (k,intervals)
    ranks[k]=ordered if k in ['hd95','assd'] else ordered[::-1]

# Table 1: quality, efficiency, paired inference, and target-scale evidence.
lines = [r'\begin{table*}[t]\centering',
 r'\caption{Consolidated comparison, statistical analysis, and target-scale behavior on Kvasir-SEG. Panel A reports mean $\pm$ sample standard deviation over seeds 13/29/47; its seed column lists individual Dice scores. HD95/ASSD are pixels at $64\times64$, and Excl. lists empty-mask exclusions for those metrics. Panel B reports BCRG-Net minus comparator paired Dice differences with within-comparator Holm-corrected Wilcoxon $p$-values (149 paired images per test). Panel C averages all case-seed Dice values within test-target-area tertiles. MACs are profiler-counted; CPU latency excludes preprocessing. Best means in Panel A are bold and second-best means underlined.}\label{tab:main}',
 r'\textbf{Panel A: segmentation quality, seed detail, computational cost, and exclusions}\\[2pt]',
 r'\resizebox{\textwidth}{!}{%',
 r'\begin{tabular}{lcccccccccc}\toprule',
 r'Method & Dice $\uparrow$ & IoU $\uparrow$ & HD95 $\downarrow$ & ASSD $\downarrow$ & BTS $\uparrow$ & Dice by seed & Params & GMACs & Latency (ms) & Excl.\\\midrule']
for m in METHODS:
    cells = []
    for k in keys:
        txt = metric_summary([get(m,s) for s in SEEDS],k)
        if m==ranks[k][0]:
            txt = r'\textbf{'+txt+'}'
        elif m==ranks[k][1]:
            txt = r'\underline{'+txt+'}'
        cells.append(txt)
    seed_dice = '/'.join(f"{get(m,s)['test']['dice']:.3f}" for s in SEEDS)
    excluded = [next(x['hd95_assd_excluded'] for x in stats['surface_exclusions']
                     if x['method']==m and x['seed']==s) for s in SEEDS]
    latency = benchmark[m]['latency_ms_mean']
    lines.append(ML[m]+' & '+' & '.join(cells)+
                 f" & {seed_dice} & {eff[m]['parameters']/1000:.1f}k & {eff[m]['gmacs_profiler']:.3f} & {latency:.2f} & "+'/'.join(map(str,excluded))+r'\\')
lines += [r'\bottomrule\end{tabular}}',r'\par\vspace{5pt}',
 r'\textbf{Panel B: paired per-image comparison with BCRG-Net}\\[2pt]',
 r'\resizebox{.86\textwidth}{!}{%',
 r'\begin{tabular}{lcccccc}\toprule',
 r'Comparator & $\Delta_{13}$ & Holm $p_{13}$ & $\Delta_{29}$ & Holm $p_{29}$ & $\Delta_{47}$ & Holm $p_{47}$\\\midrule']
for m in METHODS[:-1]:
    rr = [next(x for x in stats['paired'] if x['baseline']==m and x['seed']==s) for s in SEEDS]
    cells = []
    for x in rr:
        cells += [f"{x['mean_delta']:+.4f}", f"{x['p_holm_within_comparator']:.3g}"]
    lines.append(ML[m]+' & '+' & '.join(cells)+r'\\')
lines += [r'\bottomrule\end{tabular}}',r'\par\vspace{5pt}',
 r'\textbf{Panel C: mean per-image Dice by target-area tertile}\\[2pt]',
 r'\resizebox{.62\textwidth}{!}{%',
 r'\begin{tabular}{lccc}\toprule',r'Method & Small & Medium & Large\\\midrule']
def image_rows(m,s):
    return load_json(R/tag(m,s)/'per_image.json')
ref = pd.DataFrame(image_rows('unet',13))[['id','area']].drop_duplicates('id')
ref['bin'] = pd.qcut(ref.area,3,labels=['Small','Medium','Large'])
area_bin = dict(zip(ref['id'],ref['bin']))
for m in METHODS:
    vals = {b:[] for b in ['Small','Medium','Large']}
    for s in SEEDS:
        for x in image_rows(m,s): vals[area_bin[x['id']]].append(x['dice'])
    lines.append(ML[m]+' & '+' & '.join(f"{np.mean(vals[b]):.3f}" for b in ['Small','Medium','Large'])+r'\\')
lines += [r'\bottomrule\end{tabular}}',r'\end{table*}']
(T/'main_results.tex').write_text('\n'.join(lines))

# Table 2: cumulative training ablations plus frozen inference-mode controls.
variants = ['dual','graph_only','relation','boundary','nobranch','full','fullzero','fullgraph']
data = {}
for v in variants:
    rows = []
    for s in SEEDS:
        if v == 'full':
            r = get('bcskg',s)
        elif v in ['fullzero','fullgraph']:
            mode = 'base_only' if v == 'fullzero' else 'graph_only'
            r = {'test':next(x for x in diag['mechanism'] if x['seed']==s)[mode]['test']}
        elif v == 'nobranch':
            r = load_json(R/f'kvasir_bcskg_s{s}_r0.05_b0.1_branch0.0'/'result.json')
        else:
            r = ab(v,s)
        rows.append(r)
    data[v] = rows
lines = [r'\begin{table*}[t]\centering',
 r'\caption{Component, supervision, and inference-mode analysis under the same 12-epoch budget. A--D are cumulative retrained variants; E removes dual-branch supervision while retaining all graph and calibration losses; Full adds auxiliary-mask calibration and branch supervision. The last two rows reuse each frozen Full model, fix graph-decoder fusion, and reselect only the validation threshold. Scores are mean $\pm$ sample standard deviation over three seeds; individual Dice values are retained in the last column. HD95 is unavailable for the two frozen inference controls because their separate prediction maps were not archived.}\label{tab:ablation}',
 r'\resizebox{\textwidth}{!}{%',
 r'\begin{tabular}{lcccccccccc}\toprule',
 r'Variant & Graph & Rel. & Bound. & Aux. cal. & Branch sup. & Dice $\uparrow$ & IoU $\uparrow$ & HD95 $\downarrow$ & BTS $\uparrow$ & Dice by seed\\\midrule']
for v in variants:
    checks = ['--']*5
    if v != 'dual': checks[0] = r'\checkmark'
    if v in ['relation','boundary','nobranch','full','fullzero','fullgraph']: checks[1] = r'\checkmark'
    if v in ['boundary','nobranch','full','fullzero','fullgraph']: checks[2] = r'\checkmark'
    if v in ['nobranch','full','fullzero','fullgraph']: checks[3] = r'\checkmark'
    if v != 'nobranch': checks[4] = r'\checkmark'
    seed_dice = '/'.join(f"{r['test']['dice']:.3f}" for r in data[v])
    lines.append(VL[v]+' & '+' & '.join(checks)+' & '+
                 ' & '.join(metric_summary(data[v],k) for k in ['dice','iou','hd95','biou'])+
                 ' & '+seed_dice+r'\\')
lines += [r'\bottomrule\end{tabular}}',r'\end{table*}']
(T/'ablation.tex').write_text('\n'.join(lines))

# Frozen-model robustness for every comparator.
rob = {}
conditions = [('Clean','clean_0'), ('Noise .05','noise_0.05'),
              ('Noise .10','noise_0.1'), ('Noise .20','noise_0.2'),
              ('Bright .60','brightness_0.6'), ('Bright .80','brightness_0.8'),
              ('Bright 1.20','brightness_1.2'), ('Blur 1','blur_1'), ('Blur 2','blur_2')]
for m in METHODS:
    rob[m] = load_json(R/tag(m,13)/'robustness.json')
    for s in SEEDS:
        assert set(load_json(R/tag(m,s)/'robustness.json')) == set(rob[m])
rob_mean = {m:{key:np.mean([load_json(R/tag(m,s)/'robustness.json')[key] for s in SEEDS])
               for _,key in conditions} for m in METHODS}
lines = [r'\begin{table*}[t]\centering',
 r'\caption{Robustness, loss-weight sensitivity, and resolution checks. Panel A reports mean test Dice over the same three seeds using frozen weights and validation-selected thresholds; no corruption is applied during training. Mean loss is the average percentage Dice reduction over the eight corruptions relative to each run\'s clean score. Panel B contains single-seed exploratory studies (seed 13), not confirmatory model selection.}\label{tab:robustness}',
 r'\textbf{Panel A: deterministic test-time corruptions (Dice)}\\[2pt]',
 r'\resizebox{\textwidth}{!}{%',
 r'\begin{tabular}{lcccccccccc}\toprule',
 r'Method & Clean & Noise .05 & Noise .10 & Noise .20 & Bright .60 & Bright .80 & Bright 1.20 & Blur 1 & Blur 2 & Mean loss (\%)\\\midrule']
for m in METHODS:
    vals = [rob_mean[m][key] for _,key in conditions]
    losses = []
    for s in SEEDS:
        rr = load_json(R/tag(m,s)/'robustness.json')
        losses.extend(100*(rr['clean_0']-rr[key])/max(rr['clean_0'],1e-12) for _,key in conditions[1:])
    cells = []
    for j,v in enumerate(vals):
        best = max(rob_mean[x][conditions[j][1]] for x in METHODS)
        cells.append(r'\textbf{'+f'{v:.3f}'+'}' if abs(v-best)<1e-12 else f'{v:.3f}')
    lines.append(ML[m]+' & '+' & '.join(cells)+f' & {np.mean(losses):.1f}'+r'\\')
lines += [r'\bottomrule\end{tabular}}',r'\par\vspace{5pt}',
 r'\textbf{Panel B: boundary-loss weight and input resolution (seed 13)}\\[2pt]',
 r'\begin{tabular}{llcc}\toprule',r'Study & Setting & Dice & IoU\\\midrule']
sens = []
for b in [0,0.05,0.1,0.2]:
    rr = get('bcskg',13) if b==.1 else load_json(R/f'kvasir_bcskg_s13_r0.05_b{float(b)}'/'result.json')
    sens.append((b,rr))
    lines.append(f"Boundary weight & $\\lambda_b={b:g}$ & {rr['test']['dice']:.3f} & {rr['test']['iou']:.3f}"+r'\\')
res = {}
for m in ['unet','bcskg']:
    for z in [64,128]:
        rr = get(m,13) if z==64 else load_json(R/f'{tag(m,13)}_size128'/'result.json')
        res[(m,z)] = rr
        lines.append(f"Resolution & {ML[m]}, ${z}\\times{z}$ & {rr['test']['dice']:.3f} & {rr['test']['iou']:.3f}"+r'\\')
lines += [r'\bottomrule\end{tabular}',r'\end{table*}']
(T/'robustness.tex').write_text('\n'.join(lines))

if '--tables-only' in sys.argv:
    print('Three tables regenerated; all rounded summaries and rankings certified against metric bounds.')
    sys.exit(0)

# One figure summarizes ablation, sensitivity, resolution, and robustness.
fig, axes = plt.subplots(2,2,figsize=(10,6.6))
show = ['dual','graph_only','relation','boundary','nobranch','full','fullzero','fullgraph']
x = np.arange(len(show))
d = [np.mean([r['test']['dice'] for r in data[v]]) for v in show]
sd = [np.std([r['test']['dice'] for r in data[v]],ddof=1) for v in show]
axes[0,0].bar(x,d,yerr=sd,capsize=3,color=['#8BA6B2','#7B9FA6','#5A9D94','#2F8C82','#B28B5E','#087F78','#D09B4A','#A87C45'])
axes[0,0].set_xticks(x,['A','B','C','D','E','Full',r'$\alpha=0$',r'$\alpha=1$'])
axes[0,0].set(ylabel='Test Dice',ylim=(0,1),title='(a) Components and inference modes')
axes[0,1].plot([b for b,_ in sens],[r['test']['dice'] for _,r in sens],marker='o',lw=2,color='#087F78')
axes[0,1].set(xlabel=r'Boundary-loss weight $\lambda_b$',ylabel='Test Dice (seed 13)',ylim=(0,1),title='(b) Loss-weight sensitivity')
for m,c in [('unet','#587A8A'),('bcskg','#087F78')]:
    axes[1,0].plot([64,128],[res[m,z]['test']['dice'] for z in [64,128]],marker='o',lw=2,label=ML[m],color=c)
axes[1,0].set_xticks([64,128])
axes[1,0].set(xlabel='Input resolution',ylabel='Test Dice (seed 13)',ylim=(0,1),title='(c) Exploratory resolution check')
axes[1,0].legend(frameon=False)
colors = ['#587A8A','#A47B45','#7B62A3','#B0657A','#527A3D','#8A6F55','#087F78']
avg_corrupt = [np.mean([rob_mean[m][key] for _,key in conditions[1:]]) for m in METHODS]
axes[1,1].bar(np.arange(len(METHODS)),avg_corrupt,color=colors)
axes[1,1].set_xticks(np.arange(len(METHODS)),[SHORT[m] for m in METHODS],rotation=18)
axes[1,1].set(ylabel='Mean Dice over corruptions',ylim=(0,1),title='(d) Frozen-model robustness')
fig.tight_layout()
save(fig,'fig11_protocol_diagnostics')
print('Main Dice:',{m:ms([get(m,s)['test']['dice'] for s in SEEDS]) for m in METHODS})
print('Ablation Dice:',{v:ms([r['test']['dice'] for r in data[v]]) for v in variants})
print('Graph diagnostics:',diag['graph_stats_seed13'])
print('Boundary agreement:',diag['validation_boundary_operator_dice'])
