import numpy as np
import matplotlib.pyplot as plt
from figure_data import METHODS, SEEDS, SHORT, COL, FILL, RESULTS, result, read_json, save, style, tag
style()

fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.45))
variants = ['dual','graph_only','relation','boundary','nobranch','full','fullzero','fullgraph']
labels = ['A','B','C','D','E','Full',r'$\alpha=0$',r'$\alpha=1$']
def ablation(v,s):
    if v == 'full': return result('bcskg',s)
    if v == 'nobranch':
        return read_json(RESULTS/f'kvasir_bcskg_s{s}_r0.05_b0.1_branch0.0'/'result.json')
    if v in ('fullzero','fullgraph'):
        diagnostic=read_json(RESULTS/'revision_diagnostics.json')
        entry=next(x for x in diagnostic['mechanism'] if x['seed']==s)
        return {'test':entry['base_only' if v=='fullzero' else 'graph_only']['test']}
    return read_json(RESULTS/f'kvasir_ablation_{v}_s{s}_size64'/'result.json')
def boundary_weight(w):
    return result('bcskg',13) if w==.1 else read_json(RESULTS/f'kvasir_bcskg_s13_r0.05_b{float(w)}'/'result.json')

# (a) Components and frozen inference modes: bars with observed seed SD.
ax = axes[0, 0]
v=np.array([[ablation(a,s)['test']['dice'] for s in SEEDS] for a in variants])
colors=[FILL['unet'],FILL['attention'],FILL['attention_original'],FILL['unetpp'],FILL['deeplab'],FILL['bcskg'],FILL['unetwide'],FILL['unetwide']]
edges=[COL['unet'],COL['attention'],COL['attention_original'],COL['unetpp'],COL['deeplab'],COL['bcskg'],COL['unetwide'],COL['unetwide']]
ax.bar(np.arange(8),v.mean(1),color=colors,edgecolor=edges,linewidth=.9,yerr=v.std(1,ddof=1),capsize=2,error_kw={'lw':.8})
ax.set_xticks(np.arange(8),labels,fontsize=7)
ax.set_ylim(0,.84)
ax.set_ylabel('Test Dice')
ax.grid(axis='y',alpha=.45,linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) Boundary-loss sensitivity: seed 13 only.
ax = axes[0, 1]
x=np.array([0,.05,.1,.2]);y=[boundary_weight(z)['test']['dice'] for z in x]
ax.plot(x,y,'o-',color=COL['bcskg'],lw=1.7,ms=4,mec='white',mew=.7)
ax.axvline(.1,color=COL['unetwide'],ls='--',lw=.8,alpha=.75)
ax.set_xticks(x,['0','.05','.10','.20'])
ax.set_ylim(.68,.75)
ax.set_xlabel(r'Boundary weight $\lambda_b$')
ax.set_ylabel('Test Dice (seed 13)')
ax.grid(axis='y',alpha=.45,linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) Input-resolution check: two methods, seed 13.
ax = axes[1, 0]
for m in ('unet','bcskg'):
    z=[result(m,13) if n==64 else read_json(RESULTS/f'{tag(m,13)}_size128'/'result.json') for n in (64,128)]
    ax.plot([64,128],[r['test']['dice'] for r in z],marker='o',lw=1.7,ms=4,
            color=COL[m],mec='white',mew=.6,label=SHORT[m])
ax.set_xticks([64,128])
ax.set_ylim(.64,.76)
ax.set_xlabel('Input resolution (pixels)')
ax.set_ylabel('Test Dice (seed 13)')
ax.legend(frameon=False,loc='best')
ax.grid(axis='y',alpha=.45,linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (d) Frozen-model robustness across the eight predefined corruptions.
ax = axes[1, 1]
conditions=['noise_0.05','noise_0.1','noise_0.2','brightness_0.6',
            'brightness_0.8','brightness_1.2','blur_1','blur_2']
rob=[np.mean([read_json(RESULTS/tag(m,s)/'robustness.json')[k]
        for s in SEEDS for k in conditions]) for m in METHODS]
ax.bar(np.arange(7),rob,color=[FILL[m] for m in METHODS],
       edgecolor=[COL[m] for m in METHODS],linewidth=.9)
ax.set_xticks(np.arange(7),[SHORT[m] for m in METHODS],rotation=25,ha='right',fontsize=6.8)
ax.set_ylim(0,.75)
ax.set_ylabel('Mean Dice over corruptions')
ax.grid(axis='y',alpha=.45,linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ★ 带背景框子图标签（用 set_title 紧贴子图顶部）
titles=['(a) Components and inference','(b) Boundary weight',
        '(c) Resolution check','(d) Frozen-model robustness']
for ax,title in zip(axes.flat,titles):
    ax.set_title(title,fontsize=9,loc='left',pad=4)

fig.tight_layout(pad=.7,h_pad=1,w_pad=1)
save(fig,'fig11_protocol_diagnostics')
