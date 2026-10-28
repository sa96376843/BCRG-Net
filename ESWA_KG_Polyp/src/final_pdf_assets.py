"""Generate the final Kvasir-SEG-only PDF assets from completed 12-epoch runs."""
from pathlib import Path
import json, warnings, subprocess, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import wilcoxon
from scipy.ndimage import distance_transform_edt,binary_erosion
from PIL import Image
import torch
from experiment import ROOT,RESULTS,pairs_for,split_pairs,CachedSeg
from extended_evaluation import load_run
SEEDS=[13,29,47];METHODS=['unet','attention','attention_original','unetpp','deeplab','unetwide','bcskg'];ML={'unet':'U-Net','attention':'Skip-attention U-Net','attention_original':'Attention U-Net','unetpp':'UNet++','deeplab':'DeepLabV3+ (compact)','unetwide':'Wide U-Net','bcskg':'BCRG-Net'};COL={'unet':'#587A8A','attention':'#A47B45','attention_original':'#7B62A3','unetpp':'#B0657A','deeplab':'#527A3D','unetwide':'#8A6F55','bcskg':'#087F78'}
FIG=ROOT/'figures';TAB=ROOT/'tables';FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid',context='paper');plt.rcParams.update({'savefig.dpi':420,'figure.dpi':130,'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
def tag(m,s):
 if m=='unetwide':return f'kvasir_unet_s{s}_r0.05_b0.1_base10'
 return f'kvasir_{m}_s{s}_r0.05_b0.1'
def load(m,s):
 r=json.loads((RESULTS/tag(m,s)/'result.json').read_text());assert r['epochs']==12;return r
def rows(m,s):return json.loads((RESULTS/tag(m,s)/'per_image.json').read_text())
def save(fig,n):fig.savefig(FIG/(n+'.png'),bbox_inches='tight',facecolor='white');plt.close(fig)
def ms(v):return f'{np.mean(v):.3f} $\\pm$ {np.std(v,ddof=1):.3f}'
R={(m,s):load(m,s) for m in METHODS for s in SEEDS}
# Data landscape
pairs=pairs_for('kvasir');tr,va,te=split_pairs('kvasir',pairs)
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for items,label,c in [(tr,'Train','#8AB6C9'),(va,'Validation','#E2B66B'),(te,'Test','#367C78')]:
 a=[]
 for _,mp in items:a.append((np.asarray(Image.open(mp).convert('L').resize((64,64),Image.Resampling.NEAREST))>127).mean())
 ax[0].hist(a,bins=np.linspace(0,1,26),histtype='step',density=True,lw=1.8,label=label,color=c)
 a=np.sort(a);ax[1].plot(a,np.linspace(0,1,len(a)),lw=2,label=label,color=c)
ax[0].set(xlabel='Foreground area fraction',ylabel='Density',title='Mask-area histograms');ax[1].set(xlabel='Foreground area fraction',ylabel='Empirical CDF',title='Split coverage');ax[0].legend(frameon=False);fig.tight_layout();save(fig,'fig03_dataset_landscape')
# Training curves
fig,ax=plt.subplots(figsize=(5.2,3.6))
for m in METHODS:
 v=np.array([[e['val_dice'] for e in R[m,s]['history']] for s in SEEDS]);x=np.arange(1,13)
 ax.plot(x,v.mean(0),lw=2.2,marker='o',ms=3,label=ML[m],color=COL[m]);ax.fill_between(x,v.min(0),v.max(0),color=COL[m],alpha=.14)
ax.set(xlabel='Epoch',ylabel='Validation Dice',ylim=(0,0.8),title='Validation learning dynamics across three seeds');ax.legend(frameon=False);fig.tight_layout();save(fig,'fig04_training_dynamics')
# Per-image distributions
long=[]
for m in METHODS:
 for s in SEEDS:
  for r in rows(m,s):long.append({'method':m,'seed':s,'id':r['id'],'dice':r['dice'],'area':r['area']})
per=pd.DataFrame(long);per.to_csv(RESULTS/'final_per_image_long.csv',index=False)
fig,ax=plt.subplots(figsize=(5.2,3.7))
sns.violinplot(data=per,x='method',y='dice',hue='method',palette=COL,inner=None,cut=0,alpha=.3,legend=False,ax=ax)
sns.boxplot(data=per,x='method',y='dice',width=.32,showfliers=False,boxprops={'facecolor':'white'},ax=ax)
ax.set_xticks(range(len(METHODS)),['U-Net','Skip-att.','Attention','UNet++','DLabV3+','Wide','BCRG'],rotation=18);ax.set(xlabel='',ylabel='Per-image test Dice',ylim=(-.03,1.03),title='Case-level overlap distributions');fig.tight_layout();save(fig,'fig05_per_image_dice')
# Size strata
ref=per[per.method=='unet'][['id','area']].drop_duplicates('id');ref['bin']=pd.qcut(ref.area,3,labels=['Small','Medium','Large']);q=per.merge(ref[['id','bin']],on='id')
fig,ax=plt.subplots(figsize=(5.2,3.6))
sns.pointplot(data=q,x='bin',y='dice',hue='method',order=['Small','Medium','Large'],palette=COL,errorbar='sd',capsize=.13,ax=ax)
ax.set(xlabel='Target-area tertile',ylabel='Per-image Dice',ylim=(0,1),title='Performance across target scales');handles,labels=ax.get_legend_handles_labels();ax.legend(handles,[ML.get(x,x) for x in labels],frameon=False,fontsize=8,title=None);fig.tight_layout();save(fig,'fig06_size_stratified')
# Boundary tolerance
def bts(pred,gt,delta):
 pb=pred&~binary_erosion(pred);gb=gt&~binary_erosion(gt)
 if pb.sum()==0 or gb.sum()==0:return np.nan
 pd=distance_transform_edt(~pb);gd=distance_transform_edt(~gb)
 return ((pb&(gd<=delta)).sum()+(gb&(pd<=delta)).sum())/(pb.sum()+gb.sum()+1e-7)
fig,ax=plt.subplots(figsize=(5.2,3.6))
for m in METHODS:
 vals=[]
 for delta in range(7):
  vv=[]
  for s in SEEDS:
   th=R[m,s]['threshold']
   for f in (RESULTS/tag(m,s)/'predictions').glob('*.npz'):
    z=np.load(f);vv.append(bts(z['prob']>=th,z['gt']>.5,delta))
  vals.append(np.nanmean(vv))
 ax.plot(range(7),vals,marker='o',ms=4,lw=2,label=ML[m],color=COL[m])
ax.set(xlabel='Boundary tolerance (pixels)',ylabel='BTS',ylim=(0,1),title='Boundary precision--coverage across tolerances');ax.legend(frameon=False);fig.tight_layout();save(fig,'fig07_boundary_tolerance')
# Paired differences and statistics
stats=[];fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,base in zip(axes,['unet','attention']):
 diffs=[];ps=[]
 for s in SEEDS:
  a={r['id']:r['dice'] for r in rows(base,s)};b={r['id']:r['dice'] for r in rows('bcskg',s)};ids=sorted(a)
  d=np.array([b[i]-a[i] for i in ids]);diffs.extend(d);p=wilcoxon(d).pvalue;ps.append(p)
  stats.append({'baseline':base,'seed':s,'n':len(d),'mean_delta':float(d.mean()),'median_delta':float(np.median(d)),'p_raw':float(p)})
 order=np.argsort(ps);adj=np.empty(3);last=0
 for rank,i in enumerate(order):last=max(last,(3-rank)*ps[i]);adj[i]=min(1,last)
 for row,a in zip(stats[-3:],adj):row['p_holm']=float(a)
 ax.hist(diffs,bins=35,color='#3D8E88',alpha=.75);ax.axvline(0,color='black',lw=1);ax.axvline(np.mean(diffs),color='#B7503C',ls='--',lw=1.5,label=f'Mean = {np.mean(diffs):.3f}')
 ax.set(xlabel='BCRG-Net $-$ comparator Dice',ylabel='Case count',title='vs. '+ML[base]);ax.legend(frameon=False)
fig.tight_layout();save(fig,'fig08_paired_differences');(RESULTS/'statistical_tests.json').write_text(json.dumps(stats,indent=2))
# Qualitative outcomes
u={r['id']:r for r in rows('unet',13)};b={r['id']:r for r in rows('bcskg',13)};ranked=sorted(u,key=lambda n:b[n]['dice']-u[n]['dice']);chosen=[ranked[0],ranked[-1],min(u,key=lambda n:b[n]['dice'])]
fig,axes=plt.subplots(3,5,figsize=(12,7.2))
for i,n in enumerate(chosen):
 pair=next(p for p in pairs if p[0].stem==n);im=Image.open(pair[0]).convert('RGB');axes[i,0].imshow(im);axes[i,0].set_title('Input')
 z=np.load(RESULTS/tag('bcskg',13)/'predictions'/(n+'.npz'));g=z['gt']>.5;axes[i,1].imshow(g,cmap='gray');axes[i,1].set_title('Ground truth')
 preds=[]
 for j,m in enumerate(['unet','bcskg']):
  z=np.load(RESULTS/tag(m,13)/'predictions'/(n+'.npz'));pr=z['prob']>=R[m,13]['threshold'];preds.append(pr);axes[i,j+2].imshow(pr,cmap='gray');axes[i,j+2].set_title(ML[m])
 err=np.ones((*g.shape,3));err[preds[1]&~g]=[1,.25,.15];err[g&~preds[1]]=[.1,.35,.85];err[preds[1]&g]=[.1,.65,.45];axes[i,4].imshow(err);axes[i,4].set_title('BCRG error map')
 axes[i,0].set_ylabel(['Largest loss','Largest gain','Lowest Dice'][i])
for ax in axes.ravel():ax.set_xticks([]);ax.set_yticks([])
fig.tight_layout();save(fig,'fig09_qualitative')
# Actual graph evidence
result,model=load_run(tag('bcskg',13));x,y,n=CachedSeg(te[:1],64)[0]
with torch.no_grad():logits,aux=model(x[None])
adj=torch.softmax(aux['relation_logits'][0],dim=-1).numpy();mem=aux['membership'][0].numpy().reshape(8,8,2)
fig,axes=plt.subplots(1,4,figsize=(12,3.2));axes[0].imshow(Image.open(te[0][0]));axes[0].set_title('Input')
im=axes[1].imshow(mem[...,1],cmap='magma');axes[1].set_title('Predicted membership');fig.colorbar(im,ax=axes[1],fraction=.046)
im=axes[2].imshow(adj,cmap='viridis');axes[2].set_title('Learned relation matrix');fig.colorbar(im,ax=axes[2],fraction=.046)
coords=np.array([(i%8,i//8) for i in range(64)]);axes[3].scatter(coords[:,0],coords[:,1],c=mem[...,1].ravel(),cmap='magma',s=50);rng=np.random.default_rng(7)
for a in rng.choice(64,12,replace=False):
 for bb in np.argsort(adj[a])[-2:]:
  if a!=bb:axes[3].plot([coords[a,0],coords[bb,0]],[coords[a,1],coords[bb,1]],alpha=min(1,adj[a,bb]*30),color='#246B8E')
axes[3].invert_yaxis();axes[3].set_title('Strong region links');axes[3].set_aspect('equal')
for ax in [axes[0],axes[3]]:ax.set_xticks([]);ax.set_yticks([])
fig.tight_layout();save(fig,'fig10_graph_evidence')
# The consolidated table generator also handles bounded archived ASSD values.
subprocess.run([sys.executable,str(Path(__file__).with_name('revision_assets.py')),'--tables-only'],check=True)
lines=[r'\begin{table}[t]\centering\caption{Per-seed results and validation-selected operating points.}\label{tab:perseed}\small',r'\begin{tabular}{llcccc}\toprule',r'Method & Seed & Dice & IoU & Threshold & Fusion weight\\\midrule']
for m in METHODS:
 for s in SEEDS:
  r=R[m,s];lines.append(f"{ML[m]} & {s} & {r['test']['dice']:.3f} & {r['test']['iou']:.3f} & {r['threshold']:.2f} & {r.get('fusion_weight') if r.get('fusion_weight') is not None else '--'}\\\\")
lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'per_seed.tex').write_text('\n'.join(lines))
lines=[r'\begin{table}[t]\centering\caption{Paired per-image Wilcoxon signed-rank tests for BCRG-Net minus each comparator. Holm correction is applied across the six tests.}\label{tab:stats}\small',r'\begin{tabular}{lrrrrr}\toprule',r'Comparator & Seed & Mean $\Delta$ & Median $\Delta$ & Raw $p$ & Holm $p$\\\midrule']
allp=[r['p_raw'] for r in stats];order=np.argsort(allp);adj=np.empty(len(allp));last=0
for rank,i in enumerate(order):last=max(last,(len(allp)-rank)*allp[i]);adj[i]=min(1,last)
for r,a in zip(stats,adj):r['p_holm']=float(a);lines.append(f"{ML[r['baseline']]} & {r['seed']} & {r['mean_delta']:.4f} & {r['median_delta']:.4f} & {r['p_raw']:.3g} & {a:.3g}\\\\")
(RESULTS/'statistical_tests.json').write_text(json.dumps(stats,indent=2));lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'statistics.tex').write_text('\n'.join(lines))
lines=[r'\begin{table}[t]\centering\caption{Measured model cost and learned graph gates. CPU latency is one $64\times64$ forward pass after warm-up.}\label{tab:efficiency}\small',r'\begin{tabular}{lrrr}\toprule',r'Method & Parameters & Latency (ms) & Graph gate $\gamma$\\\midrule']
for m in METHODS:
 lines.append(f"{ML[m]} & {np.mean([R[m,s]['params'] for s in SEEDS]):,.0f} & {np.mean([R[m,s]['latency_ms'] for s in SEEDS]):.2f} & {np.mean([R[m,s]['graph_gate'] for s in SEEDS]):.3f}"+r'\\' if m=='bcskg' else f"{ML[m]} & {np.mean([R[m,s]['params'] for s in SEEDS]):,.0f} & {np.mean([R[m,s]['latency_ms'] for s in SEEDS]):.2f} & --"+r'\\')
lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'efficiency.tex').write_text('\n'.join(lines))
print('Figures',len(list(FIG.glob('fig*.png'))),'tables',len(list(TAB.glob('*.tex'))))
for m in METHODS:print(m,ms([R[m,s]['test']['dice'] for s in SEEDS]))
