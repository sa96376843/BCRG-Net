"""Deterministic redraw of the six remaining empirical figures."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from figure_data import (PROJECT, DATA, RESULTS, OUTPUT, SEEDS, METHODS, NAME,
                         SHORT, COL, FILL, MARK, tag, result, rows, save, style)

style()

def ground_truth(sample_id, shape):
    path=DATA/'Kvasir-SEG'/'masks'/(sample_id+'.jpg')
    assert path.exists(),path
    return np.asarray(Image.open(path).convert('L').resize(shape[::-1],Image.Resampling.NEAREST))>127

def grid(ax):
    ax.grid(alpha=.45, linestyle='--', axis='y')
    ax.set_axisbelow(True)

def dataset_landscape():
    paths={'train':[],'validation':[],'test':[]}
    with (RESULTS/'split_manifest.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['dataset']=='kvasir':
                paths[{'train':'train','val':'validation','validation':'validation','test':'test'}[r['split']]].append(PROJECT/r['mask'])
    assert [len(paths[k]) for k in paths]==[713,138,149]
    area={k:np.array([(np.asarray(Image.open(p).convert('L').resize((64,64),Image.Resampling.NEAREST))>127).mean()
                      for p in v]) for k,v in paths.items()}
    fig,axes=plt.subplots(1,2,figsize=(6.85,2.35))
    cmap={'train':COL['bcskg'],'validation':COL['unetwide'],'test':COL['unet']}
    for k in paths:
        v=area[k]; name=f'{k.title()} (n={len(v)})'
        axes[0].hist(v,bins=np.linspace(0,1,26),histtype='step',density=True,
                     lw=1.65,label=name,color=cmap[k])
        x=np.sort(v);axes[1].step(x,np.arange(1,len(x)+1)/len(x),where='post',
                                 lw=1.65,label=name,color=cmap[k])
    axes[0].set(xlabel='Foreground area fraction',ylabel='Density',xlim=(0,1))
    axes[1].set(xlabel='Foreground area fraction',ylabel='Empirical CDF',xlim=(0,1),ylim=(0,1))
    for ax in axes:grid(ax)
    axes[0].legend(frameon=False,fontsize=7.4)
    fig.tight_layout(pad=.5,w_pad=1)
    save(fig,'fig03_dataset_landscape')

def size_stratified():
    long=[{'method':m,'seed':s,'id':r['id'],'dice':r['dice'],'area':r['area']}
          for m in METHODS for s in SEEDS for r in rows(m,s)]
    per=pd.DataFrame(long)
    ref=per[per.method=='unet'][['id','area']].drop_duplicates('id')
    assert len(ref)==149
    ref['bin']=pd.qcut(ref.area,3,labels=['Small','Medium','Large'])
    q=per.merge(ref[['id','bin']],on='id')
    means=np.empty((7,3));sd=np.empty((7,3))
    for i,m in enumerate(METHODS):
        for j,b in enumerate(('Small','Medium','Large')):
            v=q[(q.method==m)&(q.bin==b)].dice.to_numpy()
            means[i,j]=v.mean();sd[i,j]=v.std(ddof=1)
    cmap=LinearSegmentedColormap.from_list('dice_green', ['#F7FBF5','#C7E0D1','#5E8F6A'])
    fig,ax=plt.subplots(figsize=(3.43,2.85))
    img=ax.imshow(means,cmap=cmap,vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(3),['Small','Medium','Large'])
    ax.set_yticks(range(7),[SHORT[m] for m in METHODS],fontsize=7.2)
    ax.set_xlabel('Target-area tertile')
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,7,1),minor=True)
    ax.grid(which='minor',color='white',lw=1.3)
    ax.tick_params(which='minor',bottom=False,left=False)
    for i in range(7):
        for j in range(3):
            ax.text(j,i,f'{means[i,j]:.3f}\n±{sd[i,j]:.2f}',ha='center',va='center',
                    fontsize=6.8,color='#2F4F3A')
    ax.get_yticklabels()[-1].set_color(COL['bcskg'])
    fig.colorbar(img,ax=ax,fraction=.035,pad=.025,label='Mean Dice')
    fig.tight_layout(pad=.3)
    save(fig,'fig06_size_stratified')

def boundary_tolerance():
    from scipy.ndimage import binary_erosion,distance_transform_edt
    x=np.arange(7);curves={};missing={}
    for m in METHODS:
        seed_means=[]
        for s in SEEDS:
            values=[]
            for f in sorted((RESULTS/tag(m,s)/'predictions').glob('*.npz')):
                with np.load(f) as z:
                    pr=z['prob']>=result(m,s)['threshold']
                    gt=ground_truth(f.stem,pr.shape)
                    if 'gt' in z: assert np.array_equal(gt,z['gt']>.5),f
                pb=pr&~binary_erosion(pr);gb=gt&~binary_erosion(gt)
                if not pb.any() or not gb.any(): continue
                d1=distance_transform_edt(~gb)[pb]
                d2=distance_transform_edt(~pb)[gb]
                values.append([(np.count_nonzero(d1<=d)+np.count_nonzero(d2<=d)) /
                               (len(d1)+len(d2)+1e-7) for d in x])
            a=np.asarray(values);assert a.shape[1]==7
            mean=a.mean(axis=0)
            # The published run-level BTS includes empty predictions as zero;
            # this tolerance diagnostic follows the original figure and excludes them.
            assert np.isfinite(mean).all() and 0 <= mean.min() <= mean.max() <= 1
            seed_means.append(mean)
            missing[f'{m}:{s}']=149-len(values)
        curves[m]=np.asarray(seed_means)
    (OUTPUT/'fig07_boundary_tolerance_values.json').write_text(json.dumps({
        'pixel_tolerance':list(map(int,x)),
        'seed_means':{m:curves[m].tolist() for m in METHODS},
        'empty_mask_exclusions':missing},indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(3.43,2.85))
    for m in METHODS:
        v=curves[m];y=v.mean(axis=0)
        ax.fill_between(x,v.min(axis=0),v.max(axis=0),color=COL[m],alpha=.08 if m=='bcskg' else .035)
        ax.plot(x,y,marker=MARK[m],color=COL[m],lw=1.8 if m=='bcskg' else 1.05,
                ms=3.2 if m=='bcskg' else 2.5,mec='white',mew=.3,label=SHORT[m])
    ax.set(xlabel='Boundary tolerance (pixels)',ylabel='BTS',ylim=(0,1))
    ax.set_xticks(x)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.22),fontsize=6.8,
              handlelength=1.2,columnspacing=.7)
    grid(ax);fig.tight_layout(pad=.3)
    save(fig,'fig07_boundary_tolerance')

def paired_differences():
    fig,axes=plt.subplots(1,2,figsize=(6.85,2.35),sharex=True,sharey=True)
    for ax,m in zip(axes,('unet','attention')):
        diffs=[]
        for s in SEEDS:
            a={r['id']:r['dice'] for r in rows(m,s)}
            b={r['id']:r['dice'] for r in rows('bcskg',s)}
            assert set(a)==set(b) and len(a)==149
            diffs.extend(b[k]-a[k] for k in sorted(a))
        d=np.asarray(diffs)
        ax.hist(d,bins=np.linspace(-1,1,41),color=FILL['bcskg'],edgecolor=COL['bcskg'],lw=.5)
        ax.axvline(0,color='#6B7280',lw=.9)
        ax.axvline(d.mean(),color=COL['unetwide'],ls='--',lw=1.2,label=f'Mean {d.mean():+.3f}')
        ax.set_title(f'BCRG-Net vs {NAME[m]}',loc='left')
        ax.legend(frameon=False,fontsize=7.2)
        ax.set(xlabel='Paired per-image Dice difference',xlim=(-1,1))
        grid(ax)
    axes[0].set_ylabel('Case-seed count')
    fig.tight_layout(pad=.5,w_pad=1)
    save(fig,'fig08_paired_differences')

def qualitative():
    u={r['id']:r for r in rows('unet',13)}
    b={r['id']:r for r in rows('bcskg',13)}
    ranked=sorted(u,key=lambda n:b[n]['dice']-u[n]['dice'])
    chosen=[ranked[0],ranked[-1],min(u,key=lambda n:b[n]['dice'])]
    fig,axes=plt.subplots(3,5,figsize=(6.85,4.35),gridspec_kw={'wspace':.08,'hspace':.16})
    titles=['Input','Target','U-Net','BCRG-Net','BCRG error']
    for i,n in enumerate(chosen):
        matches=list((DATA/'Kvasir-SEG/images').glob(n+'.*'));assert len(matches)==1
        im=Image.open(matches[0]).convert('RGB');axes[i,0].imshow(im)
        gt=ground_truth(n,(64,64));axes[i,1].imshow(gt,cmap='gray',vmin=0,vmax=1)
        pred={}
        for j,m in enumerate(('unet','bcskg')):
            with np.load(RESULTS/tag(m,13)/'predictions'/(n+'.npz')) as p:
                pred[m]=p['prob']>=result(m,13)['threshold']
            axes[i,j+2].imshow(pred[m],cmap='gray',vmin=0,vmax=1)
        err=np.ones((*gt.shape,3))
        err[pred['bcskg']&gt]=[.37,.56,.42]  # TP: moss green
        err[pred['bcskg']&~gt]=[.66,.53,.79] # FP: lilac purple
        err[gt&~pred['bcskg']]=[.94,.69,.56] # FN: warm coral
        axes[i,4].imshow(err)
        axes[i,0].set_ylabel(['Largest loss','Largest gain','Lowest Dice'][i],fontsize=8)
        for j,ax in enumerate(axes[i]):
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(titles[j],fontsize=8,pad=3)
    (OUTPUT/'fig09_case_selection.json').write_text(json.dumps({
        'seed':13,'ids':chosen,'rule':['minimum BCRG-U-Net Dice difference',
        'maximum BCRG-U-Net Dice difference','minimum BCRG-Net Dice'],
        'dice':[{m:round(u[n]['dice'] if m=='unet' else b[n]['dice'],6)
                 for m in ('unet','bcskg')} for n in chosen]},indent=2),encoding='utf-8')
    fig.tight_layout(pad=.2)
    save(fig,'fig09_qualitative')

def graph_evidence():
    source=OUTPUT/'fig10_graph_arrays.json'
    if not source.exists():
        print('fig10_graph_arrays.json absent; run extract_graph_evidence.py first')
        return False
    d=json.loads(source.read_text(encoding='utf-8'))
    adj=np.asarray(d['adj']);mem=np.asarray(d['membership'])
    assert adj.shape==(64,64) and mem.shape==(8,8,2)
    fig,axes=plt.subplots(1,4,figsize=(6.85,2.05),gridspec_kw={'width_ratios':[1,1,1.25,1]})
    im=Image.open(DATA/'Kvasir-SEG/images'/d['image_file']).convert('RGB')
    axes[0].imshow(im);axes[0].set_title('Input',fontsize=8)
    img=axes[1].imshow(mem[...,1],cmap='Purples',vmin=0,vmax=1)
    axes[1].set_title('Foreground membership',fontsize=8)
    fig.colorbar(img,ax=axes[1],fraction=.046,pad=.02,ticks=[0,.5,1])
    img=axes[2].imshow(adj,cmap='Greens',vmin=0,vmax=adj.max())
    axes[2].set_title('Learned relation matrix',fontsize=8)
    fig.colorbar(img,ax=axes[2],fraction=.046,pad=.02)
    coord=np.array([(i%8,i//8) for i in range(64)])
    rng=np.random.default_rng(7)
    for a in rng.choice(64,12,replace=False):
        for bb in np.argsort(adj[a])[-2:]:
            if a!=bb:
                axes[3].plot([coord[a,0],coord[bb,0]],[coord[a,1],coord[bb,1]],
                             alpha=np.clip(adj[a,bb]*30,.12,.95),color=COL['bcskg'],lw=.8)
    axes[3].scatter(coord[:,0],coord[:,1],c=mem[...,1].ravel(),cmap='Purples',
                    vmin=0,vmax=1,s=11,edgecolor='white',linewidth=.15,zorder=3)
    axes[3].invert_yaxis();axes[3].set_aspect('equal')
    axes[3].set_title('Strong region links',fontsize=8)
    for ax in axes:
        ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout(pad=.25,w_pad=.5)
    save(fig,'fig10_graph_evidence')
    return True

if __name__=='__main__':
    dataset_landscape();print('fig03 done',flush=True)
    size_stratified();print('fig06 done',flush=True)
    boundary_tolerance();print('fig07 done',flush=True)
    paired_differences();print('fig08 done',flush=True)
    qualitative();print('fig09 done',flush=True)
    graph_evidence();print('fig10 checked',flush=True)
