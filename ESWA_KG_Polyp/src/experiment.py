import os, sys, json, time, hashlib, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader, Dataset
from model import BCSKGNet, segmentation_loss
from baseline_models import build_additional_baseline
from surface_metrics import surface_distance

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; RESULTS=ROOT/'results'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def pairs_for(name):
    if name=='kvasir':
        base=DATA/'Kvasir-SEG'; ims=sorted((base/'images').glob('*')); masks=base/'masks'
        return [(p,masks/p.name) for p in ims]
    if name=='cvc':
        base=DATA/'CVC-ClinicDB'/'PNG'; ims=sorted((base/'Original').glob('*.png'),key=lambda p:int(p.stem))
        return [(p,base/'Ground Truth'/p.name) for p in ims]
    if name=='instrument':
        base=DATA/'kvasir-instrument'/'official'/'kvasir-instrument'
        ims=sorted((base/'images'/'images').glob('*.jpg'))
        masks=base/'masks'/'masks'
        return [(p,masks/(p.stem+'.png')) for p in ims if (masks/(p.stem+'.png')).exists()]
    raise ValueError(name)

def group_key(name,p):
    if name=='cvc': return (int(p.stem)-1)//25
    return p.stem

def split_pairs(name,pairs):
    groups={}
    for pair in pairs: groups.setdefault(group_key(name,pair[0]),[]).append(pair)
    def h(g): return int(hashlib.md5((name+'::'+str(g)).encode()).hexdigest()[:8],16)/0xffffffff
    train,val,test=[],[],[]
    if name=='cvc':
        # Balanced deterministic allocation avoids a tiny test set with only 25 groups.
        ordered=sorted(groups,key=lambda g:(h(g),str(g)))
        proportions=[.70,.15,.15]
        counts=[int(p*len(ordered)) for p in proportions]
        remainders=[proportions[i]*len(ordered)-counts[i] for i in range(3)]
        for idx in sorted(range(3),key=lambda i:(remainders[i],-i),reverse=True)[:len(ordered)-sum(counts)]:
            counts[idx]+=1
        selected=[ordered[:counts[0]],ordered[counts[0]:counts[0]+counts[1]],ordered[counts[0]+counts[1]:]]
        for dest,keys in zip([train,val,test],selected):
            for g in keys: dest.extend(groups[g])
    else:
        for g,items in sorted(groups.items(),key=lambda z:str(z[0])):
            r=h(g)
            (train if r<.70 else val if r<.85 else test).extend(items)
    return train,val,test

_SNAPSHOT=None

def snapshot_index():
    global _SNAPSHOT
    path=DATA/'experiment_data_64x64.npz'
    if _SNAPSHOT is None and path.exists():
        z=np.load(path)
        # CompressedNpzFile decompresses on each __getitem__; materialize once.
        arrays={'images':z['images'],'masks':z['masks']}
        _SNAPSHOT=(arrays,{str(n):i for i,n in enumerate(z['ids'])})
    return _SNAPSHOT

class CachedSeg(Dataset):
    def __init__(self,pairs,size=64,train=False):
        self.train=train; self.items=[]
        snap=snapshot_index() if size==64 else None
        for ip,mp in pairs:
            if snap is not None and ip.stem in snap[1]:
                z,idx=snap; arr=z['images'][idx[ip.stem]]; mask=z['masks'][idx[ip.stem]]
                a=arr.astype(np.float32).transpose(2,0,1)/255.; mb=mask*255
            else:
                x=Image.open(ip).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
                y=Image.open(mp).convert('L').resize((size,size),Image.Resampling.NEAREST)
                a=np.asarray(x,dtype=np.float32).transpose(2,0,1)/255.;mb=np.asarray(y)
            a=(a-np.array([.485,.456,.406],dtype=np.float32)[:,None,None])/np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
            self.items.append((torch.from_numpy(a.astype(np.float32)),torch.from_numpy((mb>127).astype(np.float32))[None],ip.stem))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        x,y,n=self.items[i]
        if self.train and random.random()<.5: x=torch.flip(x,[2]); y=torch.flip(y,[2])
        return x,y,n

def metrics(pred,gt):
    inter=(pred&gt).sum(); dice=(2*inter+1e-7)/(pred.sum()+gt.sum()+1e-7)
    iou=(inter+1e-7)/(np.logical_or(pred,gt).sum()+1e-7)
    hd,assd=surface_distance(pred,gt)
    pbd=pred & ~binary_erosion(pred); gbd=gt & ~binary_erosion(gt)
    pd=distance_transform_edt(~pbd); gd=distance_transform_edt(~gbd)
    biou=((pbd&(gd<=2)).sum()+(gbd&(pd<=2)).sum())/(pbd.sum()+gbd.sum()+1e-7)
    return {'dice':float(dice),'iou':float(iou),'hd95':hd,'assd':assd,'biou':float(biou),'area':float(gt.mean())}

def evaluate(model,loader,save_dir=None,quick=False,threshold=.5):
    model.eval(); rows=[]; device=next(model.parameters()).device
    if save_dir: Path(save_dir).mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for x,y,names in loader:
            x=x.to(device); y=y.to(device); logits,_=model(x)
            probs=torch.sigmoid(logits).cpu().numpy()[:,0]; gt=y.numpy()[:,0]
            for j,name in enumerate(names):
                if quick:
                    pr=probs[j]>=threshold; g=gt[j]>.5; inter=(pr&g).sum()
                    m={'dice':float((2*inter+1e-7)/(pr.sum()+g.sum()+1e-7))}
                else:
                    m=metrics(probs[j]>=threshold,gt[j]>.5)
                m['id']=name; rows.append(m)
                if save_dir:
                    np.savez_compressed(Path(save_dir)/(name+'.npz'),
                                        prob=probs[j].astype(np.float16),
                                        pred=(probs[j]>=threshold).astype(np.uint8),
                                        threshold=np.float64(threshold),
                                        gt=gt[j].astype(np.uint8))
    keys=['dice'] if quick else ['dice','iou','hd95','assd','biou']
    summary={k:float(np.nanmean([r[k] for r in rows])) for k in keys}
    return summary,rows

def validation_operating_point(model,loader):
    """Select fusion and threshold on validation data only, then freeze both."""
    device=next(model.parameters()).device
    cache=[]; best_fusion=None
    with torch.no_grad():
        for x,y,names in loader:
            x=x.to(device);logits,aux=model(x)
            cache.append((y,aux))
    if model.graph is not None:
        candidates=[0.,.1,.2,.3,.4,.5,.7,1.]
        scores=[]
        for w in candidates:
            vals=[]
            for y,aux in cache:
                z=aux['base_logits']+w*(aux['graph_logits']-aux['base_logits'])
                p=(torch.sigmoid(z)>=.5);g=y>.5;dims=(1,2,3)
                vals.extend(((2*(p&g).sum(dims)+1e-7)/(p.sum(dims)+g.sum(dims)+1e-7)).cpu().numpy())
            scores.append(float(np.mean(vals)))
        best_fusion=float(candidates[int(np.argmax(scores))])
        with torch.no_grad():
            model.fusion_logit.copy_(torch.tensor(np.log(best_fusion/(1-best_fusion)) if 0<best_fusion<1 else (-20 if best_fusion==0 else 20),device=device))
    thresholds=np.arange(.30,.701,.05);scores=[]
    with torch.no_grad():
        for t in thresholds:
            vals=[]
            for x,y,names in loader:
                x=x.to(device);logits,_=model(x);p=(torch.sigmoid(logits)>=t);g=y>.5;dims=(1,2,3)
                vals.extend(((2*(p&g).sum(dims)+1e-7)/(p.sum(dims)+g.sum(dims)+1e-7)).cpu().numpy())
            scores.append(float(np.mean(vals)))
    threshold=float(thresholds[int(np.argmax(scores))])
    return best_fusion,threshold

def run(args):
    seed_all(args.seed); torch.set_num_threads(2)
    pairs=pairs_for(args.dataset); tr,va,te=split_pairs(args.dataset,pairs)
    train=DataLoader(CachedSeg(tr,args.size,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val=DataLoader(CachedSeg(va,args.size),batch_size=args.batch,num_workers=0)
    test=DataLoader(CachedSeg(te,args.size),batch_size=args.batch,num_workers=0)
    graph=args.method not in ['unet','attention','attention_original','unetpp','deeplab']
    attention='original' if args.method=='attention_original' else args.method=='attention'
    model=build_additional_baseline(args.method,base=args.base)
    if model is None:
        model=BCSKGNet(base=args.base,graph=graph,attention=attention)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=-1; history=[]; best_state=None; start=time.time()
    no_aux=['unet','attention','attention_original','unetpp','deeplab','graph']
    lrel=0 if args.method in no_aux else args.lambda_rel
    lbd=0 if args.method in no_aux else args.lambda_bd
    lcal=0 if args.method in no_aux else 0.2
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for x,y,_ in train:
            x,y=x,y
            opt.zero_grad(); logits,aux=model(x,y if graph else None)
            loss,parts=segmentation_loss(logits,y,aux,lrel,lbd,lcal,args.lambda_branch); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step(); vs,_=evaluate(model,val,quick=True); history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_dice':vs['dice']})
        if vs['dice']>best:
            best=vs['dice']; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,RESULTS/(f'{args.dataset}_{args.method}_s{args.seed}_best_tmp.pt'))
        print(args.dataset,args.method,args.seed,epoch+1,round(losses[-1],4),round(vs['dice'],4),flush=True)
    model.load_state_dict(best_state)
    selected_fusion,threshold=validation_operating_point(model,val)
    tag=f'{args.dataset}_{args.method}_s{args.seed}_r{args.lambda_rel}_b{args.lambda_bd}'
    if args.size!=64: tag+=f'_size{args.size}'
    if args.base!=8: tag+=f'_base{args.base}'
    if args.lambda_branch!=0.2: tag+=f'_branch{args.lambda_branch}'
    out=RESULTS/tag; out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'model.pt')
    test_summary,rows=evaluate(model,test,out/'predictions',threshold=threshold)
    x,_y,_=test.dataset[0]; x=x[None]
    with torch.no_grad():
        for _ in range(3): model(x)
        t=time.time()
        for _ in range(10): model(x)
    latency=(time.time()-t)*100
    params=sum(p.numel() for p in model.parameters())
    graph_gate=float(model.graph.gamma.detach().cpu()) if model.graph is not None else None
    fusion=float(torch.sigmoid(model.fusion_logit).detach().cpu()) if model.graph is not None else None
    result={'dataset':args.dataset,'method':args.method,'seed':args.seed,'lambda_rel':args.lambda_rel,'lambda_bd':args.lambda_bd,'lambda_cal':lcal,'lambda_branch':args.lambda_branch,'size':args.size,'base':args.base,'epochs':args.epochs,'batch':args.batch,'lr':args.lr,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'test':test_summary,'params':params,'graph_gate':graph_gate,'fusion_weight':selected_fusion,'threshold':threshold,'latency_ms':latency,'train_seconds':time.time()-start,'history':history}
    (out/'result.json').write_text(json.dumps(result,indent=2)); (out/'per_image.json').write_text(json.dumps(rows,indent=2))
    (RESULTS/(tag+'.json')).write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--method',required=True); p.add_argument('--seed',type=int,default=13)
    p.add_argument('--epochs',type=int,default=12); p.add_argument('--size',type=int,default=64); p.add_argument('--base',type=int,default=8); p.add_argument('--batch',type=int,default=16); p.add_argument('--lr',type=float,default=2e-3)
    p.add_argument('--lambda-rel',type=float,default=.05); p.add_argument('--lambda-bd',type=float,default=.1); p.add_argument('--lambda-branch',type=float,default=.2); run(p.parse_args())
