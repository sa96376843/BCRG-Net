"""Controlled component-ablation runs; no test-based model selection."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import BCSKGNet, UpBlock, dice_loss
from experiment import ROOT, RESULTS, seed_all, pairs_for, split_pairs, CachedSeg, evaluate

class AblationNet(BCSKGNet):
    """A: dual decoder; B: +graph; C: +relation; D: +boundary; full: +calibration."""
    def __init__(self, variant, base=8):
        self.variant=variant
        super().__init__(base=base,graph=variant!='dual',attention=False)
        if variant=='dual':
            self.gd3=UpBlock(base*8,base*4,base*4,False)
            self.gd2=UpBlock(base*4,base*2,base*2,False)
            self.gd1=UpBlock(base*2,base,base,False)
            self.graph_head=torch.nn.Conv2d(base,1,1)
            self.fusion_logit=torch.nn.Parameter(torch.tensor(-4.0))
    def forward(self,x,mask=None):
        s1=self.e1(x);s2=self.e2(self.pool(s1));s3=self.e3(self.pool(s2));b=self.e4(self.pool(s3))
        d=self.d3(b,s3);d=self.d2(d,s2);d=self.d1(d,s1);base_logits=self.head(d)
        aux={}
        if self.graph is not None:
            bg,aux=self.graph(b,mask)
        else:
            bg=b
        gd=self.gd3(bg,s3);gd=self.gd2(gd,s2);gd=self.gd1(gd,s1)
        graph_logits=self.graph_head(gd)
        fusion=torch.sigmoid(self.fusion_logit)
        logits=base_logits+fusion*(graph_logits-base_logits)
        aux.update({'base_logits':base_logits,'graph_logits':graph_logits,'fusion':fusion})
        return logits,aux

def loss_for(model,logits,target,aux,lambda_rel,lambda_bd):
    seg=F.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)
    branch=.5*(F.binary_cross_entropy_with_logits(aux['base_logits'],target)+dice_loss(aux['base_logits'],target)+F.binary_cross_entropy_with_logits(aux['graph_logits'],target)+dice_loss(aux['graph_logits'],target))
    total=seg+.2*branch
    parts={'seg':seg.item(),'branch':branch.item(),'relation':0.,'boundary':0.,'calibration':0.}
    if model.variant!='dual':
        from model import segmentation_loss,boundary_map
        gt=F.interpolate(target,size=aux['pred_prob'].shape[-2:],mode='area')
        if model.variant in ['relation','boundary']:
            same=gt.flatten(2).transpose(1,2)*gt.flatten(2)+(1-gt.flatten(2).transpose(1,2))*(1-gt.flatten(2))
            same=same/(same.sum(-1,keepdim=True)+1e-6)
            pred=torch.softmax(aux['relation_logits'],dim=-1)
            rel=F.kl_div(pred.clamp_min(1e-6).log(),same.clamp_min(1e-6),reduction='none').sum(-1).mean()
            total=total+lambda_rel*rel;parts['relation']=rel.item()
        if model.variant=='boundary':
            bd=.5*(F.l1_loss(boundary_map(aux['pred_prob']),boundary_map(gt))+F.l1_loss(boundary_map(torch.sigmoid(logits)),boundary_map(target)))
            total=total+lambda_bd*bd;parts['boundary']=bd.item()
    return total,parts

def select_operating_point(model,loader,fixed_fusion=None):
    device=next(model.parameters()).device;cache=[]
    with torch.no_grad():
        for x,y,names in loader:
            x=x.to(device);logits,aux=model(x);cache.append((y,aux))
    candidates=[fixed_fusion] if fixed_fusion is not None else [0.,.1,.2,.3,.4,.5,.7,1.]
    scores=[]
    for w in candidates:
        vals=[]
        for y,aux in cache:
            z=aux['base_logits']+w*(aux['graph_logits']-aux['base_logits'])
            p=torch.sigmoid(z)>=.5;g=y>.5;dims=(1,2,3)
            vals.extend(((2*(p&g).sum(dims)+1e-7)/(p.sum(dims)+g.sum(dims)+1e-7)).cpu().numpy())
        scores.append(float(np.mean(vals)))
    fusion=float(candidates[int(np.argmax(scores))])
    with torch.no_grad():
        model.fusion_logit.copy_(torch.tensor(np.log(fusion/(1-fusion)) if 0<fusion<1 else (-20 if fusion==0 else 20),device=device))
    thresholds=np.arange(.30,.701,.05);scores=[]
    for t in thresholds:
        vals=[]
        for y,aux in cache:
            z=aux['base_logits']+fusion*(aux['graph_logits']-aux['base_logits'])
            p=torch.sigmoid(z)>=t;g=y>.5;dims=(1,2,3)
            vals.extend(((2*(p&g).sum(dims)+1e-7)/(p.sum(dims)+g.sum(dims)+1e-7)).cpu().numpy())
        scores.append(float(np.mean(vals)))
    return fusion,float(thresholds[int(np.argmax(scores))])

def run(args):
    seed_all(args.seed);torch.set_num_threads(2)
    tr,va,te=split_pairs('kvasir',pairs_for('kvasir'))
    train=DataLoader(CachedSeg(tr,args.size,True),batch_size=16,shuffle=True,num_workers=0)
    val=DataLoader(CachedSeg(va,args.size),batch_size=16,num_workers=0)
    test=DataLoader(CachedSeg(te,args.size),batch_size=16,num_workers=0)
    model=AblationNet(args.variant,base=8);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=-1;best_state=None;history=[];start=time.time()
    for epoch in range(args.epochs):
        model.train();losses=[]
        for x,y,_ in train:
            opt.zero_grad();logits,aux=model(x,y if model.graph is not None else None)
            loss,parts=loss_for(model,logits,y,aux,args.lambda_rel,args.lambda_bd)
            loss.backward();opt.step();losses.append(loss.item())
        sched.step();vs,_=evaluate(model,val,quick=True)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_dice':vs['dice']})
        if vs['dice']>best:
            best=vs['dice'];best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        print('ABLATION',args.variant,args.seed,epoch+1,round(vs['dice'],4),flush=True)
    model.load_state_dict(best_state)
    fusion,threshold=select_operating_point(model,val)
    tag=f'kvasir_ablation_{args.variant}_s{args.seed}_size{args.size}'
    out=RESULTS/tag;out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'model.pt')
    test_summary,rows=evaluate(model,test,out/'predictions',threshold=threshold)
    x,y,n=test.dataset[0];x=x[None]
    with torch.no_grad():
        for _ in range(3):model(x)
        t=time.time()
        for _ in range(10):model(x)
    result={'dataset':'kvasir','method':'ablation_'+args.variant,'variant':args.variant,'seed':args.seed,'size':args.size,'base':8,'epochs':args.epochs,'batch':16,'lr':2e-3,'lambda_rel':args.lambda_rel if args.variant in ['relation','boundary'] else 0,'lambda_bd':args.lambda_bd if args.variant=='boundary' else 0,'lambda_cal':0,'branch_weight':.2,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'test':test_summary,'params':sum(p.numel() for p in model.parameters()),'graph_gate':float(model.graph.gamma.detach().cpu()) if model.graph is not None else None,'fusion_weight':fusion,'threshold':threshold,'latency_ms':(time.time()-t)*100,'train_seconds':time.time()-start,'history':history}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'per_image.json').write_text(json.dumps(rows,indent=2))
    (RESULTS/(tag+'.json')).write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['dual','graph_only','relation','boundary'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--size',type=int,default=64);p.add_argument('--lambda-rel',type=float,default=.05);p.add_argument('--lambda-bd',type=float,default=.1);run(p.parse_args())
