import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self,cin,cskip,cout,attention=False):
        super().__init__(); self.attention=attention
        if attention == 'original':
            inter=max(cskip//2,1)
            self.gate_g=nn.Conv2d(cin,inter,1,bias=False)
            self.gate_x=nn.Conv2d(cskip,inter,1,bias=False)
            self.gate_psi=nn.Conv2d(inter,1,1)
        elif attention:
            self.gate=nn.Conv2d(cskip,1,1)
        self.conv=ConvBlock(cin+cskip,cout)
    def forward(self,x,skip):
        g=x
        x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        if self.attention == 'original':
            theta=self.gate_x(skip)
            phi=F.interpolate(self.gate_g(g),size=skip.shape[-2:],mode='bilinear',align_corners=False)
            skip=skip*torch.sigmoid(self.gate_psi(F.relu(theta+phi,inplace=True)))
        elif self.attention:
            skip=skip*torch.sigmoid(self.gate(skip))
        return self.conv(torch.cat([x,skip],1))

class SemanticKnowledgeGraph(nn.Module):
    """Region--concept graph with typed spatial, affinity and membership relations."""
    def __init__(self,channels,dim=32):
        super().__init__()
        self.node_proj=nn.Conv2d(channels,dim,1)
        self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim); self.v=nn.Linear(dim,dim)
        self.cq=nn.Linear(dim,dim); self.ck=nn.Linear(dim,dim)
        self.out=nn.Conv2d(dim,channels,1); self.norm=nn.LayerNorm(dim)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.aux=nn.Conv2d(channels,1,1)
    def forward(self,x,mask=None):
        b,c,h,w=x.shape; n=h*w
        z=self.node_proj(x).flatten(2).transpose(1,2)
        q=F.normalize(self.q(z),dim=-1); k=F.normalize(self.k(z),dim=-1)
        affinity=(q@k.transpose(1,2))/0.25
        yy,xx=torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
        coords=torch.stack([xx.flatten()/max(w-1,1),yy.flatten()/max(h-1,1)],-1)
        dist=torch.cdist(coords,coords)
        spatial=torch.exp(-dist**2/0.04).log().clamp_min(-20).unsqueeze(0)
        relation_logits=affinity+spatial
        adj=torch.softmax(relation_logits,dim=-1)
        pred_logits=self.aux(x)
        pred_prob=torch.sigmoid(pred_logits)
        if mask is None:
            prob=pred_prob
        else:
            prob=F.interpolate(mask.float(),size=(h,w),mode='area')
        membership=torch.cat([1-prob.flatten(2),prob.flatten(2)],1).transpose(1,2)
        concepts=membership.transpose(1,2)@z
        concepts=concepts/membership.sum(1,keepdim=True).transpose(1,2).clamp_min(1e-4)
        cr=F.normalize(self.cq(z),dim=-1)@F.normalize(self.ck(concepts),dim=-1).transpose(1,2)
        concept_adj=torch.softmax(cr/0.25,dim=-1)
        messages=adj@self.v(z)+concept_adj@concepts
        refined=self.norm(z+messages)
        out=self.out(refined.transpose(1,2).reshape(b,-1,h,w))
        return x+self.gamma*out, {'relation_logits':relation_logits,'membership':membership,'prob':prob,'pred_prob':pred_prob,'pred_logits':pred_logits}

class BCSKGNet(nn.Module):
    def __init__(self,base=8,graph=True,attention=False):
        super().__init__(); self.graph_enabled=graph
        self.e1=ConvBlock(3,base); self.e2=ConvBlock(base,base*2); self.e3=ConvBlock(base*2,base*4); self.e4=ConvBlock(base*4,base*8)
        self.pool=nn.MaxPool2d(2); self.graph=SemanticKnowledgeGraph(base*8) if graph else None
        self.d3=UpBlock(base*8,base*4,base*4,attention); self.d2=UpBlock(base*4,base*2,base*2,attention); self.d1=UpBlock(base*2,base,base,attention)
        self.head=nn.Conv2d(base,1,1)
        if graph:
            self.gd3=UpBlock(base*8,base*4,base*4,False)
            self.gd2=UpBlock(base*4,base*2,base*2,False)
            self.gd1=UpBlock(base*2,base,base,False)
            self.graph_head=nn.Conv2d(base,1,1)
            # Start near the base decoder and learn whether graph decoding helps.
            self.fusion_logit=nn.Parameter(torch.tensor(-4.0))
    def forward(self,x,mask=None):
        s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); b=self.e4(self.pool(s3)); aux={}
        d=self.d3(b,s3); d=self.d2(d,s2); d=self.d1(d,s1)
        base_logits=self.head(d)
        if self.graph is not None:
            bg,aux=self.graph(b,mask)
            gd=self.gd3(bg,s3); gd=self.gd2(gd,s2); gd=self.gd1(gd,s1)
            graph_logits=self.graph_head(gd)
            fusion=torch.sigmoid(self.fusion_logit)
            logits=base_logits+fusion*(graph_logits-base_logits)
            aux.update({'base_logits':base_logits,'graph_logits':graph_logits,'fusion':fusion})
            return logits,aux
        return base_logits,aux

def dice_loss(logits,target):
    p=torch.sigmoid(logits); dims=(1,2,3)
    return (1-(2*(p*target).sum(dims)+1)/(p.sum(dims)+target.sum(dims)+1)).mean()

def boundary_map(mask):
    m=F.max_pool2d(mask,3,1,1)
    return (m-F.avg_pool2d(m,3,1,1)).abs()

def segmentation_loss(logits,target,aux=None,lambda_rel=0.05,lambda_bd=0.1,lambda_cal=0.2,lambda_branch=0.2):
    seg=F.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)
    rel=torch.tensor(0.,device=logits.device); bd=rel.clone()
    cal=torch.tensor(0.,device=logits.device); branch=cal.clone()
    if aux:
        if 'base_logits' in aux:
            branch=0.5*(F.binary_cross_entropy_with_logits(aux['base_logits'],target)+dice_loss(aux['base_logits'],target)+F.binary_cross_entropy_with_logits(aux['graph_logits'],target)+dice_loss(aux['graph_logits'],target))
        gt=F.interpolate(target,size=aux['pred_prob'].shape[-2:],mode='area')
        cal=F.binary_cross_entropy_with_logits(aux['pred_logits'],gt)
        same=(gt.flatten(2).transpose(1,2)*gt.flatten(2)+(1-gt.flatten(2).transpose(1,2))*(1-gt.flatten(2)))
        same=same/(same.sum(-1,keepdim=True)+1e-6)
        pred=torch.softmax(aux['relation_logits'],dim=-1)
        # Mean over both query nodes and the batch; avoid a hidden factor of N.
        rel=F.kl_div(pred.clamp_min(1e-6).log(),same.clamp_min(1e-6),reduction='none').sum(-1).mean()
        bd_aux=F.l1_loss(boundary_map(aux['pred_prob']),boundary_map(gt))
        bd_final=F.l1_loss(boundary_map(torch.sigmoid(logits)),boundary_map(target))
        bd=0.5*(bd_aux+bd_final)
    total=seg+lambda_rel*rel+lambda_bd*bd+lambda_cal*cal+lambda_branch*branch
    return total, {'seg':seg.item(),'relation':rel.item(),'boundary':bd.item(),'calibration':cal.item(),'branch':branch.item()}
