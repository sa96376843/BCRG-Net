"""Export one actual seed-13 graph forward pass without depending on Matplotlib."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image
import torch

WORK=Path(__file__).resolve().parents[1]
PROJECT=WORK.parent/'ESWA_KG_Polyp'
sys.path.insert(0,str(PROJECT/'src'))
from model import BCSKGNet

with (PROJECT/'results/split_manifest.csv').open(newline='',encoding='utf-8') as f:
    first=next(r for r in csv.DictReader(f) if r['dataset']=='kvasir' and r['split']=='test')
image=PROJECT/first['image']
run=PROJECT/'results/kvasir_bcskg_s13_r0.05_b0.1'
record=json.loads((run/'result.json').read_text(encoding='utf-8'))
model=BCSKGNet(base=record['base'],graph=True)
model.load_state_dict(torch.load(run/'model.pt',map_location='cpu'))
model.eval()
im=Image.open(image).convert('RGB').resize((64,64),Image.Resampling.BILINEAR)
x=torch.tensor(list(im.getdata()),dtype=torch.float32).reshape(64,64,3).permute(2,0,1)/255.
mean=torch.tensor([.485,.456,.406],dtype=torch.float32)[:,None,None]
std=torch.tensor([.229,.224,.225],dtype=torch.float32)[:,None,None]
x=(x-mean)/std
with torch.no_grad():
    _,aux=model(x[None])
    adj=torch.softmax(aux['relation_logits'][0],dim=-1)
    membership=aux['membership'][0].reshape(8,8,2)
payload={'dataset':'Kvasir-SEG','split':'test','seed':13,'sample_id':image.stem,
    'image_file':image.name,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
    'model_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
    'preprocessing':'RGB bilinear to 64x64; ImageNet channel mean/std',
    'adj':adj.tolist(),'membership':membership.tolist()}
out=WORK/'experimental_redraw/fig10_graph_arrays.json'
out.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
print(image.stem,adj.shape,membership.shape,str(out))
