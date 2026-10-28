"""Small executable correctness tests; not medical performance tests."""
import torch
from model import BCSKGNet, segmentation_loss

torch.manual_seed(3)
x=torch.randn(2,3,64,64)
y=(torch.rand(2,1,64,64)>.65).float()
for method in ['unet','attention','graph','bcskg']:
    model=BCSKGNet(base=8,graph=method not in ['unet','attention'],attention=method=='attention')
    logits,aux=model(x,y if method not in ['unet','attention'] else None)
    assert logits.shape==y.shape and torch.isfinite(logits).all()
    loss,parts=segmentation_loss(logits,y,aux,.05,.1,.2 if method=='bcskg' else 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(method,float(loss),parts)

model=BCSKGNet(base=8,graph=True)
model.eval()
with torch.no_grad():
    train_logits,train_aux=model(x,y)
    test_logits,test_aux=model(x,None)
assert train_aux['membership'].shape[-1]==2
assert 'pred_prob' in test_aux and test_aux['pred_prob'].shape[-2:]==(8,8)
assert test_aux['base_logits'].shape==y.shape and test_aux['graph_logits'].shape==y.shape
print('Inference without target masks: OK')
