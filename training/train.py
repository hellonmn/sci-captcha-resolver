"""Train a compact 16-class CNN from generated grayscale glyphs."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataset_generator import CLASSES

class TinyCNN(nn.Module):
    def __init__(self, classes=15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(48 * 4 * 4, 64), nn.ReLU(inplace=True), nn.Linear(64, classes))
    def forward(self, x): return self.classifier(self.features(x))

def loaders(data, batch):
    tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return [DataLoader(ImageFolder(Path(data) / split, transform=tf), batch_size=batch, shuffle=split == "train", num_workers=0) for split in ("train", "validation", "test")]

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None; model.train(training); loss_fn = nn.CrossEntropyLoss(); losses=[]; correct=total=0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = loss_fn(logits, y)
            if training: loss.backward(); optimizer.step()
            losses.append(loss.item()); correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    return float(np.mean(losses)), correct / max(total, 1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/generated'); ap.add_argument('--output',default='models'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--patience',type=int,default=5); ap.add_argument('--seed',type=int,default=1729); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(4)
    device=torch.device('cpu'); train,val,test=loaders(a.data,a.batch_size); model=TinyCNN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); best=float('inf'); stale=0; history=[]
    for epoch in range(1,a.epochs+1):
        tr=run_epoch(model,train,device,opt); vr=run_epoch(model,val,device); history.append({'epoch':epoch,'train_loss':tr[0],'train_accuracy':tr[1],'validation_loss':vr[0],'validation_accuracy':vr[1]})
        print(json.dumps(history[-1]),flush=True)
        if vr[0] < best:
            best=vr[0]; stale=0; torch.save({'model':model.state_dict(),'classes':CLASSES},out/'best.pt')
        else: stale+=1
        if stale>=a.patience: break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(checkpoint['model']); tl,ta=run_epoch(model,test,device)
    (out/'history.json').write_text(json.dumps({'history':history,'test_loss':tl,'test_accuracy':ta},indent=2))
    print(f'test_loss={tl:.4f} test_accuracy={ta:.4%}')
if __name__=='__main__': main()
