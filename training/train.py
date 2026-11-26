"""Train a compact CNN, optionally fine-tuning on application-labeled samples."""
import argparse, json, random, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
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
    datasets=[ImageFolder(Path(data) / split, transform=tf) for split in ("train", "validation", "test")]
    if len(datasets[1]) > 2048:
        indices=torch.randperm(len(datasets[1]),generator=torch.Generator().manual_seed(1729))[:2048].tolist()
        datasets[1]=Subset(datasets[1],indices)
    return [DataLoader(ds, batch_size=batch, shuffle=i == 0, num_workers=0) for i,ds in enumerate(datasets)]

def combined_train_loader(data, labeled, batch, seed):
    tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    synthetic = ImageFolder(Path(data) / "train", transform=tf)
    real = ImageFolder(Path(labeled) / "train", transform=tf, allow_empty=True)
    if synthetic.classes != real.classes:
        raise ValueError(f"Class folders differ: synthetic={synthetic.classes}, labeled={real.classes}")
    dataset = ConcatDataset([synthetic, real])
    weights = torch.cat([
        torch.full((len(synthetic),), 0.5 / len(synthetic)),
        torch.full((len(real),), 0.5 / len(real)),
    ])
    samples_per_epoch = max(2048, len(real) * 8)
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=0), len(real)

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
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/generated'); ap.add_argument('--output',default='models'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--patience',type=int,default=5); ap.add_argument('--seed',type=int,default=1729); ap.add_argument('--samples-csv',help='Application expression labels CSV; fine-tunes from the existing checkpoint.'); ap.add_argument('--init-checkpoint',default='models/best.pt'); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(4)
    device=torch.device('cpu'); train,val,test=loaders(a.data,a.batch_size); model=TinyCNN().to(device)
    labeled_tmp = tempfile.TemporaryDirectory(prefix='ocr-labeled-') if a.samples_csv else None
    sample_val = None
    if a.samples_csv:
        csv_path=Path(a.samples_csv)
        if not csv_path.is_file(): raise FileNotFoundError(f'Labeled CSV not found: {csv_path}')
        init=Path(a.init_checkpoint)
        if not init.is_file(): raise FileNotFoundError(f'Initial checkpoint not found: {init}')
        subprocess.run(['node', 'training/extract_labeled_samples.js', str(csv_path), labeled_tmp.name, str(a.seed)], check=True)
        ck=torch.load(init,map_location=device,weights_only=True)
        if ck.get('classes') != CLASSES: raise ValueError('Initial checkpoint class order does not match the current trainer')
        model.load_state_dict(ck['model'])
        train, real_count = combined_train_loader(a.data,labeled_tmp.name,a.batch_size,a.seed)
        tf=transforms.Compose([transforms.Grayscale(),transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        sample_val=DataLoader(ImageFolder(Path(labeled_tmp.name)/'validation',transform=tf,allow_empty=True),batch_size=a.batch_size,shuffle=False,num_workers=0)
        print(f'Fine-tuning from {init} with {real_count} labeled glyphs; validation split is grouped by expression.', flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=0.0001 if a.samples_csv else 0.001)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); best=float('inf'); stale=0; history=[]; best_record=None
    for epoch in range(1,a.epochs+1):
        tr=run_epoch(model,train,device,opt); vr=run_epoch(model,val,device)
        record={'epoch':epoch,'train_loss':tr[0],'train_accuracy':tr[1],'validation_loss':vr[0],'validation_accuracy':vr[1]}
        score=vr[0]
        if sample_val:
            sv=run_epoch(model,sample_val,device)
            record.update({'sample_validation_loss':sv[0],'sample_validation_accuracy':sv[1]})
            score=(vr[0]+sv[0])/2
        history.append(record)
        print(json.dumps(history[-1]),flush=True)
        if score < best:
            best=score; stale=0; best_record=record.copy(); torch.save({'model':model.state_dict(),'classes':CLASSES},out/'best.pt')
        else: stale+=1
        (out/'history.json').write_text(json.dumps({'history':history,'selection_metric':'mean synthetic and labeled validation loss' if sample_val else 'synthetic validation loss','test_status':'pending'},indent=2))
        if stale>=a.patience: break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(checkpoint['model']); tl,ta=run_epoch(model,test,device)
    result={'history':history,'best_epoch':best_record['epoch'],'test_loss':tl,'test_accuracy':ta,'test_split':'synthetic'}
    if sample_val:
        result['sample_validation']={'loss':best_record['sample_validation_loss'],'accuracy':best_record['sample_validation_accuracy'],'glyphs':len(sample_val.dataset)}
    (out/'history.json').write_text(json.dumps(result,indent=2))
    print(f'test_loss={tl:.4f} test_accuracy={ta:.4%}')
    if labeled_tmp: labeled_tmp.cleanup()
if __name__=='__main__': main()
