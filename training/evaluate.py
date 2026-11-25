"""Evaluate a checkpoint on a separately generated split and save confusion matrix."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from train import TinyCNN, loaders
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/generated'); p.add_argument('--checkpoint',default='models/best.pt'); p.add_argument('--output',default='models/evaluation.json'); a=p.parse_args()
    torch.set_num_threads(4); _,_,test=loaders(a.data,256); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model=TinyCNN(); model.load_state_dict(ck['model']); model.eval(); cm=np.zeros((15,15),dtype=np.int64)
    with torch.no_grad():
        for x,y in test:
            pred=model(x).argmax(1); np.add.at(cm,(y.numpy(),pred.numpy()),1)
    accuracy=float(np.trace(cm)/cm.sum()); result={'classes':ck['classes'],'accuracy':accuracy,'confusion_matrix':cm.tolist()}; Path(a.output).write_text(json.dumps(result,indent=2)); print(f"accuracy={accuracy:.4%}")
if __name__=='__main__': main()
