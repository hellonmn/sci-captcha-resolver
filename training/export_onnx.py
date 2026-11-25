import argparse
import sys
from pathlib import Path
import torch
from train import TinyCNN

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='models/best.pt'); p.add_argument('--output',default='models/model.onnx'); a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model=TinyCNN().eval(); model.load_state_dict(ck['model']); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
torch.onnx.export(model,torch.zeros(1,1,32,32),a.output,input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}},opset_version=18)
Path(a.output).with_suffix('.classes.json').write_text(__import__('json').dumps(ck['classes']))
print(f'Exported {a.output} ({Path(a.output).stat().st_size} bytes)')
