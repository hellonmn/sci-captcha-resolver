import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
p=argparse.ArgumentParser(); p.add_argument('--input',default='models/model.onnx'); p.add_argument('--output',default='models/model.int8.onnx'); a=p.parse_args()
quantize_dynamic(a.input,a.output,weight_type=QuantType.QInt8)
print(f'Quantized model: {Path(a.output).stat().st_size} bytes')
