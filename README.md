# Arithmetic Image Recognition

This project contains a character-recognition model for arithmetic images. It uses a small CNN exported to ONNX, with Node.js and ONNX Runtime for CPU inference. Python/PyTorch is used for model training. The dataset generator creates synthetic glyph images with variations in fonts, size, rotation, contrast, and noise.

## Current model

The recognizer segments a horizontal image into glyphs, orders them from left to right, and predicts one of 15 classes: digits `0–9` and the operators `+`, `-`, `×`, `÷`, and `=`. A local pretrained MNIST model is also present for low-confidence digit predictions.

## Evaluation status

The saved training history reports **94.33% character accuracy** on a generated test split. Another saved evaluation file reports **95.07%**. Because these results do not identify the same evaluation run, neither should be presented as the single verified accuracy figure.

These are character-level results on generated glyphs. They do not measure complete-image coverage or complete-expression accuracy, and they do not establish accuracy on a separate real-world image set. The inference code does not currently use a calibrated `unsure` threshold.

The project report records a benchmark snapshot of **4.966 ms average** total prediction time and about **92.6 MiB** peak process memory. That run used one image on a machine reporting 12 logical processors; it was not a four-core acceptance test.

The model files, including the ONNX external weights file and the MNIST model, total about **320,849 bytes**. The arithmetic ONNX model references `models/model.onnx.data`, so keep those two files together.

## Project files

- `training/` — synthetic data generation, training, evaluation, ONNX export, and quantization source.
- `src/` — image preprocessing, segmentation, model inference, and Express server code.
- `models/` — trained model artifacts and saved evaluation records.
- `benchmarks/` — inference benchmark script.
- `tests/` — automated checks for preprocessing, segmentation, model artifacts, and API helpers.
- `PROJECT_REPORT.md` — the investigation and solver report, including API observations and evidence screenshots.

## Important deployment note

The current `src/server.js` includes integration with `www.sci.gov.in`, including CAPTCHA-related search handling. As a result, this checkout is **not a local-only image-recognition service** and should not be published or deployed as an own-app-only OCR API. The model's local CPU inference does not make the full server local-only.

The report's CAPTCHA behavior and API details are based on the supplied manual observations and request/response samples. They should be shared only with the authorized site owner, with session cookies, tokens, nonces, challenge IDs, and personal case data removed.
