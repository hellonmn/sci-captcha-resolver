# Solver Report

## Scope

This report reflects the current files in this folder as reviewed on 25 November 2026. It describes the local image-recognition pipeline and its saved measurements. It does not treat the external case-search integration in `src/server.js` as part of the local model evaluation.

## Local recognition pipeline

The recognizer in `src/predict.js` segments an image into glyphs, classifies them left to right, and returns the recognized text, per-character predictions, confidence scores, bounding boxes, and processing timings. Preprocessing and connected-component segmentation are implemented in `src/preprocess.js` using Sharp and an Otsu threshold. The current classifier supports 15 classes: digits 0–9 and `+`, `-`, `×`, `÷`, and `=`.

The primary classifier is a compact CNN exported to ONNX and run by `onnxruntime-node` on CPU. A local pretrained MNIST ONNX model is used as a fallback for low-confidence digit predictions. The API also contains expression-evaluation logic, but no saved evaluation artifact reports end-to-end expression-answer accuracy.

## Training data and evaluation

`training/dataset_generator.py` creates synthetic glyph images with varied fonts, size, rotation, contrast, noise, blur, and speckles. It creates separate train, validation, and test streams. The README says no application-specific rendering rules were available, so the synthetic distribution may not represent images produced by a particular application.

The saved training history in `models/history.json` reports **94.33% character accuracy** on its test split. The separate `models/evaluation.json` reports **95.07%**. These artifacts disagree and do not identify an identical checkpoint and dataset run. Therefore, there is no single verified accuracy figure to claim until evaluation is rerun against the exact shipped model and test set.

These figures are character-classification accuracy on generated glyphs, not full-image or expression accuracy. The project does not currently report the requested image-level **coverage** and **accuracy** on a held-out set of complete images. The inference path normally chooses a class for each detected glyph; it has no calibrated abstention rule that returns “unsure.” Images that fail validation or segmentation return errors, but those cases have not been counted in a coverage evaluation.

## Performance snapshot

The README records a 500-run benchmark with 20 warmups, using the supplied `3-1` image. The recorded values are:

| Measure | Recorded value |
|---|---:|
| Average end-to-end prediction | 4.966 ms |
| P95 end-to-end prediction | 6.576 ms |
| Preprocessing and segmentation | 4.061 ms average; 5.319 ms P95 |
| Model inference | 0.894 ms average; 1.423 ms P95 |
| Model load | 141.56 ms |
| Peak process RSS | 97,087,488 bytes (about 92.6 MiB) |
| Model bundle size | 46,097 bytes |

The benchmark host reported 12 logical processors and was not pinned to four cores. These results are a recorded snapshot, not a verified four-core acceptance test. They also measure one supplied image repeatedly and do not establish latency across a representative image set.

## Limitations and implementation notes

- Connected-component segmentation can split or merge glyphs incorrectly when strokes touch, images have unusual backgrounds, or the layout is multi-line.
- Synthetic character accuracy does not predict performance on a different image-generation process. Complete-image accuracy may be lower because a full result depends on segmentation and every glyph prediction.
- A confidence score is returned, but the code does not use it to abstain from uncertain predictions; low-confidence digits instead go to the MNIST fallback.
- The INT8 artifact is described in the README as stale after retraining. The current server loads `models/model.onnx` by default, not the INT8 artifact.
- The current `src/server.js` includes routes that contact `www.sci.gov.in`. As a result, the whole application is not local-only and does not satisfy a strict no-third-party-integration requirement, even though model inference itself runs locally. This report does not document how to operate that integration.
- The repository contains an older root-level `server.js` using Tesseract. `npm start` launches `src/server.js`, which is the CNN implementation.

## Conclusion

The folder contains a working local character-classification pipeline and saved synthetic-data and performance artifacts. The current evidence supports only a provisional **94.33% generated-glyph test accuracy** from the training history, with a conflicting **95.07%** evaluation artifact. It does not support image-level coverage/accuracy or a verified four-core performance claim. Those figures should be reported only after a single reproducible evaluation is run against the exact shipped model and representative held-out complete images.
