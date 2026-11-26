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
- `src/` — image preprocessing, segmentation, model inference, folder solver (`src/solve.js`, `src/solver.js`), and Express server code.
- `models/` — trained model artifacts and saved evaluation records.
- `benchmarks/` — inference benchmark script.
- `tests/` — automated checks for preprocessing, segmentation, model artifacts, solver, and API helpers.
- `PROJECT_REPORT.md` — the investigation and solver report, including API observations and evidence screenshots.

## Running the solver on a folder of images

Run the local ONNX solver on all images inside a folder with one command:

```bash
npm run solve -- path/to/images
```

Or invoke it directly:

```bash
node src/solve.js path/to/images --threshold 0.8 --json --out results.json
```

Optional flags: `--threshold <0..1>` (marks low-confidence predictions as `unsure`), `--model <path>`, and `--out <file.json>`.

## Train, evaluate, and promote the model

Use this workflow to add new application samples and update the model used by the API.

### 1. Download more sample images

The downloader continues numbering after the highest numbered image already in `data/samples`:

```powershell
npm run download-images -- --count 20
```

### 2. Label the images

Add one row per image to `data/samples/labels.csv`, keeping its existing header and rows. The columns are `file,expression,left,operator,right`; enter the image filename, full expression, and its parts.

### 3. Fine-tune a new candidate

Start from the latest candidate checkpoint. For example, if v2 is the latest trained candidate:

```powershell
.venv\Scripts\python.exe training\train.py --samples-csv data\samples\labels.csv --init-checkpoint models\sample_candidate_v2\best.pt --output models\sample_candidate_v3 --epochs 20 --batch-size 64 --patience 5
```

For the first fine-tuning run, omit `--init-checkpoint` to use the default `models/best.pt`. Keep each run's output directory so its checkpoint and history remain available. The trainer extracts glyphs with the app's segmentation, keeps each expression together in the train/validation split, and balances labeled and synthetic glyphs during fine-tuning.

### 4. Export the candidate to ONNX

```powershell
.venv\Scripts\python.exe training\export_onnx.py --checkpoint models\sample_candidate_v3\best.pt --output models\model.sample_candidate_v3.onnx
```

### 5. Compare predictions with the labels

```powershell
npm run evaluate:samples -- --model models\model.sample_candidate_v3.onnx --out models\sample_candidate_v3\evaluation.json
```

The report includes exact expression accuracy, character error rate, and per-image results. The full-CSV evaluation includes training images; use held out expression metrics in `history.json` to estimate performance on unseen expressions. The current CSV has real examples of digits and `+`/`-`; other operator classes continue to use synthetic examples.

### 6. Promote the candidate to the API's default model

After reviewing a candidate, promote it with:

```powershell
.venv\Scripts\python.exe training\promote_model.py --model models\model.sample_candidate_v3.onnx
```

The script validates the ONNX file and class list, then installs the model as `models\model.onnx` with the matching `models\model.onnx.data` and `models\model.classes.json` files. Don't rename only the ONNX file by hand; it references its external weights sidecar.

### 7. Restart the API

Stop the running server with `Ctrl+C`, then start it again:

```powershell
npm start
```

The API loads `models\model.onnx` on startup. For the next training round, use the latest approved candidate's `best.pt` as `--init-checkpoint`, such as `models\sample_candidate_v3\best.pt`.

## Important deployment note

The current `src/server.js` includes integration with `www.sci.gov.in`, including CAPTCHA-related search handling. As a result, this checkout is **not a local-only image-recognition service** and should not be published or deployed as an own-app-only OCR API. The model's local CPU inference does not make the full server local-only.

The report's CAPTCHA behavior and API details are based on the supplied manual observations and request/response samples. They should be shared only with the authorized site owner, with session cookies, tokens, nonces, challenge IDs, and personal case data removed.
