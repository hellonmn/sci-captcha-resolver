"""Install a validated ONNX candidate at the application's default model path."""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


def promote(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Candidate and destination paths must differ.")
    if source.suffix.lower() != ".onnx" or not source.is_file():
        raise FileNotFoundError(f"Candidate ONNX model not found: {source}")

    source_classes = source.with_suffix(".classes.json")
    if not source_classes.is_file():
        raise FileNotFoundError(f"Candidate classes file not found: {source_classes}")
    classes = json.loads(source_classes.read_text(encoding="utf-8-sig"))
    if not isinstance(classes, list) or not classes or not all(isinstance(item, str) for item in classes):
        raise ValueError(f"Invalid class list in {source_classes}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar_name = destination.name + ".data"
    candidate = onnx.load(str(source), load_external_data=False)
    locations = {
        entry.value
        for tensor in candidate.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    if len(locations) > 1:
        raise ValueError(f"Candidate references multiple external weight files: {sorted(locations)}")
    source_data = source.parent / next(iter(locations)) if locations else None
    if source_data and not source_data.is_file():
        raise FileNotFoundError(f"Candidate external weights not found: {source_data}")

    with tempfile.TemporaryDirectory(prefix="promote-model-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        staged_model = stage / destination.name
        staged_classes = staged_model.with_suffix(".classes.json")
        staged_data = Path(str(staged_model) + ".data")

        for tensor in candidate.graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = sidecar_name
        onnx.save_model(candidate, str(staged_model))
        if source_data:
            shutil.copyfile(source_data, staged_data)
        shutil.copyfile(source_classes, staged_classes)

        session = ort.InferenceSession(str(staged_model), providers=["CPUExecutionProvider"])
        output_classes = session.get_outputs()[0].shape[-1]
        if isinstance(output_classes, int) and output_classes != len(classes):
            raise ValueError(f"Model output has {output_classes} classes but class file has {len(classes)}")

        destination_classes = destination.with_suffix(".classes.json")
        destination_data = Path(str(destination) + ".data")
        if source_data:
            os.replace(staged_data, destination_data)
        os.replace(staged_classes, destination_classes)
        os.replace(staged_model, destination)

    print(f"Promoted {source} to {destination}")
    print(f"Classes: {destination_classes}")
    if source_data:
        print(f"External weights: {destination_data}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Candidate ONNX file to promote")
    parser.add_argument("--destination", default="models/model.onnx", help="Default model path used by the app")
    args = parser.parse_args()
    promote(Path(args.model), Path(args.destination))


if __name__ == "__main__":
    main()
