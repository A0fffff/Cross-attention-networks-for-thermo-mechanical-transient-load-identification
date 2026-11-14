"""Predict loads with the retrained attention model and saved training statistics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Net_v3_Attention import BPPointNet


def _features(path: Path, width: int):
    array = np.loadtxt(path)
    if array.ndim == 1:
        if width != 1:
            raise ValueError(f"Expected {width} columns in {path}")
        array = array.reshape(-1, 1)
    elif array.shape[1] == width:
        pass
    elif array.shape[0] == width:
        array = array.T
    else:
        raise ValueError(
            f"Expected {width} features in {path}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values in {path}")
    return torch.tensor(array, dtype=torch.float32)


def predict(checkpoint_path: Path, strains_file: Path, thermal_file: Path,
            uq_file: Path, target_file: Path | None = None,
            output_file: Path | None = None):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = BPPointNet(**saved["architecture"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    stats = saved["normalization"]

    x = _features(strains_file, 20)
    z1 = _features(thermal_file, 3)
    z2 = _features(uq_file, 2)
    if len({len(x), len(z1), len(z2)}) != 1:
        raise ValueError(
            "Strain, thermal, and uncertainty files have different sample counts")

    x = (x - stats["x_mean"]) / stats["x_std"]
    z1 = (z1 - stats["z1_mean"]) / stats["z1_std"]
    z2 = (z2 - stats["z2_mean"]) / stats["z2_std"]
    with torch.no_grad():
        normalized, attention = model(x, z1, z2, return_weights=True)
        prediction = normalized * stats["y_std"] + stats["y_mean"]
        prediction = (prediction - prediction[0])

    result = pd.DataFrame({"Predicted": prediction[:, 0].numpy()})
    if target_file is not None:
        target = _features(target_file, 1)
        if len(target) != len(prediction):
            raise ValueError(
                "Target and input files have different sample counts")
        result["Real"] = target[:, 0].numpy()
        relative_l2 = float(torch.linalg.vector_norm(prediction - target) /
                            torch.linalg.vector_norm(target).clamp_min(1e-8))
        print(f"Relative L2 error: {relative_l2:.6%}")
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_excel(output_file, index=False)
        print(f"Saved predictions to {output_file}")
    return result, attention


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path,
                        default=script_dir / "Net_Attention_v3_amplitude_candidate.pth")
    parser.add_argument("--case-dir", type=Path,
                        default=script_dir / "testdata" / "3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-target", action="store_true")
    args = parser.parse_args()
    case_dir = args.case_dir
    target = None if args.no_target else case_dir / "F_data.txt"
    output = args.output or case_dir / "Inference_Results_attention.xlsx"
    predict(args.checkpoint, case_dir / "strains.txt", case_dir / "Thermal_Coefs.txt",
            case_dir / "uq_data.txt", target, output)
