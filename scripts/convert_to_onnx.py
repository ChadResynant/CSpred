#!/usr/bin/env python3
"""
Convert scikit-learn models to ONNX format for GPU-accelerated inference.

Usage:
    python scripts/convert_to_onnx.py [--models-dir models/]

Requirements:
    pip install onnx skl2onnx
"""

import argparse
import os
import sys

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import joblib
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install onnx skl2onnx joblib")
    sys.exit(1)

ATOMS = ["H", "HA", "C", "CA", "CB", "N"]
LEVELS = ["R0", "R1", "R2"]


def convert_model(sav_path, onnx_path, verbose=True):
    """Convert a single sklearn model to ONNX format."""
    if not os.path.exists(sav_path):
        if verbose:
            print(f"  Skipping {sav_path} (not found)")
        return False

    try:
        model = joblib.load(sav_path)
        n_features = model.n_features_in_

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        if verbose:
            print(f"  Converted: {sav_path} -> {onnx_path}")
            print(f"    Features: {n_features}, Model type: {type(model).__name__}")
        return True

    except Exception as e:
        if verbose:
            print(f"  Error converting {sav_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert sklearn models to ONNX")
    parser.add_argument("--models-dir", default="models/",
                        help="Directory containing .sav model files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing ONNX files")
    args = parser.parse_args()

    models_dir = args.models_dir
    if not os.path.isdir(models_dir):
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    print(f"Converting models in {models_dir}")
    print("=" * 50)

    converted = 0
    skipped = 0
    failed = 0

    for atom in ATOMS:
        print(f"\n{atom}:")
        for level in LEVELS:
            sav_path = os.path.join(models_dir, f"{atom}_{level}.sav")
            onnx_path = os.path.join(models_dir, f"{atom}_{level}.onnx")

            if os.path.exists(onnx_path) and not args.force:
                print(f"  Skipping {atom}_{level} (ONNX exists, use --force to overwrite)")
                skipped += 1
                continue

            if convert_model(sav_path, onnx_path):
                converted += 1
            else:
                failed += 1

    print("\n" + "=" * 50)
    print(f"Summary: {converted} converted, {skipped} skipped, {failed} failed")

    if converted > 0:
        print("\nTo use GPU inference, install onnxruntime-gpu:")
        print("  pip install onnxruntime-gpu")


if __name__ == "__main__":
    main()
