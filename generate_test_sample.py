"""
generate_test_sample.py
=======================
Helper script to extract test samples from the SHC-PWM dataset and
format them directly as C++ code for embedded deployment (Day 2 and Day 3).

Usage:
    python generate_test_sample.py             # Uses sample 0 by default
    python generate_test_sample.py --sample 5  # Uses row 5 from the dataset
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_shcpwm_10000.csv"
SCALER_PATHS = [
    BASE_DIR / "data" / "input_scaler.joblib",
    BASE_DIR / "exercise_1" / "artifacts" / "input_scaler.joblib",
]
SCALER_PATH = next((p for p in SCALER_PATHS if p.exists()), BASE_DIR / "data" / "input_scaler.joblib")

INPUT_COLUMNS = ["m1", "m5", "m7", "m11", "m13", "m17", "m19", "phi5"]
OUTPUT_COLUMNS = [f"alpha_{i}" for i in range(1, 18)]


def generate_cpp_code(sample_idx: int = 0):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)

    if sample_idx < 0 or sample_idx >= len(df):
        raise ValueError(f"sample_idx must be between 0 and {len(df)-1}")

    row = df.iloc[sample_idx]
    x_raw = row[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y_true = row[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

    # 1. Day 2: Scaled float inputs (via StandardScaler)
    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]

    # 2. Day 3: INT8 inputs (QuantIdentity 8-bit symmetric, scale = 1/127)
    scale_int8 = 1.0 / 127.0
    x_int8 = np.clip(np.round(x_scaled / scale_int8), -128, 127).astype(np.int8)

    print("=" * 70)
    print(f"  EXTRACTED TEST SAMPLE #{sample_idx}")
    print("=" * 70)
    print("\n[1] Physical Inputs (raw values from CSV):")
    for col, val in zip(INPUT_COLUMNS, x_raw):
        print(f"    {col:<6} = {val:.6f}")

    print("\n[2] Ground-Truth Angles (raw radians & milliradians):")
    for i, val in enumerate(y_true, 1):
        print(f"    alpha_{i:<2} = {val:.6f} rad  ({val*1000.0:.1f} mrad)")

    print("\n" + "=" * 70)
    print("  DAY 2 C++ CODE (for exercise_4/zybo/main.cc)")
    print("=" * 70)
    print("    float input_scaled[kNumInputs] = {")
    for col, s_val, r_val in zip(INPUT_COLUMNS, x_scaled, x_raw):
        print(f"        {s_val:11.6f}f,  // {col:<5} (raw: {r_val:.4f})")
    print("    };\n")

    print("    float true_angles[kNumOutputs] = {")
    formatted_angles = [f"{v:.4f}f" for v in y_true]
    for i in range(0, len(formatted_angles), 6):
        print("        " + ", ".join(formatted_angles[i:i+6]) + ("," if i+6 < len(formatted_angles) else ""))
    print("    };")

    print("\n" + "=" * 70)
    print("  DAY 3 C++ CODE (for exercise_3/vitis/main.cc)")
    print("=" * 70)
    print("static const int8_t test_input[INPUT_DIM] = {")
    int8_str = ", ".join(str(v) for v in x_int8)
    print(f"    {int8_str}")
    print("};\n")

    print("static const float test_true_angles[OUTPUT_DIM] = {")
    for i in range(0, len(formatted_angles), 6):
        print("        " + ", ".join(formatted_angles[i:i+6]) + ("," if i+6 < len(formatted_angles) else ""))
    print("};")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract test sample and format as C++ code.")
    parser.add_argument("--sample", type=int, default=0, help="Row index in the dataset (default: 0)")
    args = parser.parse_args()

    generate_cpp_code(args.sample)
