"""
exercise_4_base.py
==================
Exercise 4: Bare-Metal Deployment on Zybo Z7 with TFLite Micro.

This script orchestrates the complete deployment pipeline:
    Step 1 — Reconvert the model for TFLM compatibility.
    Step 2 — Generate the C++ model array (model_data.cc).
    Step 3 — Display Vitis project setup instructions.
    Step 4 — Cross-validate PC and Zybo inference results.

PREREQUISITES:
    - Exercises 1, 2 and 3 completed (artifacts must exist).
    - TFmicroZynq cloned: https://github.com/SensorsINI/TFmicroZynq
    - Vitis 2022.2 with Zynq-7000 support.
    - Zybo Z7-10 connected via micro-USB data cable.
    - Git Bash available for the xxd command.
"""

import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
ROOT_DIR     = BASE_DIR.parent
ZYBO_DIR     = BASE_DIR / "zybo"

DATA_PATH    = ROOT_DIR / "data" / "dataset_shcpwm_10000.csv"
WEIGHTS_PATH = ROOT_DIR / "exercise_2" / "artifacts" / "mlp_pruned_fp32_weights.npz"
SCALER_PATH  = ROOT_DIR / "exercise_1" / "artifacts" / "input_scaler.joblib"
CONFIG_PATH  = ROOT_DIR / "exercise_2" / "artifacts" / "mlp_config.json"
ARTIFACTS_E3 = ROOT_DIR / "exercise_3" / "artifacts"
TFLM_MODEL   = ARTIFACTS_E3 / "model_pruned_int8_tflm.tflite"
MODEL_CC     = ARTIFACTS_E3 / "model_data.cc"

INPUT_COLUMNS  = ["m1", "m5", "m7", "m11", "m13", "m17", "m19", "phi5"]
OUTPUT_COLUMNS = [f"alpha_{i}" for i in range(1, 18)]
SAMPLE_IDX     = 0


def step1_convert_model():
    """
    Step 1: Reconvert the pruned model for TFLM bare-metal compatibility.

    The model from Exercise 3 uses per-channel quantization, which TFLM
    does not support for dense layers. This step reconverts it with the
    per-tensor quantization flag required by TFLM.
    """
    print("=" * 60)
    print("STEP 1: Converting model for TFLite Micro compatibility")
    print("=" * 60)

    # TODO 1: Rebuild the pruned FP32 model from mlp_config.json and
    # mlp_pruned_fp32_weights.npz (same pattern as Exercise 3).
    # Then convert it with:
    #   converter._experimental_disable_per_channel_quantization_for_dense_layers = True
    # Use 512 representative samples from the dataset for calibration.
    # Save the result to TFLM_MODEL.

    raise NotImplementedError("Complete Step 1: convert model for TFLM.")


def step2_generate_model_cc():
    """
    Step 2: Convert the .tflite model to a C++ byte array.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Generating model_data.cc")
    print("=" * 60)

    if not TFLM_MODEL.exists():
        print("ERROR: TFLM model not found. Run Step 1 first.")
        return False

    # TODO 2: Run xxd via subprocess to generate MODEL_CC from TFLM_MODEL.
    # If xxd is not available, generate the C array manually using Python
    # by reading the binary and writing hex bytes to a .cc file.

    raise NotImplementedError("Complete Step 2: generate model_data.cc.")


def step3_vitis_instructions():
    """Step 3: Display Vitis project setup instructions."""
    print("\n" + "=" * 60)
    print("STEP 3: Vitis 2022.2 Project Setup")
    print("=" * 60)
    print(f"\nFull instructions : {ZYBO_DIR / 'README_vitis.md'}")
    print(f"C++ source        : {ZYBO_DIR / 'main.cc'}")
    print(f"Model array       : {MODEL_CC}")
    print("\nKey steps:")
    print("  1. Generate .xsa from Vivado (Zynq PS only design).")
    print("  2. Create platform + Empty Application (C++) in Vitis.")
    print("  3. Copy TFmicroZynqSrc/, model_data.cc and main.cc to src/.")
    print("  4. Configure includes, symbols and flags (see README_vitis.md).")
    print("  5. Build and run. Read UART at 115200 baud.")
    print("\nOnce the Zybo prints results, proceed to Step 4.")


def step4_validate(zybo_results_mrad=None):
    """
    Step 4: Cross-validate PC (FP32 + INT8) vs Zybo inference results.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Cross-validation (PC vs Zybo)")
    print("=" * 60)

    # TODO 3: If zybo_results_mrad is None, prompt the user to enter the
    # 17 mrad values printed by the Zybo serial terminal.
    # Then run inference with the FP32 model and the INT8 TFLite model
    # on sample SAMPLE_IDX of the dataset, and print a comparison table
    # with columns: True | FP32 | INT8 | Zybo.
    # Report MAE for each column vs the true reference, and the MAE
    # between INT8 (PC) and Zybo (hardware).

    raise NotImplementedError("Complete Step 4: collect Zybo results and validate.")


if __name__ == "__main__":
    print("Exercise 4: Bare-Metal Deployment on Zybo Z7 with TFLite Micro\n")

    step1_convert_model()
    step2_generate_model_cc()
    step3_vitis_instructions()

    zybo_results_mrad = None  # Fill with Zybo values or leave None for input prompt
    step4_validate(zybo_results_mrad)