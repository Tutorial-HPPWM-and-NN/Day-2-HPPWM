"""
exercise_4_solution.py
=======================
Exercise 4: Bare-Metal Deployment on Zybo Z7 with TFLite Micro — SOLUTION.
 
This script orchestrates the complete deployment pipeline in four steps:
    1. Reconvert the pruned model for TFLM bare-metal compatibility.
    2. Generate model_data.cc (C++ byte array from the .tflite binary).
    3. Display Vitis project setup instructions.
    4. Cross-validate PC and Zybo Z7 inference results.
 
PREREQUISITES:
    - All Day 2 exercises (1-3) must have been executed successfully.
    - TFmicroZynq cloned: https://github.com/SensorsINI/TFmicroZynq
    - Vivado 2022.2 + Vitis 2022.2 installed with Zynq-7000 support.
    - Zybo Z7-10 connected via micro-USB data cable.
    - Git Bash available for the xxd command.
"""
 
import json
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
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
 
INPUT_COLUMNS      = ["m1", "m5", "m7", "m11", "m13", "m17", "m19", "phi5"]
OUTPUT_COLUMNS     = [f"alpha_{i}" for i in range(1, 18)]
REP_SAMPLES        = 512
SAMPLE_IDX         = 0
 
 
def _rebuild_fp32_model():
    """Helper: reconstructs the pruned FP32 Keras model from saved artifacts."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    data    = np.load(WEIGHTS_PATH)
    keys    = sorted(data.files, key=lambda x: int(x.split("_")[1]))
    weights = [data[k] for k in keys]
 
    layers = [tf.keras.layers.Input(shape=(config["input_dim"],))]
    for units in config["hidden_units"]:
        layers.append(tf.keras.layers.Dense(units, activation=config["hidden_activation"]))
    layers.append(tf.keras.layers.Dense(config["output_dim"], activation=config["output_activation"]))
 
    model = tf.keras.Sequential(layers)
    model.set_weights(weights)
    return model, config
 
 
def step1_convert_model():
    """
    Step 1: Reconvert the pruned model for TFLM bare-metal compatibility.
 
    The model from Exercise 3 uses per-channel quantization, which TFLM
    does not support for dense layers. This step regenerates the model
    with per-tensor quantization using the TFLM-compatible conversion flag.
 
    Output: exercise_3/artifacts/model_pruned_int8_tflm.tflite
    """
    print("=" * 60)
    print("STEP 1: Converting model for TFLite Micro compatibility")
    print("=" * 60)
 
    model, _ = _rebuild_fp32_model()
 
    # Calibration data
    df     = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    X      = scaler.transform(df[INPUT_COLUMNS].to_numpy(dtype=np.float32))
 
    def representative_dataset():
        for i in range(min(REP_SAMPLES, len(X))):
            yield [X[i:i+1].astype(np.float32)]
 
    # Convert with TFLM-compatible per-tensor quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_disable_per_channel_quantization_for_dense_layers = True
 
    tflite_model = converter.convert()
    with open(TFLM_MODEL, "wb") as f:
        f.write(tflite_model)
 
    print(f"Model saved : {TFLM_MODEL}")
    print(f"Size        : {len(tflite_model):,} bytes ({len(tflite_model)/1024:.1f} KB)")
    return True
 
 
def step2_generate_model_cc():
    """
    Step 2: Convert the .tflite binary to a C++ byte array.
 
    Uses xxd if available, otherwise generates the array in Python.
    Output: exercise_3/artifacts/model_data.cc
    """
    print("\n" + "=" * 60)
    print("STEP 2: Generating model_data.cc")
    print("=" * 60)
 
    if not TFLM_MODEL.exists():
        print(f"ERROR: {TFLM_MODEL} not found. Complete Step 1 first.")
        return False
 
    try:
        with open(MODEL_CC, "w") as out:
            subprocess.run(["xxd", "-i", str(TFLM_MODEL)], stdout=out, check=True)
        with open(MODEL_CC) as f:
            first_line = f.readline().strip()
        print(f"Generated : {MODEL_CC}")
        print(f"Array     : {first_line[:60]}...")
    except FileNotFoundError:
        # xxd not available — generate manually with Python
        print("xxd not found. Generating model_data.cc with Python...")
        with open(TFLM_MODEL, "rb") as f:
            data = f.read()
        var_name = TFLM_MODEL.stem.replace("-", "_").replace(".", "_")
        with open(MODEL_CC, "w") as f:
            f.write(f"unsigned char {var_name}[] = {{\n  ")
            f.write(", ".join(f"0x{b:02x}" for b in data))
            f.write("\n};\n")
            f.write(f"unsigned int {var_name}_len = {len(data)};\n")
        print(f"Generated : {MODEL_CC}")
 
    return True
 
 
def step3_vitis_instructions():
    """Step 3: Display Vitis project setup instructions."""
    print("\n" + "=" * 60)
    print("STEP 3: Vitis 2022.2 Project Setup")
    print("=" * 60)
    print(f"\nFull instructions : {ZYBO_DIR / 'README_vitis.md'}")
    print(f"C++ source file   : {ZYBO_DIR / 'main.cc'}")
    print(f"Model array file  : {MODEL_CC}")
    print("\nSummary:")
    print("  1. Generate .xsa from Vivado 2022.2 (Zynq PS only design).")
    print("  2. Create platform project in Vitis from the .xsa.")
    print("  3. Create Empty Application (C++) project.")
    print("  4. Copy TFmicroZynqSrc/, model_data.cc and main.cc to src/.")
    print("  5. Configure include paths, symbols and flags (README_vitis.md).")
    print("  6. Build and run on hardware. Read UART output at 115200 baud.")
    print("\nOnce the Zybo prints the results, proceed to Step 4.")
 
 
def step4_validate(zybo_results_mrad=None):
    """
    Step 4: Cross-validate PC and Zybo inference results.
 
    Args:
        zybo_results_mrad: list of 17 integer mrad values from the Zybo
                           serial terminal. If None, the user is prompted.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Cross-validation (PC vs Zybo)")
    print("=" * 60)
 
    # Collect Zybo results interactively if not provided
    if zybo_results_mrad is None:
        print("\nEnter the 17 alpha values from the Zybo terminal (in mrad),")
        print("comma-separated or one per line. Press Enter twice when done.\n")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        raw = " ".join(lines).replace(",", " ")
        zybo_results_mrad = [int(x) for x in raw.split()]
 
    if len(zybo_results_mrad) != 17:
        print(f"ERROR: Expected 17 values, got {len(zybo_results_mrad)}.")
        return
 
    y_zybo = np.array(zybo_results_mrad) / 1000.0
 
    # Load data and compute reference
    df     = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    x_raw    = df[INPUT_COLUMNS].iloc[SAMPLE_IDX].values
    x_scaled = scaler.transform(x_raw.reshape(1, -1)).astype(np.float32)
    y_true   = df[OUTPUT_COLUMNS].iloc[SAMPLE_IDX].values
 
    print(f"\nOperating point (sample {SAMPLE_IDX}):")
    print(f"  m1={x_raw[0]:.4f}, m5={x_raw[1]:.4f}, phi5={x_raw[7]:.4f}")
 
    # FP32 inference
    model_fp32, _ = _rebuild_fp32_model()
    y_fp32 = model_fp32.predict(x_scaled, verbose=0)[0]
 
    # INT8 TFLite inference (PC)
    with open(TFLM_MODEL, "rb") as f:
        tflite_content = f.read()
    interp = tf.lite.Interpreter(model_content=tflite_content)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x_q = np.round(x_scaled / inp["quantization"][0] + inp["quantization"][1])
    x_q = np.clip(x_q, -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x_q)
    interp.invoke()
    y_q    = interp.get_tensor(out["index"])
    y_int8 = (y_q.astype(np.float32) - out["quantization"][1]) * out["quantization"][0]
 
    # Print comparison table
    print(f"\n{'Angle':<10} {'True':>10} {'FP32':>10} {'INT8':>10} {'Zybo':>10}")
    print("-" * 52)
    for i in range(17):
        print(f"alpha_{i+1:<4} {y_true[i]:>10.4f} {y_fp32[i]:>10.4f} "
              f"{y_int8[0][i]:>10.4f} {y_zybo[i]:>10.4f}")
 
    mae_fp32         = np.mean(np.abs(y_true - y_fp32))
    mae_int8         = np.mean(np.abs(y_true - y_int8[0]))
    mae_zybo         = np.mean(np.abs(y_true - y_zybo))
    mae_int8_vs_zybo = np.mean(np.abs(y_int8[0] - y_zybo))
 
    print(f"\nMAE vs reference:")
    print(f"  FP32 : {mae_fp32*1000:.1f} mrad")
    print(f"  INT8 : {mae_int8*1000:.1f} mrad")
    print(f"  Zybo : {mae_zybo*1000:.1f} mrad")
    print(f"\nINT8 (PC) vs Zybo (hardware): {mae_int8_vs_zybo*1000:.1f} mrad")
 
    if mae_int8_vs_zybo < 0.005:
        print("\n[OK] Deployment validated: Zybo == INT8 within rounding margin.")
    else:
        print("\n[!] Difference > 5 mrad. Check model quantization.")
 
 
if __name__ == "__main__":
    print("Exercise 4: Bare-Metal Deployment on Zybo Z7 with TFLite Micro\n")
 
    step1_convert_model()
    step2_generate_model_cc()
    step3_vitis_instructions()
 
    # Provide Zybo results directly or leave None to enter interactively
    zybo_results_mrad = None   # Replace with your values, e.g.: [270, 331, ...]
    zybo_results_mrad = [270, 331, 514, 593, 776, 846, 1003, 1125,
                     1204, 1291, 1492, 1588, 1727, 1849, 1963, 2128, 2224]
    step4_validate(zybo_results_mrad)
