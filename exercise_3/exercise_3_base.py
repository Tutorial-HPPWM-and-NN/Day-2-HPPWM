from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_PATH = ROOT_DIR / "data" / "dataset_shcpwm_10000.csv"
EXERCISE_2_ARTIFACTS = ROOT_DIR / "exercise_2" / "artifacts"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = EXERCISE_2_ARTIFACTS / "mlp_pruned_fp32_weights.npz"
SCALER_PATH = ROOT_DIR / "exercise_1" / "artifacts" / "input_scaler.joblib"
CONFIG_PATH = EXERCISE_2_ARTIFACTS / "mlp_config.json"

TFLITE_MODEL_PATH = ARTIFACTS_DIR / "model_pruned_int8.tflite"

INPUT_COLUMNS = [
    "m1",
    "m5",
    "m7",
    "m11",
    "m13",
    "m17",
    "m19",
    "phi5",
]

OUTPUT_COLUMNS = [f"alpha_{i}" for i in range(1, 18)]

RANDOM_SEED = 42
LEARNING_RATE = 1e-4
REPRESENTATIVE_SAMPLES = 512


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_model_from_config(config: dict) -> tf.keras.Model:
    """
    Rebuild the dense regression model from the saved configuration.
    """
    # TODO:
    # 1. Create a Sequential model
    # 2. Add the input layer
    # 3. Add hidden Dense layers using config["hidden_units"]
    # 4. Add the output layer
    # 5. Compile the model
    raise NotImplementedError("Complete the model reconstruction logic.")


def compile_regression_model(model: tf.keras.Model, learning_rate: float = LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def load_weights_from_npz(weights_path: Path):
    # TODO:
    # 1. Load the NPZ file
    # 2. Recover the arrays in the correct order
    raise NotImplementedError("Complete NPZ weight loading.")


def load_artifacts(
    weights_path: Path,
    scaler_path: Path,
    config_path: Path,
):
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # TODO:
    # 1. Load the model configuration JSON
    # 2. Rebuild the model
    # 3. Load the weights from NPZ
    # 4. Assign weights using model.set_weights(...)
    # 5. Load the scaler
    raise NotImplementedError("Complete artifact loading.")


def prepare_inputs(df: pd.DataFrame, scaler):
    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y = df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y


def evaluate_fp32_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray):
    loss, mae = model.evaluate(X, y, verbose=0)
    y_pred = model.predict(X, verbose=0)

    mse = np.mean((y - y_pred) ** 2)
    mae_np = np.mean(np.abs(y - y_pred))

    print("\nFP32 model metrics")
    print(f"Loss (Keras): {loss:.8f}")
    print(f"MAE  (Keras): {mae:.8f}")
    print(f"MSE  (NumPy): {mse:.8f}")
    print(f"MAE  (NumPy): {mae_np:.8f}")

    return y_pred, {
        "fp32_loss": float(loss),
        "fp32_mae": float(mae),
        "fp32_mse_numpy": float(mse),
        "fp32_mae_numpy": float(mae_np),
    }


def representative_dataset_generator(X_scaled: np.ndarray, num_samples: int = REPRESENTATIVE_SAMPLES):
    """
    Yield representative calibration samples for PTQ.
    """
    # TODO:
    # 1. Select up to num_samples from X_scaled
    # 2. Yield one sample at a time with shape (1, input_dim)
    raise NotImplementedError("Complete the representative dataset generator.")


def convert_to_tflite_int8(model: tf.keras.Model, X_scaled: np.ndarray, output_path: Path):
    """
    Convert the FP32 model to fully quantized INT8 TFLite.
    """
    # TODO:
    # 1. Create the converter from the Keras model
    # 2. Enable default optimizations
    # 3. Set the representative dataset
    # 4. Force INT8 built-in ops
    # 5. Set INT8 input and output types
    # 6. Convert and save the .tflite model
    raise NotImplementedError("Complete the TFLite conversion.")


def run_tflite_inference(tflite_model_path: Path, X_scaled: np.ndarray):
    """
    Run inference with the TFLite interpreter and return dequantized outputs.
    """
    # TODO:
    # 1. Create and allocate the TFLite interpreter
    # 2. Read input/output tensor details
    # 3. Quantize each input sample if needed
    # 4. Invoke the interpreter sample by sample
    # 5. Dequantize outputs if needed
    # 6. Return predictions as float32
    raise NotImplementedError("Complete the TFLite inference loop.")


def evaluate_tflite_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    print("\nTFLite INT8 metrics")
    print(f"MSE  (NumPy): {mse:.8f}")
    print(f"MAE  (NumPy): {mae:.8f}")

    return {
        "tflite_int8_mse_numpy": float(mse),
        "tflite_int8_mae_numpy": float(mae),
    }


def get_file_size_bytes(file_path: Path) -> int:
    return file_path.stat().st_size


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Resolved dataset path: {DATA_PATH}")
    print(f"Resolved weights path: {WEIGHTS_PATH}")
    print(f"Resolved scaler path: {SCALER_PATH}")
    print(f"Resolved config path: {CONFIG_PATH}")

    df = load_dataset(DATA_PATH)
    model, scaler, model_config = load_artifacts(
        WEIGHTS_PATH,
        SCALER_PATH,
        CONFIG_PATH,
    )

    X_scaled, y = prepare_inputs(df, scaler)

    print("Dataset and model artifacts loaded successfully.")
    print(f"Samples: {len(df)}")
    print(f"Input shape: {X_scaled.shape}")
    print(f"Output shape: {y.shape}")

    fp32_predictions, fp32_metrics = evaluate_fp32_model(model, X_scaled, y)

    convert_to_tflite_int8(model, X_scaled, TFLITE_MODEL_PATH)
    tflite_predictions = run_tflite_inference(TFLITE_MODEL_PATH, X_scaled)
    tflite_metrics = evaluate_tflite_predictions(y, tflite_predictions)

    diff_metrics = {
        "fp32_vs_tflite_mse_numpy": float(np.mean((fp32_predictions - tflite_predictions) ** 2)),
        "fp32_vs_tflite_mae_numpy": float(np.mean(np.abs(fp32_predictions - tflite_predictions))),
    }

    fp32_size_bytes = get_file_size_bytes(WEIGHTS_PATH)
    tflite_size_bytes = get_file_size_bytes(TFLITE_MODEL_PATH)

    comparison = {
        **fp32_metrics,
        **tflite_metrics,
        **diff_metrics,
        "fp32_weights_file_size_bytes": fp32_size_bytes,
        "tflite_model_size_bytes": tflite_size_bytes,
    }

    pd.DataFrame([comparison]).to_csv(
        ARTIFACTS_DIR / "tflite_metrics_comparison.csv",
        index=False,
    )

    n_samples_to_export = min(100, len(df))
    sample_df = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples_to_export),
        }
    )

    for i, col in enumerate(OUTPUT_COLUMNS):
        sample_df[f"{col}_fp32"] = fp32_predictions[:n_samples_to_export, i]
        sample_df[f"{col}_int8"] = tflite_predictions[:n_samples_to_export, i]
        sample_df[f"{col}_abs_error"] = np.abs(
            fp32_predictions[:n_samples_to_export, i] - tflite_predictions[:n_samples_to_export, i]
        )

    sample_df.to_csv(ARTIFACTS_DIR / "sample_predictions_comparison.csv", index=False)

    with open(ARTIFACTS_DIR / "mlp_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print("\nArtifacts saved in:")
    print(ARTIFACTS_DIR.resolve())


if __name__ == "__main__":
    main()