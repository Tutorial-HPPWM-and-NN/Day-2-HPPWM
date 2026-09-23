from pathlib import Path
import json
import tempfile
import zipfile

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_PATH = ROOT_DIR / "data" / "dataset_shcpwm_10000.csv"
EXERCISE_1_ARTIFACTS = ROOT_DIR / "exercise_1" / "artifacts"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = EXERCISE_1_ARTIFACTS / "mlp_fp32_weights.npz"
SCALER_PATH = EXERCISE_1_ARTIFACTS / "input_scaler.joblib"
CONFIG_PATH = EXERCISE_1_ARTIFACTS / "mlp_config.json"

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
BATCH_SIZE = 128
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 1e-4
TARGET_SPARSITY = 0.80


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_base_model_from_config(config: dict) -> tf.keras.Model:
    """
    Rebuild the model architecture from the saved configuration.
    """
    # TODO:
    # 1. Create a Sequential model
    # 2. Add the input layer
    # 3. Add hidden Dense layers according to config["hidden_units"]
    # 4. Add the output layer
    # 5. Compile the model
    raise NotImplementedError("Complete the base model reconstruction.")

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
    # 2. Recover the arrays in the correct order: arr_0, arr_1, ...
    raise NotImplementedError("Complete weight loading from NPZ.")


def load_training_artifacts(
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
    # 1. Load the model config JSON
    # 2. Rebuild the model with build_base_model_from_config
    # 3. Load the weights from NPZ
    # 4. Assign them with model.set_weights(...)
    # 5. Load the scaler
    raise NotImplementedError("Complete artifact loading.")


def prepare_inputs(df: pd.DataFrame, scaler):
    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y = df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, y


def evaluate_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, tag: str):
    loss, mae = model.evaluate(X, y, verbose=0)
    y_pred = model.predict(X, verbose=0)

    mse = np.mean((y - y_pred) ** 2)
    mae_np = np.mean(np.abs(y - y_pred))

    print(f"\n{tag} metrics")
    print(f"Loss (Keras): {loss:.8f}")
    print(f"MAE  (Keras): {mae:.8f}")
    print(f"MSE  (NumPy): {mse:.8f}")
    print(f"MAE  (NumPy): {mae_np:.8f}")

    return {
        f"{tag.lower()}_loss": float(loss),
        f"{tag.lower()}_mae": float(mae),
        f"{tag.lower()}_mse_numpy": float(mse),
        f"{tag.lower()}_mae_numpy": float(mae_np),
    }


def get_zipped_file_size_bytes(file_path: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "file.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, arcname=file_path.name)
        return zip_path.stat().st_size


def compute_layer_sparsity(model: tf.keras.Model):
    sparsity_report = []

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        kernel = weights[0]
        if kernel.ndim != 2:
            continue

        zero_count = np.sum(kernel == 0.0)
        total_count = kernel.size
        sparsity = zero_count / total_count

        sparsity_report.append(
            {
                "layer_name": layer.name,
                "shape": kernel.shape,
                "zero_count": int(zero_count),
                "total_count": int(total_count),
                "sparsity": float(sparsity),
            }
        )

    return sparsity_report


def print_sparsity_report(report):
    print("\nLayer sparsity report")
    for item in report:
        print(
            f"{item['layer_name']}: "
            f"shape={item['shape']}, "
            f"sparsity={item['sparsity']:.4f}"
        )


def build_pruned_model(base_model: tf.keras.Model, num_samples: int):
    """
    Wrap the base model with magnitude pruning.
    """
    # TODO:
    # 1. Compute steps_per_epoch
    # 2. Compute end_step
    # 3. Define a PolynomialDecay schedule
    # 4. Apply prune_low_magnitude
    # 5. Compile the pruned model
    raise NotImplementedError("Complete the pruned model definition.")


def save_model_weights_npz(model: tf.keras.Model, output_path: Path):
    weights = model.get_weights()
    weights_dict = {f"arr_{i}": arr for i, arr in enumerate(weights)}
    np.savez(output_path, **weights_dict)


def strip_and_save_model(pruned_model: tf.keras.Model, output_weights_path: Path):
    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    save_model_weights_npz(stripped_model, output_weights_path)
    return stripped_model


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Resolved dataset path: {DATA_PATH}")
    print(f"Resolved weights path: {WEIGHTS_PATH}")
    print(f"Resolved scaler path: {SCALER_PATH}")
    print(f"Resolved config path: {CONFIG_PATH}")

    df = load_dataset(DATA_PATH)
    base_model, scaler, model_config = load_training_artifacts(
        WEIGHTS_PATH,
        SCALER_PATH,
        CONFIG_PATH,
    )
    X_scaled, y = prepare_inputs(df, scaler)

    print("Dataset and training artifacts loaded successfully.")
    print(f"Samples: {len(df)}")
    print(f"Input shape: {X_scaled.shape}")
    print(f"Output shape: {y.shape}")

    baseline_metrics = evaluate_model(base_model, X_scaled, y, "Baseline")
    baseline_size_bytes = get_zipped_file_size_bytes(WEIGHTS_PATH)

    print(f"\nBaseline zipped weights size: {baseline_size_bytes / 1024:.2f} KB")

    pruned_model = build_pruned_model(base_model, num_samples=len(X_scaled))

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]

    history = pruned_model.fit(
        X_scaled,
        y,
        epochs=FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    pruned_weights_path = ARTIFACTS_DIR / "mlp_pruned_fp32_weights.npz"
    pruned_model_stripped = strip_and_save_model(pruned_model, pruned_weights_path)

    with open(ARTIFACTS_DIR / "mlp_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    pruned_metrics = evaluate_model(pruned_model_stripped, X_scaled, y, "Pruned")
    pruned_size_bytes = get_zipped_file_size_bytes(pruned_weights_path)

    print(f"\nPruned zipped weights size: {pruned_size_bytes / 1024:.2f} KB")

    sparsity_report = compute_layer_sparsity(pruned_model_stripped)
    print_sparsity_report(sparsity_report)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(ARTIFACTS_DIR / "pruning_history.csv", index=False)

    pd.DataFrame(sparsity_report).to_csv(
        ARTIFACTS_DIR / "sparsity_report.csv",
        index=False,
    )

    comparison = {
        **baseline_metrics,
        **pruned_metrics,
        "baseline_zipped_weights_size_bytes": baseline_size_bytes,
        "pruned_zipped_weights_size_bytes": pruned_size_bytes,
        "target_sparsity": TARGET_SPARSITY,
    }

    pd.DataFrame([comparison]).to_csv(
        ARTIFACTS_DIR / "metrics_comparison.csv",
        index=False,
    )

    print("\nArtifacts saved in:")
    print(ARTIFACTS_DIR.resolve())

    _ = history


if __name__ == "__main__":
    main()
