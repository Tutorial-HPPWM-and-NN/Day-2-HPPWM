from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "dataset_shcpwm_10000.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
EPOCHS = 200
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.01

MODEL_CONFIG = {
    "hidden_units": [128, 128, 64],
    "hidden_activation": "relu",
    "output_activation": "linear",
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = VALIDATION_SIZE,
    random_seed: int = RANDOM_SEED,
):
    """
    Split the dataset into training and validation subsets.

    In this workflow, the goal is faithful approximation of an offline-generated
    solution manifold inside a bounded operating domain. A very small validation
    split is kept only for internal monitoring.
    """
    # TODO:
    # 1. If val_size <= 0, return X, None, y, None
    # 2. Otherwise split into training and validation subsets
    raise NotImplementedError("Complete the dataset split logic.")


def scale_inputs(
    X_train: np.ndarray,
    X_val: np.ndarray | None,
):
    """
    Fit the scaler on training data and transform the available subsets.
    """
    # TODO:
    # 1. Create a StandardScaler
    # 2. Fit on X_train
    # 3. Transform X_train
    # 4. Transform X_val if it exists
    raise NotImplementedError("Complete the input scaling logic.")


def build_mlp(
    input_dim: int,
    output_dim: int,
    model_config: dict,
) -> tf.keras.Model:
    """
    Build a configurable dense MLP for multi-output regression.
    """
    # TODO:
    # 1. Create a Sequential model
    # 2. Add the input layer
    # 3. Add one Dense layer per hidden_units entry
    # 4. Add the output layer
    # 5. Compile with Adam, MSE, and MAE
    raise NotImplementedError("Complete the model definition.")


def evaluate_model(
    model: tf.keras.Model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    split_name: str,
):
    loss, mae = model.evaluate(X_eval, y_eval, verbose=0)
    y_pred = model.predict(X_eval, verbose=0)

    mse = np.mean((y_eval - y_pred) ** 2)
    mae_np = np.mean(np.abs(y_eval - y_pred))

    print(f"\n{split_name} results")
    print(f"Loss (Keras): {loss:.8f}")
    print(f"MAE  (Keras): {mae:.8f}")
    print(f"MSE  (NumPy): {mse:.8f}")
    print(f"MAE  (NumPy): {mae_np:.8f}")

    return {
        f"{split_name.lower()}_loss": float(loss),
        f"{split_name.lower()}_mae": float(mae),
        f"{split_name.lower()}_mse_numpy": float(mse),
        f"{split_name.lower()}_mae_numpy": float(mae_np),
    }


def save_model_weights_npz(model: tf.keras.Model, output_path: Path):
    weights = model.get_weights()
    weights_dict = {f"arr_{i}": arr for i, arr in enumerate(weights)}
    np.savez(output_path, **weights_dict)


def save_training_artifacts(
    model: tf.keras.Model,
    scaler,
    input_dim: int,
    output_dim: int,
    history: tf.keras.callbacks.History,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
):
    save_model_weights_npz(model, ARTIFACTS_DIR / "mlp_fp32_weights.npz")
    joblib.dump(scaler, ARTIFACTS_DIR / "input_scaler.joblib")

    full_config = {
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_units": MODEL_CONFIG["hidden_units"],
        "hidden_activation": MODEL_CONFIG["hidden_activation"],
        "output_activation": MODEL_CONFIG["output_activation"],
    }

    with open(ARTIFACTS_DIR / "mlp_config.json", "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)

    np.savez(
        ARTIFACTS_DIR / "dataset_splits.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=np.array([]) if X_val is None else X_val,
        y_val=np.array([]) if y_val is None else y_val,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(ARTIFACTS_DIR / "training_history.csv", index=False)


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Resolved dataset path: {DATA_PATH}")
    df = load_dataset(DATA_PATH)

    X = df[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    y = df[OUTPUT_COLUMNS].to_numpy(dtype=np.float32)

    print("Dataset loaded successfully.")
    print(f"Samples: {len(df)}")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    X_train, X_val, y_train, y_val = split_dataset(X, y)

    scaler, X_train_scaled, X_val_scaled = scale_inputs(X_train, X_val)

    model = build_mlp(
        input_dim=X_train_scaled.shape[1],
        output_dim=y_train.shape[1],
        model_config=MODEL_CONFIG,
    )

    model.summary()

    validation_data = None
    callbacks = []

    if X_val_scaled is not None and y_val is not None:
        validation_data = (X_val_scaled, y_val)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6,
            ),
        ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=validation_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    train_metrics = evaluate_model(model, X_train_scaled, y_train, "Train")

    val_metrics = {}
    if X_val_scaled is not None and y_val is not None:
        val_metrics = evaluate_model(model, X_val_scaled, y_val, "Validation")

    save_training_artifacts(
        model=model,
        scaler=scaler,
        input_dim=X_train_scaled.shape[1],
        output_dim=y_train.shape[1],
        history=history,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    print("\nArtifacts saved in:")
    print(ARTIFACTS_DIR.resolve())

    print("\nFinal metrics:")
    for key, value in {**train_metrics, **val_metrics}.items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()