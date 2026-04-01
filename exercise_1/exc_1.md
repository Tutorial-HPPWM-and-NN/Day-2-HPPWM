# Exercise 1: MLP Training for SHC-PWM

## Objective

Implement a multi-layer perceptron (MLP) capable of approximating the mapping
between the harmonic reference inputs and the optimal switching angles of the
SHC-PWM problem.

The model receives as inputs:
- `m1`, `m5`, `m7`, `m11`, `m13`, `m17`, `m19` — harmonic magnitudes
- `phi5` — phase reference for the 5th harmonic

and must predict the switching angles:
- `alpha_1` through `alpha_17`

## Methodological Note

This problem does not follow the classical statistical learning paradigm
oriented toward generalization over arbitrary unseen data. The dataset is
an analytically exact solution manifold generated offline by the hybrid
optimizer from Day 1 Exercise 4.

Therefore, the primary objective is high-fidelity replication of the solution
manifold within the trained domain. Consequently:
- No test split is used.
- Only a very small validation fraction (1%) is kept for internal monitoring.
- Extrapolation outside the training range is not expected and would produce
  invalid switching patterns.

## Instructions

Open `exercise_1_base.py` and complete the following tasks.

### Task 1: Load and Split the Dataset

1. Load the CSV file from the `data/` folder.
2. Separate input and output columns correctly.
3. Split into training and a small validation subset.

### Task 2: Input Preprocessing

Implement input normalization using a StandardScaler fitted on the training
data. Save the scaler — it is required in all subsequent exercises.

### Task 3: Define the Model

Build a dense MLP for regression:
1. Input layer consistent with the 8 input variables.
2. Configurable hidden layers (default: [128, 128, 64] with ReLU).
3. Linear output layer of dimension 17.
4. Compile with Adam optimizer and MSE loss.

### Task 4: Train and Evaluate

Train the model and report:
1. Final training loss.
2. Validation loss (if applicable).
3. MAE on the training set.
4. MAE on the validation set (if applicable).

## Analysis

1. Verify that the training loss decreases steadily.
2. If using validation, confirm that the train/val gap remains small.
3. Analyze whether the network replicates the 17 output angles with
   high fidelity.
4. Verify that the following artifacts are generated in `artifacts/`:
   - `mlp_fp32_weights.npz`
   - `input_scaler.joblib`
   - `mlp_config.json`
   - `dataset_splits.npz`
   - `training_history.csv`

## Expected Result

A trained model in FP32 precision, its structural configuration, and the
input scaler — all stored as artifacts for use in the following exercises.

## Files

- `exercise_1_base.py` — incomplete script for in-class work.
- `exercise_1_solution.py` — complete reference solution.
