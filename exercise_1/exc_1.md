# Exercise 1: Training an MLP Network for SHC-PWM

## Objective
Implement a multilayer perceptron (MLP) capable of approximating the mapping
between input harmonic references and optimal output switching angles.

The model will receive the following input variables:

- `m1`
- `m5`
- `m7`
- `m11`
- `m13`
- `m17`
- `m19`
- `phi5`

and must predict the angles:

- `alpha_1` through `alpha_17`

The purpose of this exercise is to build the first stage of replacing
lookup tables with a functional approximation based on supervised learning.

## Important Methodological Note
This problem does not follow the classical statistical-learning paradigm
oriented toward generalization over arbitrary unseen data. The dataset
corresponds to a solution space generated numerically offline and bounded
to a valid operating region.

Therefore, the main objective is to achieve a high-fidelity replication of
the solution manifold within the trained domain. Consequently:

- no classical test set is used,
- only a very small fraction is held out for internal validation,
- extrapolation outside the dataset range is not the goal.

## Directory Structure

- `exercise_1_base.py`: incomplete base file for in-class work.
- `exercise_1_solution.py`: complete reference solution.
- `artifacts/`: output folder generated automatically when the script runs.

## Instructions
Open `exercise_1_base.py`.

### Task 1: Load and Split the Dataset
Complete the logic to:
1. Load the CSV file.
2. Correctly separate the input and output columns.
3. Split the dataset into a training set and a minimal internal validation set.

### Task 2: Preprocessing
Implement normalization of the input variables using an appropriate scaler.
Save this scaler, since it will be required in later stages.

### Task 3: Model Definition
Build a dense MLP for regression, defining:
1. An input layer consistent with the 8 problem variables.
2. Configurable hidden layers.
3. An output layer of dimension 17 to predict the angles.

### Task 4: Training and Evaluation
Train the model and evaluate its performance.
Report at least:
1. Final training loss.
2. Validation loss, if used.
3. Fit error on the training set.
4. Fit error on the validation split, if any.

## Results Analysis
Run the script in your environment.

1. Verify that the loss decreases steadily during training.
2. If using validation, confirm that the gap between training and
   validation is small.
3. Analyze whether the network reproduces the 17 output angles with high
   fidelity.
4. Verify that the following artifacts are generated correctly:
   - `mlp_fp32_weights.npz`
   - `input_scaler.joblib`
   - `mlp_config.json`
   - `dataset_splits.npz`
   - `training_history.csv`

## Expected Outcome
By the end of this exercise, you should have a model trained in floating
point, its structural configuration, and the stored input scaler. All of
these will be used in the following exercises.
