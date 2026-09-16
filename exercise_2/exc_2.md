# Exercise 2: Model Compression via Pruning

## Objective
Reduce the effective complexity of the network trained in Exercise 1 through
weight pruning, while keeping the approximation error low over the same
solution space used during training.

This exercise applies magnitude pruning to the already-trained MLP, followed
by a fine-tuning stage to recover accuracy.

## Relation to the Day 2 Pipeline
This exercise acts as an intermediate stage between floating-point training
and quantization. The idea is to demonstrate that the model can be made
sparser before exporting it to TensorFlow Lite, allowing us to study the
trade-off between compactness and error.

## Directory Structure

- `exercise_2_base.py`: incomplete base file for in-class work.
- `exercise_2_solution.py`: complete reference solution.
- `artifacts/`: output folder generated automatically when the script runs.

## Prerequisites
This exercise requires Exercise 1 to have been run first, since it reuses:

- `exercise_1/artifacts/mlp_fp32_weights.npz`
- `exercise_1/artifacts/input_scaler.joblib`
- `exercise_1/artifacts/mlp_config.json`

## Instructions
Open `exercise_2_base.py`.

### Task 1: Load Previous Artifacts
Complete the logic to:
1. Load the model's structural configuration.
2. Reconstruct the architecture.
3. Load the weights from an NPZ file.
4. Assign them to the model with `set_weights`.
5. Load the input scaler.
6. Reapply preprocessing to the full dataset.

### Task 2: Define the Pruned Model
Implement the wrapping of the base model using magnitude pruning.
You must define:
1. A progressive pruning schedule.
2. A target sparsity.
3. Recompilation of the model for fine-tuning.

### Task 3: Fine-Tune the Pruned Model
Train the pruned model for a few additional epochs to recover accuracy and
consolidate the sparse structure.

### Task 4: Before/After Comparison
Compare the base model and the pruned model in terms of:
1. Fit error.
2. Serialized weight file size.
3. Sparsity per layer.

## Results Analysis
Run the script in your environment.

1. Compare the metrics of the original model and the pruned model.
2. Check whether the error degradation remains acceptable for the
   SHC-PWM problem.
3. Review the per-layer sparsity report to confirm that pruning was
   actually applied.
4. Verify that the following artifacts are generated:
   - `mlp_pruned_fp32_weights.npz`
   - `mlp_config.json`
   - `pruning_history.csv`
   - `sparsity_report.csv`
   - `metrics_comparison.csv`

## Expected Outcome
By the end of this exercise, you should have a pruned version of the
trained model, along with comparative metrics that let you decide whether
the compression is compatible with the next quantization stage.
