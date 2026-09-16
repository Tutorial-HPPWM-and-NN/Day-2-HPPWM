# Exercise 3: Quantizing the Model to TensorFlow Lite

## Objective
Convert the pruned model from Exercise 2 into an 8-bit integer quantized
TensorFlow Lite format, and compare the error of the floating-point model
against the quantized model.

The purpose is to evaluate whether the degradation introduced by
Post-Training Quantization (PTQ) remains acceptable for the SHC-PWM
problem, before moving on to embedded deployment.

## Relation to the Day 2 Pipeline
This exercise connects model compression with execution on a real embedded
system. The main output is a quantized `.tflite` file, which will serve as
the direct input for inference on the Zybo Z7 in Exercise 4.

## Directory Structure

- `exercise_3_base.py`: incomplete base file for in-class work.
- `exercise_3_solution.py`: complete reference solution.
- `artifacts/`: output folder generated automatically when the script runs.

## Prerequisites
This exercise requires Exercise 2 to have been run first, since it reuses:

- `exercise_2/artifacts/mlp_pruned_fp32_weights.npz`
- `exercise_2/artifacts/mlp_config.json`

It also reuses from Exercise 1:

- `exercise_1/artifacts/input_scaler.joblib`

## Instructions
Open `exercise_3_base.py`.

### Task 1: Reconstruct the FP32 Model
Complete the logic to:
1. Load the model's structural configuration.
2. Reconstruct the architecture.
3. Load the pruned weights from NPZ.
4. Load the input scaler.
5. Evaluate the reconstructed FP32 model.

### Task 2: Convert to TensorFlow Lite INT8
Implement Post-Training Quantization (PTQ), defining:
1. The converter from the Keras model.
2. A representative calibration dataset.
3. Full INT8 quantization of the input and output.

### Task 3: Inference with the TFLite Interpreter
Implement the inference loop using the TensorFlow Lite interpreter:
1. Read input and output details.
2. Quantize the input.
3. Run the model sample by sample.
4. Dequantize the output.
5. Build the final predictions array.

### Task 4: Metrics Comparison
Compare the FP32 model and the quantized TFLite model in terms of:
1. Error against the dataset.
2. Difference between FP32 and INT8 predictions.
3. File size.

## Results Analysis
Run the script in your environment.

1. Compare the error of the floating-point model and the quantized model.
2. Analyze whether the quantization degradation is small enough.
3. Check the `.tflite` file size against the floating-point weights file.
4. Verify that the following artifacts are generated:
   - `model_pruned_int8.tflite`
   - `tflite_metrics_comparison.csv`
   - `sample_predictions_comparison.csv`
   - `mlp_config.json`

## Expected Outcome
By the end of this exercise, you should have a quantized version of the
model in TensorFlow Lite format, along with comparative metrics validating
its use in the next deployment stage on the Zybo Z7.
