# Exercise 4: Bare-Metal Deployment on Zybo Z7 with TFLite Micro

## Objective

Deploy the quantized neural network model onto the ARM Cortex-A9 processor
of the Zybo Z7 in bare-metal mode, using TensorFlow Lite Micro (TFLM) as
the inference engine. The exercise validates that the bare-metal hardware
inference produces results equivalent to the quantized software inference
on PC, closing the complete Day 2 pipeline.

## Relation to the Day 2 Pipeline

- **Exercise 1:** FP32 MLP training.
- **Exercise 2:** Model compression via pruning (80% sparsity).
- **Exercise 3:** PTQ quantization to INT8 for TFLite standard.
- **Exercise 4 (this):** TFLM reconversion + bare-metal deployment on Zybo Z7.

## Prerequisites

All previous exercises must have been executed. The following artifacts
are required:

- `exercise_1/artifacts/input_scaler.joblib`
- `exercise_2/artifacts/mlp_pruned_fp32_weights.npz`
- `exercise_2/artifacts/mlp_config.json`
- `exercise_3/artifacts/model_pruned_int8.tflite`
- `data/dataset_shcpwm_10000.csv`

Additionally:

- Vivado 2022.2 with Zynq-7000 device support.
- Vitis 2022.2.
- Zybo Z7-10 connected via micro-USB data cable.
- Install Git Bash from [here](https://git-scm.com/install/windows) to be able to use the xxd command.
- Once gitbash installed, run thew following command at DOS: "git clone https://github.com/SensorsINI/TFmicroZynq".
    - Run the line inside the folder where you want to clone the repo. 
    - Command xxd transform the file .tflite to C++. THis is required to run it into Vitis. 

## Repository Structure

```
exercise_4/
├── exc_4.md                <- this file
├── exercise_4_base.py      <- orchestrator base script (with TODOs)
├── exercise_4_solution.py  <- complete solution script
└── zybo/
    ├── main.cc             <- C++ source for the Vitis project
    └── README_vitis.md     <- step-by-step Vivado + Vitis setup guide
```

## Instructions

Open `exercise_4_base.py` and complete the three tasks.

### Task 1: Reconvert the Model for TFLM Compatibility

The model from Exercise 3 uses per-channel quantization, which TFLM does
not support for dense layers. Reconvert it with the flag:

    converter._experimental_disable_per_channel_quantization_for_dense_layers = True

Save the result as `exercise_3/artifacts/model_pruned_int8_tflm.tflite`.

### Task 2: Generate the C++ Model Array

Convert the .tflite binary to a C++ byte array. 
Go to the root folder of "exercise_3". Click right bottom and open git bash. Then in Git Bash run:

    cd exercise_3/artifacts
    xxd -i model_pruned_int8_tflm.tflite > model_data.cc
### Task 3: Create Zybo Project.
Go to zybo folere [here](https://github.com/Tutorial-HPPWM-and-NN/Day-2-HPPWM/tree/main/exercise_4/zybo) and follow the instructions of README_vitis.md.

### Task 4: Cross-Validation

After running the C++ program on the Zybo Z7:

1. Copy the 17 alpha values (in mrad) from the UART serial terminal.
2. Fill in zybo_results_mrad in the script.
3. Run to compare FP32, INT8 and Zybo columns.

## Expected Output Format

The Zybo serial terminal should print the 17 predicted angles alongside
the ground-truth reference values. The cross-validation table should show
that the Zybo column closely matches the INT8 column:

```
=== HPPWM Inference (Zybo Z7 bare-metal) ===
Angle        Pred(mrad) True(mrad)
--------------------------------------------
alpha_1             XXX        XXX
...
alpha_17           XXXX       XXXX
--------------------------------------------
MAE = XX mrad
============================================
```

The PC cross-validation script should then print:

```
Angle            True       FP32       INT8       Zybo
----------------------------------------------------
alpha_1        X.XXXX     X.XXXX     X.XXXX     X.XXXX
...
alpha_17       X.XXXX     X.XXXX     X.XXXX     X.XXXX

MAE vs reference:
  FP32 : XX.X mrad
  INT8 : XX.X mrad
  Zybo : XX.X mrad

INT8 (PC) vs Zybo (hardware): X.X mrad
[OK] Deployment validated: Zybo == INT8 within rounding margin.
```

The exact values depend on the model trained in your environment.
The key result to verify is that the difference between INT8 (PC) and
Zybo (hardware) is less than 5 mrad.

## Analysis

1. MAE degrades from FP32 (19.1 mrad) to INT8/Zybo (~23-24 mrad) due to
   80% pruning and INT8 quantization combined.
2. The difference between INT8 (PC) and Zybo (hardware) is ~0.4 mrad,
   attributable to integer rounding in the UART print. This confirms
   the deployment is correct.
3. Is 23 mrad MAE acceptable for SHC-PWM? How does this angular error
   translate to harmonic distortion in the output voltage spectrum?
