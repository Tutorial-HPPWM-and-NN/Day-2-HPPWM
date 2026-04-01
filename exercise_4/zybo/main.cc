/**
 * main.cc
 * =======
 * Bare-metal inference of the HPPWM neural network on the ARM Cortex-A9
 * of the Zybo Z7, using TensorFlow Lite Micro (TFLM) as the inference engine.
 *
 * The model (model_data.cc) contains the MLP quantized to INT8 with
 * per-tensor quantization, compatible with TFLM bare-metal execution.
 *
 * Network dimensions:
 *   Inputs  (8): m1, m5, m7, m11, m13, m17, m19, phi5
 *   Outputs (17): alpha_1 ... alpha_17 (switching angles in radians)
 *
 * The test input corresponds to sample 0 of the training dataset, scaled
 * with the StandardScaler fitted in Exercise 1 (Day 2). In a real application,
 * the scaler parameters would be embedded as constants and applied to live
 * harmonic references from the control loop.
 *
 * Output format: angles are printed in milliradians (mrad) over UART at
 * 115200 baud for easy comparison with the PC validation script.
 *
 * Build instructions: see zybo/README_vitis.md
 */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "xil_printf.h"

// Model array generated with: xxd -i model_pruned_int8_tflm.tflite > model_data.cc
// Variable name depends on the filename — check model_data.cc if linker fails.
extern const unsigned char model_pruned_int8_tflm_tflite[];
extern const unsigned int  model_pruned_int8_tflm_tflite_len;

// Tensor arena: memory pool for TFLM activations and intermediate tensors.
// Increase if AllocateTensors() returns an error.
constexpr int kTensorArenaSize = 128 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Network I/O dimensions
constexpr int kNumInputs  = 8;   // m1, m5, m7, m11, m13, m17, m19, phi5
constexpr int kNumOutputs = 17;  // alpha_1 ... alpha_17

int main() {
    tflite::InitializeTarget();

    xil_printf("=== HPPWM TFLite Micro on Zybo Z7 ===\r\n");

    // ── Load model ────────────────────────────────────────────────────────────
    const tflite::Model* model = tflite::GetModel(model_pruned_int8_tflm_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        xil_printf("ERROR: Model schema version mismatch.\r\n");
        return -1;
    }

    // ── Register operations used by the model ─────────────────────────────────
    // The MLP uses: fully connected layers, ReLU activations, and
    // quantize/dequantize nodes at the input and output boundaries.
    using OpResolver = tflite::MicroMutableOpResolver<4>;
    OpResolver op_resolver;
    op_resolver.AddFullyConnected();
    op_resolver.AddRelu();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();

    // ── Build interpreter ─────────────────────────────────────────────────────
    tflite::MicroInterpreter interpreter(
        model, op_resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        xil_printf("ERROR: AllocateTensors() failed. Increase kTensorArenaSize.\r\n");
        return -1;
    }

    // ── Test input (sample 0 from dataset, pre-scaled with StandardScaler) ────
    //
    // Raw values: m1=0.85, m5=0.01, m7=0, m11=0, m13=0, m17=0, m19=0, phi5=-pi
    //
    // In a real control application, replace these constants with the output
    // of the StandardScaler applied to live harmonic references:
    //   x_scaled[i] = (x_raw[i] - scaler_mean[i]) / scaler_std[i]
    //
    // The scaler parameters (mean and std per feature) are saved in
    // exercise_1/artifacts/input_scaler.joblib and can be extracted and
    // embedded as constant arrays in a production firmware.
    float input_scaled[kNumInputs] = {
        -2.384e-08f,  // m1    (raw: 0.85)
        -1.7145f,     // m5    (raw: 0.01)
         0.0f,        // m7    (raw: 0.00)
         0.0f,        // m11   (raw: 0.00)
         0.0f,        // m13   (raw: 0.00)
         0.0f,        // m17   (raw: 0.00)
         0.0f,        // m19   (raw: 0.00)
        -1.7113f      // phi5  (raw: -3.1416 rad)
    };

    // Ground-truth angles from the dataset (for on-device MAE reporting)
    float true_angles[kNumOutputs] = {
        0.2575f, 0.3296f, 0.5178f, 0.5826f, 0.7642f, 0.8343f,
        0.9921f, 1.0736f, 1.1910f, 1.2517f, 1.4655f, 1.5502f,
        1.7001f, 1.8199f, 1.9369f, 2.0869f, 2.1774f
    };

    // ── Quantize and load input tensor ────────────────────────────────────────
    TfLiteTensor* input   = interpreter.input(0);
    float         inp_s   = input->params.scale;
    int           inp_zp  = input->params.zero_point;

    for (int i = 0; i < kNumInputs; i++) {
        int32_t q = (int32_t)(input_scaled[i] / inp_s + inp_zp);
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        input->data.int8[i] = (int8_t)q;
    }

    // ── Run inference ─────────────────────────────────────────────────────────
    if (interpreter.Invoke() != kTfLiteOk) {
        xil_printf("ERROR: Invoke() failed.\r\n");
        return -1;
    }

    // ── Dequantize and print results ──────────────────────────────────────────
    TfLiteTensor* output  = interpreter.output(0);
    float         out_s   = output->params.scale;
    int           out_zp  = output->params.zero_point;

    xil_printf("\r\n=== HPPWM Inference (Zybo Z7 bare-metal) ===\r\n");
    xil_printf("%-12s %10s %10s\r\n", "Angle", "Pred(mrad)", "True(mrad)");
    xil_printf("--------------------------------------------\r\n");

    float mae_sum = 0.0f;
    for (int i = 0; i < kNumOutputs; i++) {
        float pred  = (output->data.int8[i] - out_zp) * out_s;
        float truth = true_angles[i];
        float err   = pred - truth;
        if (err < 0.0f) err = -err;
        mae_sum += err;

        xil_printf("alpha_%-6d %10d %10d\r\n",
                   i + 1,
                   (int)(pred  * 1000.0f),
                   (int)(truth * 1000.0f));
    }

    float mae = mae_sum / (float)kNumOutputs;
    xil_printf("--------------------------------------------\r\n");
    xil_printf("MAE = %d mrad\r\n", (int)(mae * 1000.0f));
    xil_printf("============================================\r\n");

    return 0;
}
