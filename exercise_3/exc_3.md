# Ejercicio 3: Cuantización del Modelo a TensorFlow Lite

## Objetivo del Ejercicio
Convertir el modelo podado del Ejercicio 2 a un formato TensorFlow Lite cuantizado en enteros de 8 bits, y comparar el error del modelo en precisión flotante frente al error del modelo cuantizado.

El propósito es evaluar si la degradación introducida por la cuantización post-entrenamiento (PTQ) sigue siendo aceptable para el problema SHC-PWM, antes de pasar a la etapa de despliegue embebido.

## Relación con el flujo del Día 2
Este ejercicio conecta la compresión del modelo con su ejecución en un sistema embebido real. El resultado principal será un archivo `.tflite` cuantizado, que servirá como entrada directa para la inferencia sobre la Zybo Z7 en el Ejercicio 4.

## Estructura del directorio

- `exercise_3_base.py`: archivo base incompleto para el trabajo práctico.
- `exercise_3_solution.py`: solución completa del ejercicio.
- `artifacts/`: carpeta de salida generada automáticamente al ejecutar el script.

## Dependencias previas
Este ejercicio requiere que el Ejercicio 2 haya sido ejecutado previamente, ya que reutiliza:

- `exercise_2/artifacts/mlp_pruned_fp32_weights.npz`
- `exercise_2/artifacts/mlp_config.json`

Además, reutiliza desde el Ejercicio 1:

- `exercise_1/artifacts/input_scaler.joblib`

## Instrucciones
Abra el archivo `exercise_3_base.py`.

### Tarea 1: Reconstrucción del modelo FP32
Complete la lógica para:
1. Cargar la configuración estructural del modelo.
2. Reconstruir la arquitectura.
3. Cargar los pesos podados desde NPZ.
4. Cargar el escalador de entrada.
5. Evaluar el modelo FP32 reconstruido.

### Tarea 2: Conversión a TensorFlow Lite INT8
Implemente la cuantización post-entrenamiento (PTQ), definiendo:
1. El conversor desde el modelo Keras.
2. Un conjunto representativo de calibración.
3. La cuantización completa de entrada y salida en INT8.

### Tarea 3: Inferencia con el intérprete TFLite
Implemente el lazo de inferencia usando el intérprete de TensorFlow Lite:
1. Lectura de detalles de entrada y salida.
2. Cuantización de la entrada.
3. Ejecución del modelo muestra a muestra.
4. De-cuantización de la salida.
5. Construcción del arreglo de predicciones finales.

### Tarea 4: Comparación de métricas
Compare el modelo FP32 y el modelo TFLite cuantizado en términos de:
1. Error respecto al dataset.
2. Diferencia entre predicciones FP32 e INT8.
3. Tamaño de archivo.

## Análisis de Resultados
Ejecute el script en su entorno.

1. Compare el error del modelo flotante y del modelo cuantizado.
2. Analice si la degradación por cuantización es suficientemente pequeña.
3. Revise el tamaño del archivo `.tflite` frente al archivo de pesos flotantes.
4. Verifique la generación de los siguientes artefactos:
   - `model_pruned_int8.tflite`
   - `tflite_metrics_comparison.csv`
   - `sample_predictions_comparison.csv`
   - `mlp_config.json`

## Resultado esperado
Al finalizar este ejercicio, deberá contar con una versión cuantizada del modelo en formato TensorFlow Lite y con métricas comparativas que validen su uso en la siguiente etapa de despliegue sobre la Zybo Z7.