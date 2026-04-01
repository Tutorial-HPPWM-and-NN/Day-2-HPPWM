# Ejercicio 2: Compresión del Modelo mediante Pruning

## Objetivo del Ejercicio
Reducir la complejidad efectiva de la red entrenada en el Ejercicio 1 mediante poda de pesos (pruning), manteniendo un error de aproximación bajo sobre el mismo espacio solución utilizado durante el entrenamiento.

En este ejercicio se aplicará pruning por magnitud sobre la red MLP ya entrenada, seguido por una etapa de ajuste fino (fine-tuning) para recuperar precisión.

## Relación con el flujo del Día 2
Este ejercicio actúa como etapa intermedia entre el entrenamiento flotante y la cuantización. La idea es demostrar que el modelo puede hacerse más esparso antes de exportarlo a TensorFlow Lite, lo que permite estudiar el compromiso entre compacidad y error.

## Estructura del directorio

- `exercise_2_base.py`: archivo base incompleto para el trabajo práctico.
- `exercise_2_solution.py`: solución completa del ejercicio.
- `artifacts/`: carpeta de salida generada automáticamente al ejecutar el script.

## Dependencias previas
Este ejercicio requiere que el Ejercicio 1 haya sido ejecutado previamente, ya que reutiliza:

- `exercise_1/artifacts/mlp_fp32_weights.npz`
- `exercise_1/artifacts/input_scaler.joblib`
- `exercise_1/artifacts/mlp_config.json`

## Instrucciones
Abra el archivo `exercise_2_base.py`.

### Tarea 1: Carga de artefactos previos
Complete la lógica para:
1. Cargar la configuración estructural del modelo.
2. Reconstruir la arquitectura.
3. Cargar los pesos desde un archivo NPZ.
4. Asignarlos al modelo con `set_weights`.
5. Cargar el escalador de entrada.
6. Reaplicar el preprocesamiento sobre el dataset completo.

### Tarea 2: Definición del modelo podado
Implemente el wrapping del modelo base usando pruning por magnitud.
Debe definir:
1. Un esquema de poda progresiva.
2. Una sparsity objetivo.
3. La recompilación del modelo para ajuste fino.

### Tarea 3: Fine-tuning del modelo podado
Entrene el modelo podado durante unas pocas épocas adicionales para recuperar precisión y consolidar la estructura esparsa.

### Tarea 4: Comparación antes y después
Compare el modelo base y el modelo podado en términos de:
1. Error de ajuste.
2. Tamaño de pesos serializados.
3. Sparsity por capa.

## Análisis de Resultados
Ejecute el script en su entorno.

1. Compare las métricas del modelo original y del modelo podado.
2. Verifique si la degradación del error sigue siendo aceptable para el problema SHC-PWM.
3. Revise el reporte de sparsity por capa para confirmar que la poda fue realmente aplicada.
4. Verifique la generación de los siguientes artefactos:
   - `mlp_pruned_fp32_weights.npz`
   - `mlp_config.json`
   - `pruning_history.csv`
   - `sparsity_report.csv`
   - `metrics_comparison.csv`

## Resultado esperado
Al finalizar este ejercicio, deberá contar con una versión podada del modelo entrenado, junto con métricas comparativas que permitan decidir si la compresión es compatible con la siguiente etapa de cuantización.