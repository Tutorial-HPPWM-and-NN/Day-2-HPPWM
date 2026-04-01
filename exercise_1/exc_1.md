# Ejercicio 1: Entrenamiento de una Red MLP para SHC-PWM

## Objetivo del Ejercicio
Implementar una red neuronal multicapa (MLP) capaz de aproximar el mapeo entre referencias armónicas de entrada y ángulos óptimos de conmutación de salida.

El modelo recibirá como entradas las variables:

- `m1`
- `m5`
- `m7`
- `m11`
- `m13`
- `m17`
- `m19`
- `phi5`

y deberá predecir los ángulos:

- `alpha_1` hasta `alpha_17`

El propósito de este ejercicio es construir la primera etapa del reemplazo de tablas de búsqueda por una aproximación funcional basada en aprendizaje supervisado.

## Nota metodológica importante
En este problema no se adopta el paradigma clásico de aprendizaje estadístico orientado a generalización sobre datos arbitrarios no vistos. El dataset corresponde a un espacio solución generado numéricamente offline y acotado a una región válida de operación.

Por tanto, el objetivo principal es lograr una replicación de alta fidelidad del manifold de soluciones dentro del dominio entrenado. En consecuencia:

- no se utiliza un conjunto de prueba clásico,
- se deja solo una fracción muy pequeña para validación interna,
- no se busca extrapolación fuera del rango del dataset.

## Estructura del directorio

- `exercise_1_base.py`: archivo base incompleto para el trabajo práctico.
- `exercise_1_solution.py`: solución completa del ejercicio.
- `artifacts/`: carpeta de salida generada automáticamente al ejecutar el script.

## Instrucciones
Abra el archivo `exercise_1_base.py`.

### Tarea 1: Carga y separación del dataset
Complete la lógica para:
1. Cargar el archivo CSV.
2. Separar correctamente las columnas de entrada y salida.
3. Dividir el dataset en conjunto de entrenamiento y una validación interna mínima.

### Tarea 2: Preprocesamiento
Implemente la normalización de las variables de entrada utilizando un escalador apropiado.
Guarde este escalador, ya que será requerido en las etapas posteriores.

### Tarea 3: Definición del modelo
Construya una red MLP densa para regresión, definiendo:
1. Capa de entrada consistente con las 8 variables del problema.
2. Capas ocultas configurables.
3. Capa de salida de dimensión 17 para predecir los ángulos.

### Tarea 4: Entrenamiento y evaluación
Entrene el modelo y evalúe su desempeño.
Reporte al menos:
1. Pérdida final de entrenamiento.
2. Pérdida de validación, si se utiliza.
3. Error de ajuste sobre el conjunto de entrenamiento.
4. Error de ajuste sobre la partición de validación, si existe.

## Análisis de Resultados
Ejecute el script en su entorno.

1. Verifique que la pérdida disminuya de forma estable durante el entrenamiento.
2. Si usa validación, confirme que la discrepancia entre entrenamiento y validación sea baja.
3. Analice si la red logra replicar con alta fidelidad los 17 ángulos de salida.
4. Verifique que se generen correctamente los siguientes artefactos:
   - `mlp_fp32_weights.npz`
   - `input_scaler.joblib`
   - `mlp_config.json`
   - `dataset_splits.npz`
   - `training_history.csv`

## Resultado esperado
Al finalizar este ejercicio, deberá contar con un modelo entrenado en precisión flotante, su configuración estructural y el escalador de entrada almacenado. Todos estos elementos serán utilizados en los ejercicios siguientes.