# Tutorial: Deep Learning y Despliegue Embebido para HPPWM - Día 2

Este repositorio contiene el material práctico del segundo día del tutorial, enfocado en el uso de redes neuronales para aproximar espacios de solución de modulación programada y su posterior despliegue en sistemas embebidos.

El objetivo central es reemplazar tablas de búsqueda masivas por modelos neuronales compactos, entrenados sobre datasets generados offline, para finalmente ejecutar inferencia en un sistema SoC real basado en la Zybo Z7.

La progresión de los ejercicios está diseñada para cubrir el flujo completo de implementación:
entrenamiento supervisado, compresión del modelo, cuantización a TensorFlow Lite y despliegue final sobre el procesador ARM de la plataforma embebida.

## Estructura del Repositorio

El material práctico está dividido en cuatro carpetas secuenciales. Cada directorio contiene un script base (`_base.py`) para el trabajo en sala, un script con la solución completa y un archivo de instrucciones locales (README).

* **Ejercicio_1 (Entrenamiento de una MLP):** Implementación de una red neuronal multicapa para aproximar los ángulos de disparo a partir de las referencias armónicas del problema SHC-PWM. Incluye partición de datos, normalización, entrenamiento y evaluación.
* **Ejercicio_2 (Compresión del Modelo):** Aplicación de estrategias de reducción de complejidad, como poda de pesos, para disminuir el tamaño efectivo de la red y estudiar el impacto sobre el error de regresión.
* **Ejercicio_3 (Cuantización a TFLite):** Conversión del modelo entrenado a TensorFlow Lite cuantizado. Se comparan métricas antes y después de la cuantización para evaluar la degradación inducida.
* **Ejercicio_4 (Despliegue en Zybo Z7):** Ejecución del modelo TFLite cuantizado sobre el procesador ARM del SoC Zybo Z7, validando inferencia embebida y latencia de ejecución.

## Dataset Utilizado

El conjunto de datos utilizado en este día corresponde a un archivo CSV con referencias armónicas de entrada y los ángulos óptimos de salida:

**Entradas**
- `m1`
- `m5`
- `m7`
- `m11`
- `m13`
- `m17`
- `m19`
- `phi5`

**Salidas**
- `alpha_1` a `alpha_17`

## Requisitos y Configuración del Entorno

Para ejecutar los laboratorios de esta sesión, se requiere un entorno Python 3.9+ (o superior). Siga estos pasos para preparar su entorno:

1. Clone o descargue este repositorio.
2. Abra una terminal en la carpeta raíz del Día 2.
3. Instale las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install -r requirements.txt