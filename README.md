# TF-HAAGCD

## Título: Implementación de una red neuronal MLP para el modelado de la evolución de la temperatura de un fluido por convección natural en un recinto cúbico calentado lateralmente.

### Autor: Sosa Matias

En este trabajo, se implementa y evalúa una red neuronal del tipo Multi-Layer Perceptron (MLP) para el modelado de la evolución de la temperatura de un fluido por convección natural en un recinto cúbico calentado lateralmente. Como modelo de referencia, se realizó un ajuste exponencial independiente para la serie temporal de temperatura de cada punto de medición. El MLP fue entrenado mostrando una alta precisión en la predicción de temperaturas en datos no vistos. La comparación directa en los puntos medidos reveló que, si bien el modelo exponencial exhibió un error ligeramente menor en la mayoría de los casos, los valores de error de ambos modelos fueron muy próximos y del mismo orden de magnitud.

La validación clave del estudio se centró en la capacidad de extrapolación del MLP. Para ello, se generaron mapas de isotermas que se compararon cualitativamente con las franjas de correlación obtenidas experimentalmente mediante Interferometría de Speckle Digital (DSPI). A pesar de que el modelo no predijo las temperaturas de forma totalmente precisa en todo el recinto, se encontró una notable similitud cualitativa entre la forma de las isotermas predichas por el MLP y las imágenes de DSPI. Se concluye que esta limitación podría deberse a la escasa cantidad de puntos de medición utilizados para el entrenamiento, lo que sugiere que un dataset más denso podría mejorar significativamente la capacidad de generalización del modelo a todo el campo de temperatura.

En el directorio "Datos-Crudos" se encuentras archivos de excel(.xslx) que contienen los datos de temperatura medidos con un datalogger CR23X de forma minutal. Los nombre de estos archivos estan compuestos por 3 partes, separado con un guion. Los primeros 4 cararteres tiene la potencia del calentamiento de la experiencia,La segunda parte es es la fecha de la experiencia en formato "DD-MM-AA". y el ultimo caracter es una "O" o una "X" segun si se cuentas con mediciones en el canal 8(Tc_B).

