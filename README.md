# TF-HAAGCD

## Título: Implementación de una red neuronal MLP para el modelado de la evolución de la temperatura de un fluido por convección natural en un recinto cúbico calentado lateralmente.

#### Autor: Sosa Matias

### Resumen:

En este trabajo, se implementa y evalúa una red neuronal del tipo Multi-Layer Perceptron (MLP) para el modelado de la evolución de la temperatura de un fluido por convección natural en un recinto cúbico calentado lateralmente. Como modelo de referencia, se realizó un ajuste exponencial independiente para la serie temporal de temperatura de cada punto de medición. El MLP fue entrenado mostrando una alta precisión en la predicción de temperaturas en datos no vistos. La comparación directa en los puntos medidos reveló que, si bien el modelo exponencial exhibió un error ligeramente menor en la mayoría de los casos, los valores de error de ambos modelos fueron muy próximos y del mismo orden de magnitud.

La validación clave del estudio se centró en la capacidad de extrapolación del MLP. Para ello, se generaron mapas de isotermas que se compararon cualitativamente con las franjas de correlación obtenidas experimentalmente mediante Interferometría de Speckle Digital (DSPI). A pesar de que el modelo no predijo las temperaturas de forma totalmente precisa en todo el recinto, se encontró una notable similitud cualitativa entre la forma de las isotermas predichas por el MLP y las imágenes de DSPI. Se concluye que esta limitación podría deberse a la escasa cantidad de puntos de medición utilizados para el entrenamiento, lo que sugiere que un dataset más denso podría mejorar significativamente la capacidad de generalización del modelo a todo el campo de temperatura.

### Contenido y uso:

En el directorio *"Datos-Crudos"* se encuentras archivos de excel(*.xslx*) que contienen los datos de temperatura medidos con un datalogger CR23X de forma minutal. Los nombre de estos archivos estan compuestos por 3 partes, separado con un guion. Los primeros 4 cararteres tiene la potencia del calentamiento de la experiencia,La segunda parte es es la fecha de la experiencia en formato "DD-MM-AA". y el ultimo caracter es una "O" o una "X" segun si se cuentas con mediciones en el canal 8(Tc_B).

El programa *"creador_base_datos.py"* se encarga de crear un archivo *.cvs* con los datos que se encuentren en el directorio indicado que tengan la extension *.xslx*, este toma informacion de los nombres del archivo. la salida de este programa es una tabla de datos cuyas columnas con el tiempo transcurrido desde el inicio del calentamiento[h], coordenada x de la temperatura[m], coordenada y de la temperatura[m], Temperatura inicial del recinto[°C], potencia entregada al recinto[W] y Temperatura registrada por la termocupla[°C]. Los archivos *.cvs* en el directorio *Datos* son producto de este programas y con estos se entrena la red.

El programa *"RED-Entrenamiento2.py"* define el MLP utilizado usando la libreria *PyTorch* y luego instancia la red y entrena usando el archivos *DATOS2.csv* destinando el 50% a entrenamiento y el resto para validacion y testeo. Luego del bloque de entrenamiento prueba la red con el conjunto de testeo y genera graficas con las estadisticas. por ultimo genera un archivo *.pth* que guarda el diccionario con el estado de la red generado con el metodo .state_dict() del modelo entrenado.

El programa *"RED-Validacion2.py"* dreconstriye la red y luego carga los datos de uno de los archivos para la validacion y comparacion respecto de un ajuste exponencial $T(t)= y_0 + A_1 * exp(-t/t_1)$.
Luego compara los resultados las predicciones de la red con los datos cargados con los valores del ajuste. Por ultimo genera una grilla d epuntos corrspondiente al interion del recinto y valua el modelo en todos estos puntos y genera mapas de temperatura con isotermas marcadas para poder compararla visualmente con las imagenes de franjas de correlacion que se encuentran en el directorio *Imagenes/Isotermas/DSPI* que corresponden a las medidas del archivo "4.0w-08-01-25_X.xlsx"
