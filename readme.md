Autor: Jorge Isidro Blanco Martínez
# k-NN para Clasificación de Dígitos Escritos a Mano

Este repositorio contiene un ejemplo de implementación del algoritmo k-NN (k-Nearest Neighbors) en Python para la clasificación de dígitos escritos a mano utilizando el conjunto de datos `load_digits()` de la biblioteca scikit-learn.

## Contenido

- `knn_digits.py`: El archivo principal que contiene el código para cargar los datos, definir las funciones `euclideana` y `k_nearest_neighbors`, realizar predicciones y mostrar los resultados.

## Requisitos

1. Clonar repositorio de Github
2. Asegúrate de tener Python instalado en tu sistema.
3. Instala las bibliotecas necesarias usando el siguiente comando:

```
pip install scikit-learn
```

## Uso

1. Ejecuta el archivo `Noframework.py`.

El script realizará las siguientes acciones:

1. Cargará el conjunto de datos de dígitos escritos a mano utilizando `load_digits()` de scikit-learn.
2. Dividirá los datos en conjuntos de entrenamiento (70%) y prueba (30%).
3. Definirá la función `euclideana` para calcular la distancia euclidiana entre dos puntos.
4. Definirá la función `k_nearest_neighbors` para implementar el algoritmo k-NN.
5. Realizará predicciones en los datos de prueba utilizando el algoritmo k-NN.
6. Mostrará las predicciones generadas por el algoritmo y las etiquetas reales.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---