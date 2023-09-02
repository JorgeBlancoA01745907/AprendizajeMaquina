import math
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar datos
df1 = load_digits()
df2 = load_iris()

# Dividir datos de entrada y de salida para digits
X_1 = df1.data
y_1 = df1.target

# Dividir datos de entrada y de salida para digits
X_2 = df2.data
y_2 = df2.target

# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%) para digits con una semilla random
# Conjunto 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X_1, y_1, test_size=0.3, random_state=random.randint(1,100))
# Conjunto 2
X3_train, X3_test, y3_train, y3_test = train_test_split(X_1, y_1, test_size=0.3, random_state=random.randint(1,100))

# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%) para iris con una semilla random
# Conjunto 1
X2_train, X2_test, y2_train, y2_test = train_test_split(X_2, y_2, test_size=0.3, random_state=random.randint(1,100))
# Conjunto 2
X4_train, X4_test, y4_train, y4_test = train_test_split(X_2, y_2, test_size=0.3, random_state=random.randint(1,100))


# Pasar datos de salida esperados de las pruebas a lista
y1_test = y1_test.tolist()
y2_test = y2_test.tolist()
y3_test = y3_test.tolist()
y4_test = y4_test.tolist()

def euclideana(x1, x2):
    # Calcula la distancia euclidiana entre dos puntos
    squared_distance = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return math.sqrt(squared_distance)

def k_nearest_neighbors(X_train, y_train, x_test, k=3):
    # Función de algoritmo knn
    
    # Lista vacía donde se guardarán las predicciones del modelo
    predicciones = []
    
    # Calcula la predicción (y) para cada valor x de las pruebas
    for x in x_test:
        # Calcula las distancias euclidianas entre el ejemplo de prueba y todos los ejemplos en el conjunto de entrenamiento
        distancias = [euclideana(x, x_train) for x_train in X_train]
        
        # Ordena los índices de las distancias en orden ascendente y selecciona los primeros 'k' índices
        k_indices = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
        
        # Obtiene las etiquetas de los 'k' vecinos más cercanos
        k_cercanos = [y_train[i] for i in k_indices]
        
        # Encuentra la etiqueta más común entre los vecinos cercanos
        comun = max(set(k_cercanos), key=k_cercanos.count)
        
        # Agrega la etiqueta predicha para el valor x de prueba
        predicciones.append(comun)
    
    # Regresa las predicciones del modelo
    return predicciones

# Realizar predicciones en los datos de prueba

# Realiza predicciones para el dataset digits 1
predictions_digits = k_nearest_neighbors(X1_train, y1_train, X1_test)
# Realiza predicciones para le dataset iris 1
predictions_iris = k_nearest_neighbors(X2_train, y2_train, X2_test)
# Medir precisión del modelo con dataset digits 1
precision1 = accuracy_score(y1_test, predictions_digits)
# Medir precisión del modelo con dataset iris 1
precision2 = accuracy_score(y2_test, predictions_iris)
# Matriz de confusion del modelo con dataset digits 1
conf_matrix1 = confusion_matrix(y1_test, predictions_digits)
# Matriz de confusion del modelo con dataset iris 1
conf_matrix2 = confusion_matrix(y2_test, predictions_iris)

# Realiza predicciones para el dataset digits 2
predictions_digits2 = k_nearest_neighbors(X3_train, y3_train, X3_test)
# Realiza predicciones para le dataset iris 2
predictions_iris2 = k_nearest_neighbors(X4_train, y4_train, X4_test)
# Medir precisión del modelo con dataset digits 2
precision3 = accuracy_score(y3_test, predictions_digits2)
# Medir precisión del modelo con dataset iris 2
precision4 = accuracy_score(y4_test, predictions_iris2)
# Matriz de confusion del modelo con dataset digits 2
conf_matrix3 = confusion_matrix(y3_test, predictions_digits2)
# Matriz de confusion del modelo con dataset iris 2
conf_matrix4 = confusion_matrix(y4_test, predictions_iris2)

# Impresión de los datos y métricas en consola
print("Predicciones digits:", predictions_digits)
print("\nReales digits\n", y1_test)
print("\nPrecision digits\n", precision1)
print("\nMatriz de confusion digits: \n", conf_matrix1)
print(classification_report(y1_test, predictions_digits))
print("\nPredicciones iris:\n", predictions_iris)
print("\nReales iris\n", y2_test)
print("\nPrecision iris\n", precision2)
print("\nMatriz de confusion iris: \n", conf_matrix2)
print(classification_report(y2_test, predictions_iris))
print("\nPredicciones digits 2:\n", predictions_digits2)
print("\nReales digits 2\n", y3_test)
print("\nPrecision digits\n", precision3)
print("\nMatriz de confusion digits: \n", conf_matrix3)
print(classification_report(y3_test, predictions_digits2))
print("\nPredicciones iris:\n", predictions_iris2)
print("\nReales iris\n", y4_test)
print("\nPrecision iris\n", precision4)
print("\nMatriz de confusion iris: \n", conf_matrix4)
print(classification_report(y4_test, predictions_iris2))
