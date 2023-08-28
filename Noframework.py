import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# Cargar datos
df = load_digits()

X = df.data
y = df.target

# Dividir los datos en conjunto de entrenamiento (70%) y conjunto de prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def euclideana(x1, x2):
    # Calcula la distancia euclidiana entre dos puntos
    squared_distance = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return math.sqrt(squared_distance)

def k_nearest_neighbors(X_train, y_train, x_test, k=3):
    # Calcula las distancias euclidianas entre el ejemplo de prueba y todos los ejemplos en el conjunto de entrenamiento
    distancias = [euclideana(x_test, x_train) for x_train in X_train]
    
    # Ordena los índices de las distancias en orden ascendente y selecciona los primeros 'k' índices
    k_indices = sorted(range(len(distancias)), key=lambda i: distancias[i])[:k]
    
    # Obtiene las etiquetas de los 'k' vecinos más cercanos
    k_cercanos = [y_train[i] for i in k_indices]
    
    # Encuentra la etiqueta más común entre los vecinos cercanos
    comun = max(set(k_cercanos), key=k_cercanos.count)
    
    # Retorna la etiqueta predicha para el ejemplo de prueba
    return comun

# Realizar predicciones en los datos de prueba
predictions = [k_nearest_neighbors(X_train, y_train, x) for x in X_test]
print("Predicciones:", predictions)
print("Reales", y_test)
