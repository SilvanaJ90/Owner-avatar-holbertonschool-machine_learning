#!/usr/bin/env python3
""" Doc """
import numpy as np


def kmeans(X, k, iterations=1000):
    n, d = X.shape

    # Inicializar los centroides de los clusters
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # Inicializar las asignaciones de cluster
    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        # Calcular las distancias entre los puntos y los centroides
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Asignar cada punto al cluster del centroide más cercano
        clss = np.argmin(distances, axis=1)

        # Actualizar los centroides
        new_C = np.array([X[clss == i].mean(axis=0) for i in range(k)])

        # Si no hay cambio en los centroides, detener el bucle
        if np.allclose(C, new_C):
            break

        C = new_C

    # Verificar si algún cluster está vacío
    if np.any([np.sum(clss == i) == 0 for i in range(k)]):
        return None, None

    return C, clss
