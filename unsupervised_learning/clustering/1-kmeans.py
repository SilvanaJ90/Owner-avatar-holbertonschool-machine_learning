#!/usr/bin/env python3
""" Doc """
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Doc"""
    n, d = X.shape

    # Inicializar los centroides de los clusters
    C = initialize(X, k)

    if C is None:
        return None, None

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

    return C, clss
