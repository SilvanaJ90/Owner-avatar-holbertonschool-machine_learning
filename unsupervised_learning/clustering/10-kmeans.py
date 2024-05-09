#!/usr/bin/env python3
""" Performs K-means on a dataset: """
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
