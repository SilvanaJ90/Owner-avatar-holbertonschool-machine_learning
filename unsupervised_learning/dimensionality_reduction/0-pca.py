#!/usr/bin/env python3
""" That performs PCA on a dataset: """
import numpy as np


def pca(X, var=0.95):
    """ Doc """
    # Calculate the covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate the cumulative explained variance
    explained_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    # Determine the number of components to retain
    nd = np.argmax(explained_variance >= var) + 1
    
    # Select the top nd eigenvectors
    W = sorted_eigenvectors[:, :nd]
    
    return W
