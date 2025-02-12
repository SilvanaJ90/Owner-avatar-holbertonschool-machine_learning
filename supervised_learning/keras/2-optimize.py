#!/usr/bin/env python3
"""  sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """

    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    Returns: None

    """
    network.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
