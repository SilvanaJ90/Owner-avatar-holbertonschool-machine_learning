#!/usr/bin/env python3
""" rains a model using mini-batch gradient descent """
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        verbose=True, shuffle=False):
    """
    Doc

    """

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
