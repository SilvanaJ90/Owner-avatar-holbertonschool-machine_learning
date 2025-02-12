#!/usr/bin/env python3
""" Test neural Network"""


import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """
    Arg:
        - network is the network model to test
        - data is the input data to test the model with
        - labels are the correct one-hot labels of data
        - verbose is a boolean that determines if output
    should be printed during the testing process
    - Returns: the loss and accuracy of the model with
    the testing data, respectively

    """
    return network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )
