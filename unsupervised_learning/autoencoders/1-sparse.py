#!/usr/bin/env python3
""" creates a sparse autoencoder """
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for
    each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of
    the latent space representation
    lambtha is the regularization parameter used for L1
    regularization on the encoded output
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model
    The sparse autoencoder model should be compiled using adam
    optimization and binary cross-entropy loss
    All layers should use a relu activation except for the last
    layer in the decoder, which should use sigmoid

    """
    input_layer = layers.Input(shape=(input_dims,))
    encoded = input_layer
    
    for nodes in hidden_layers:
        encoded = layers.Dense(nodes, activation='relu')(encoded)
    
    latent_layer = layers.Dense(latent_dims, activation='relu', activity_regularizer=regularizers.L1(lambtha))(encoded)
    
    encoder = models.Model(input_layer, latent_layer)
    
    # Decoder
    decoded_input = layers.Input(shape=(latent_dims,))
    decoded = decoded_input
    
    for nodes in reversed(hidden_layers):
        decoded = layers.Dense(nodes, activation='relu')(decoded)
    
    output_layer = layers.Dense(input_dims, activation='sigmoid')(decoded)
    
    decoder = models.Model(decoded_input, output_layer)
    
    # Autoencoder
    auto_input = input_layer
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    
    auto = models.Model(auto_input, decoded_output)
    
    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto