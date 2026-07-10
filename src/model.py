from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers


def build_q_network(input_shape: Tuple[int, int, int], n_actions: int) -> keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 8, strides=4, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = layers.Conv2D(64, 4, strides=2, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
    outputs = layers.Dense(int(n_actions), activation=None, kernel_initializer='he_uniform')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
