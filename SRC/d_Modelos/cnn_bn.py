# SRC/modelos/cnn_bn.py
import tensorflow as tf
from config import IMAGEN_SHAPE, NUM_CLASES, TASA_APRENDIZAJE_BASELINE


def construir_modelo() -> tf.keras.Model:
    """
    CNN con Batch Normalization:
    mejora estabilidad del entrenamiento.
    """
    inputs = tf.keras.layers.Input(shape=IMAGEN_SHAPE)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(inputs, outputs, name="cnn_bn")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE_BASELINE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo
