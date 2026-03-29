# SRC/modelos/cnn_bn_dropout.py
import tensorflow as tf
from config import IMAGEN_SHAPE, NUM_CLASES, TASA_APRENDIZAJE_BASELINE


def construir_modelo() -> tf.keras.Model:
    """
    CNN con BatchNorm + Dropout:
    reduce sobreajuste.
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
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(inputs, outputs, name="cnn_bn_dropout")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE_BASELINE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo
