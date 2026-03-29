# SRC/modelos/cnn_simple.py
import tensorflow as tf
from config import IMAGEN_SHAPE, NUM_CLASES, TASA_APRENDIZAJE_BASELINE


def construir_modelo() -> tf.keras.Model:
    """
    CNN simple:
    Conv2D + MaxPool x3 + GAP + Dense
    """
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGEN_SHAPE),

            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASES, activation="softmax"),
        ],
        name="cnn_simple",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE_BASELINE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return modelo
