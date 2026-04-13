# SRC/modelos/baseline_cnn.py

import tensorflow as tf

# Opción B (tu elección): import plano "config"
# Asegúrate de ejecutar con CWD que permita encontrar config.py (en tu caso lo está cargando desde SRC/config.py)
from config import IMAGEN_SHAPE, NUM_CLASES, TASA_APRENDIZAJE_BASELINE


def construir_cnn_basico() -> tf.keras.Model:
    """
    CNN baseline mínima:
    Input -> [Conv + MaxPool] x3 -> GAP -> Dense softmax
    Diseñada para clasificación multiclase con etiquetas int32 (sparse).
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
        name="cnn_baseline_minimo",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE_BASELINE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return modelo
