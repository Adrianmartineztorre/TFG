"""
Modelo baseline de red neuronal convolucional (CNN) para clasificación de imágenes.
Implementa una arquitectura simple con 3 bloques Conv2D + MaxPooling.
Utiliza GlobalAveragePooling para reducir dimensionalidad antes de la capa final.
Está diseñado para clasificación multiclase con etiquetas enteras (SparseCategorical).
Se emplea como modelo base para comparar con arquitecturas más complejas.
"""

import tensorflow as tf

from a_Configuracion.config_antiguo import (
    IMAGEN_SHAPE,
    NUM_CLASES,
    TASA_APRENDIZAJE_BASELINE,
)


def construir_modelo_baseline() -> tf.keras.Model:
    """
    Construye una CNN básica para clasificación de imágenes.
    """

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGEN_SHAPE),

            tf.keras.layers.Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                32,
                3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                64,
                3,
                padding="same",
                activation="relu",
            ),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(
                NUM_CLASES,
                activation="softmax",
            ),
        ],
        name="baseline_cnn",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TASA_APRENDIZAJE_BASELINE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return modelo


# ==============================
# Debug rápido
# ==============================
if __name__ == "__main__":
    print("=== DEBUG MODELO BASELINE ===")

    modelo = construir_modelo_baseline()
    modelo.summary()

    print("\n✅ Modelo construido correctamente.")