"""
CNN con Batch Normalization.
Mejora la estabilidad del entrenamiento respecto al baseline.
"""

import tensorflow as tf

from a_Configuracion.config import (
    IMAGEN_SHAPE,
    NUM_CLASES,
    TASA_APRENDIZAJE_BASELINE,
)


def construir_modelo_cnn_bn() -> tf.keras.Model:
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGEN_SHAPE),

            tf.keras.layers.Conv2D(
                32,
                3,
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                64,
                3,
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(
                128,
                3,
                padding="same",
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(NUM_CLASES, activation="softmax"),
        ],
        name="cnn_bn",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TASA_APRENDIZAJE_BASELINE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return modelo


if __name__ == "__main__":
    print("=== DEBUG MODELO CNN_BN ===")
    modelo = construir_modelo_cnn_bn()
    modelo.summary()
    print("\n✅ Modelo construido correctamente.")