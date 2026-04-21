"""
CNN profunda tipo VGG para clasificación histopatológica.
Arquitectura optimizada para reducir consumo de memoria
manteniendo una extracción jerárquica de características.
"""

import tensorflow as tf

from a_Configuracion.config_antiguo import (
    IMAGEN_SHAPE,
    NUM_CLASES,
    TASA_APRENDIZAJE_CNN_VGG,
)

L2 = tf.keras.regularizers.l2(1e-4)


def construir_modelo_cnn_vgg() -> tf.keras.Model:

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGEN_SHAPE),

            # Bloque 1
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),

            # Bloque 2
            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),

            # Bloque 3
            tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.15),

            # Bloque 4
            tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.20),

            # Bloque 5
            tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),

            # Cabeza
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(192, use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.35),

            tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(NUM_CLASES, activation="softmax"),
        ],
        name="cnn_vgg",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(TASA_APRENDIZAJE_CNN_VGG),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    return modelo


if __name__ == "__main__":
    print("=== DEBUG MODELO CNN_VGG ===")
    modelo = construir_modelo_cnn_vgg()
    modelo.summary()
    print("\n✅ Modelo construido correctamente.")