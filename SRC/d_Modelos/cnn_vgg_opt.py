"""
CNN profunda tipo VGG optimizada para clasificación histopatológica.
Versión conservadora: mantiene intacto el extractor convolucional y
aligera ligeramente la cabeza final para reducir parámetros sin comprometer
demasiado la capacidad del modelo.
"""

import tensorflow as tf

from a_Configuracion.config import (
    IMAGEN_SHAPE,
    NUM_CLASES,
    TASA_APRENDIZAJE_CNN_VGG,
)

L2 = tf.keras.regularizers.l2(1e-4)


def construir_modelo_cnn_vgg_opt() -> tf.keras.Model:

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
            tf.keras.layers.Dropout(0.20),

            # Cabeza más ligera pero conservadora
            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=L2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.20),

            tf.keras.layers.Dense(NUM_CLASES, activation="softmax"),
        ],
        name="cnn_vgg_opt",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TASA_APRENDIZAJE_CNN_VGG),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return modelo


if __name__ == "__main__":
    print("=== DEBUG MODELO CNN_VGG_OPT ===")
    modelo = construir_modelo_cnn_vgg_opt()
    modelo.summary()
    print("\n✅ Modelo construido correctamente.")