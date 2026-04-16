"""
EfficientNetB0 para clasificación multiclase de imágenes.
Utiliza un backbone preentrenado congelado y una cabeza de
clasificación reforzada para obtener un rendimiento alto y estable.
El preprocesado de imágenes se realiza previamente en el pipeline de datos.
"""

import tensorflow as tf

from a_Configuracion.config import (
    IMAGEN_SHAPE,
    NUM_CLASES,
    TASA_APRENDIZAJE_TRANSFER,
)


def construir_modelo_efficientnet_b0() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=IMAGEN_SHAPE)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    # Se congela el backbone para aprovechar las características preentrenadas
    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Cabeza de clasificación mejorada
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(inputs, outputs, name="efficientnet_b0")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TASA_APRENDIZAJE_TRANSFER
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return modelo


if __name__ == "__main__":
    print("=== DEBUG MODELO EFFICIENTNET_B0 ===")
    modelo = construir_modelo_efficientnet_b0()
    modelo.summary()
    print("\n✅ Modelo construido correctamente.")