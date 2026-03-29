# SRC/preprocesado.py

import tensorflow as tf
from config import (
    SEED,
    TAMANO_IMG,
    REESCALADO,
    MEDIA_IMAGEN,
    DESV_IMAGEN,
    CLAVES_CLASES,
    PARAMETROS_AUMENTO,
)

AUTOTUNE = tf.data.AUTOTUNE


def construir_tabla_etiquetas() -> tf.lookup.StaticHashTable:
    """
    Tabla: etiqueta (texto) -> id (int32)
    """
    claves = tf.constant(CLAVES_CLASES, dtype=tf.string)
    valores = tf.constant(list(range(len(CLAVES_CLASES))), dtype=tf.int32)

    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(claves, valores),
        default_value=tf.constant(-1, dtype=tf.int32),
    )


def construir_aumentador() -> tf.keras.Sequential:
    rotacion_grados = float(PARAMETROS_AUMENTO.get("rotation_range", 0))
    factor_rotacion = rotacion_grados / 180.0

    desplazamiento_horizontal = float(PARAMETROS_AUMENTO.get("width_shift_range", 0.0))
    desplazamiento_vertical = float(PARAMETROS_AUMENTO.get("height_shift_range", 0.0))

    zoom = PARAMETROS_AUMENTO.get("zoom_range", 0.0)
    if isinstance(zoom, (list, tuple)) and len(zoom) == 2:
        # si te pasan (0.9, 1.1) lo convertimos a +/-0.1
        factor_zoom = max(abs(1 - float(zoom[0])), abs(1 - float(zoom[1])))
    else:
        factor_zoom = float(zoom)

    volteo_horizontal = bool(PARAMETROS_AUMENTO.get("horizontal_flip", True))

    capas = []
    if volteo_horizontal:
        capas.append(tf.keras.layers.RandomFlip("horizontal", seed=SEED))
    if factor_rotacion > 0:
        capas.append(tf.keras.layers.RandomRotation(factor_rotacion, seed=SEED))
    if factor_zoom > 0:
        capas.append(
            tf.keras.layers.RandomZoom(
                height_factor=(-factor_zoom, factor_zoom),
                width_factor=(-factor_zoom, factor_zoom),
                seed=SEED,
            )
        )
    if (desplazamiento_horizontal > 0) or (desplazamiento_vertical > 0):
        capas.append(
            tf.keras.layers.RandomTranslation(
                height_factor=desplazamiento_vertical,
                width_factor=desplazamiento_horizontal,
                seed=SEED,
            )
        )

    return tf.keras.Sequential(capas, name="aumentador")


def _decodificar_y_redimensionar(ruta: tf.Tensor) -> tf.Tensor:
    bytes_imagen = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(bytes_imagen, channels=3, expand_animations=False)
    # Evita shapes desconocidas en pipelines tf.data
    imagen.set_shape([None, None, 3])

    imagen = tf.image.resize(imagen, TAMANO_IMG, method="bilinear")
    imagen = tf.cast(imagen, tf.float32)
    return imagen


def _normalizar(imagen: tf.Tensor) -> tf.Tensor:
    imagen = imagen * REESCALADO

    media = tf.constant(MEDIA_IMAGEN, dtype=tf.float32)
    desviacion = tf.constant(DESV_IMAGEN, dtype=tf.float32)

    imagen = (imagen - media) / desviacion
    return imagen


def preprocesar_ejemplo(
    ruta: tf.Tensor,
    etiqueta_texto: tf.Tensor,
    entrenamiento: bool = False,
    tabla_etiquetas: tf.lookup.StaticHashTable | None = None,
    aumentador: tf.keras.Sequential | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    imagen = _decodificar_y_redimensionar(ruta)
    imagen = _normalizar(imagen)

    if entrenamiento:
        if aumentador is None:
            aumentador = construir_aumentador()
        imagen = aumentador(imagen, training=True)

    if tabla_etiquetas is None:
        tabla_etiquetas = construir_tabla_etiquetas()

    id_etiqueta = tabla_etiquetas.lookup(etiqueta_texto)  # int32

    tf.debugging.assert_greater_equal(
        id_etiqueta,
        0,
        message="Etiqueta no encontrada en CLAVES_CLASES. Revisa las etiquetas del dataset.",
    )

    return imagen, id_etiqueta


def crear_dataset_tf(
    df,
    batch_size: int,
    entrenamiento: bool = False,
    buffer_barajado: int = 2000,
    usar_cache: bool = False,
):
    rutas = df["filepath"].astype(str).values
    etiquetas = df["label"].astype(str).values

    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))

    if entrenamiento:
        dataset = dataset.shuffle(
            buffer_size=buffer_barajado,
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    tabla_etiquetas = construir_tabla_etiquetas()
    aumentador = construir_aumentador() if entrenamiento else None

    dataset = dataset.map(
        lambda r, e: preprocesar_ejemplo(
            r,
            e,
            entrenamiento=entrenamiento,
            tabla_etiquetas=tabla_etiquetas,
            aumentador=aumentador,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    if usar_cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


# ==============================
# Debug rápido (ejecutar este archivo directamente)
# ==============================
if __name__ == "__main__":
    import os
    import pandas as pd
    from pathlib import Path
    import config as cfg

    print("\n=== DEBUG PREPROCESADO ===")
    print("CWD:", Path.cwd())
    print("config importado desde:", Path(cfg.__file__).resolve())
    print("TAMANO_IMG:", cfg.TAMANO_IMG)
    print("CLAVES_CLASES:", cfg.CLAVES_CLASES)

    # 1) Buscar una imagen dentro del dataset RAW (tú ajustas si tu estructura difiere)
    # Intentamos en colon y lung
    posibles = []
    for base in [cfg.RUTA_COLON, cfg.RUTA_LUNG]:
        if base.exists():
            posibles += list(base.rglob("*.jpeg"))
            posibles += list(base.rglob("*.jpg"))
            posibles += list(base.rglob("*.png"))

    print("Imágenes encontradas:", len(posibles))
    if not posibles:
        raise FileNotFoundError(
            f"No encuentro imágenes en {cfg.RUTA_COLON} ni {cfg.RUTA_LUNG}. "
            "Revisa rutas en config.py o tu carpeta DATA/RAW."
        )

    img_path = str(posibles[0])
    # Inferir label desde carpeta (si tus carpetas se llaman como CLAVES_CLASES)
    label_guess = Path(img_path).parent.name

    print("Ejemplo imagen:", img_path)
    print("Etiqueta inferida:", label_guess)

    # 2) Crear DF de prueba
    df = pd.DataFrame([{"filepath": img_path, "label": label_guess}])

    # 3) Probar pipeline sin entrenamiento
    print("\n--- Probando dataset (sin aumento) ---")
    ds = crear_dataset_tf(df, batch_size=1, entrenamiento=False)

    for x, y in ds.take(1):
        print("Batch imagen shape:", x.shape, "dtype:", x.dtype)
        print("Batch label:", y.numpy(), "dtype:", y.dtype)

    # 4) Probar pipeline con entrenamiento (aumentos)
    print("\n--- Probando dataset (con aumento) ---")
    ds2 = crear_dataset_tf(df, batch_size=1, entrenamiento=True)

    for x2, y2 in ds2.take(1):
        print("Batch imagen shape:", x2.shape, "dtype:", x2.dtype)
        print("Batch label:", y2.numpy(), "dtype:", y2.dtype)

    print("\n✅ DEBUG OK: preprocesado funciona con este ejemplo.")
