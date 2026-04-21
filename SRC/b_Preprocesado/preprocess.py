"""
Módulo de preprocesado de imágenes para el pipeline de entrenamiento.

Responsabilidades:
- Cargar imágenes desde disco y adaptarlas al formato esperado por el modelo.
- Aplicar normalización y aumento de datos cuando el modo es entrenamiento.
- Convertir dataframes con rutas y etiquetas en datasets de TensorFlow.

IMPORTANTE:
- Este módulo no corrige por sí solo el data leakage entre train/val/test.
- La fuga real debe corregirse al generar los splits en data.py.
- Aquí solo se eliminan duplicados internos dentro del dataframe recibido.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from a_Configuracion.config import (
    CLAVES_CLASES,
    DESV_IMAGEN,
    MEDIA_IMAGEN,
    PARAMETROS_AUMENTO,
    REESCALADO,
    SEED,
    TAMANO_IMG,
)

AUTOTUNE = tf.data.AUTOTUNE


def construir_tabla_etiquetas() -> tf.lookup.StaticHashTable:
    """
    Construye una tabla hash que transforma etiquetas de texto
    en identificadores enteros para el entrenamiento.
    """
    claves = tf.constant(CLAVES_CLASES, dtype=tf.string)
    valores = tf.constant(list(range(len(CLAVES_CLASES))), dtype=tf.int32)

    return tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(claves, valores),
        default_value=tf.constant(-1, dtype=tf.int32),
    )


def construir_aumentador() -> tf.keras.Sequential:
    """
    Construye la secuencia de aumento de datos definida en configuración.
    """
    rotacion_grados = float(PARAMETROS_AUMENTO.get("rotation_range", 0))
    factor_rotacion = rotacion_grados / 180.0

    desplazamiento_horizontal = float(PARAMETROS_AUMENTO.get("width_shift_range", 0.0))
    desplazamiento_vertical = float(PARAMETROS_AUMENTO.get("height_shift_range", 0.0))

    zoom = PARAMETROS_AUMENTO.get("zoom_range", 0.0)
    if isinstance(zoom, (list, tuple)) and len(zoom) == 2:
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

    if desplazamiento_horizontal > 0 or desplazamiento_vertical > 0:
        capas.append(
            tf.keras.layers.RandomTranslation(
                height_factor=desplazamiento_vertical,
                width_factor=desplazamiento_horizontal,
                seed=SEED,
            )
        )

    return tf.keras.Sequential(capas, name="aumentador")


def _validar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que el dataframe tenga las columnas necesarias y
    elimina filas inválidas o duplicadas internas.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("crear_dataset_tf espera un pd.DataFrame.")

    columnas_requeridas = {"filepath", "label"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias en el DataFrame: {sorted(faltantes)}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.copy()

    df["filepath"] = df["filepath"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[(df["filepath"] != "") & (df["label"] != "")]
    df = df.dropna(subset=["filepath", "label"])

    # Solo elimina duplicados internos exactos del dataframe recibido
    df = df.drop_duplicates(subset=["filepath", "label"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("El DataFrame quedó vacío tras la validación.")

    etiquetas_invalidas = sorted(set(df["label"]) - set(CLAVES_CLASES))
    if etiquetas_invalidas:
        raise ValueError(
            "Se encontraron etiquetas no válidas en el DataFrame: "
            f"{etiquetas_invalidas}. CLAVES_CLASES válidas: {CLAVES_CLASES}"
        )

    return df


def _decodificar_y_redimensionar(ruta: tf.Tensor) -> tf.Tensor:
    """
    Lee una imagen desde disco, la decodifica y la redimensiona.
    """
    bytes_imagen = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(bytes_imagen, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])

    imagen = tf.image.resize(imagen, TAMANO_IMG, method="bilinear")
    imagen = tf.cast(imagen, tf.float32)

    return imagen


def _normalizar(imagen: tf.Tensor) -> tf.Tensor:
    """
    Reescala y normaliza una imagen usando media y desviación estándar.
    """
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
    """
    Preprocesa una muestra individual compuesta por imagen y etiqueta.
    """
    imagen = _decodificar_y_redimensionar(ruta)
    imagen = _normalizar(imagen)

    if entrenamiento:
        if aumentador is None:
            aumentador = construir_aumentador()
        imagen = aumentador(imagen, training=True)

    if tabla_etiquetas is None:
        tabla_etiquetas = construir_tabla_etiquetas()

    id_etiqueta = tabla_etiquetas.lookup(etiqueta_texto)

    tf.debugging.assert_greater_equal(
        id_etiqueta,
        0,
        message="Etiqueta no encontrada en CLAVES_CLASES. Revisa las etiquetas del dataset.",
    )

    return imagen, id_etiqueta


def crear_dataset_tf(
    df: pd.DataFrame,
    batch_size: int,
    entrenamiento: bool = False,
    buffer_barajado: int = 2000,
    usar_cache: bool = False,
) -> tf.data.Dataset:
    """
    Convierte un dataframe con rutas y etiquetas en un dataset de TensorFlow.

    IMPORTANTE:
    - Este método solo limpia duplicados internos del dataframe recibido.
    - No evita data leakage entre splits distintos.
    """
    df = _validar_dataframe(df)

    rutas = df["filepath"].astype(str).values
    etiquetas = df["label"].astype(str).values

    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))

    if entrenamiento:
        dataset = dataset.shuffle(
            buffer_size=min(buffer_barajado, len(df)),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    tabla_etiquetas = construir_tabla_etiquetas()
    aumentador = construir_aumentador() if entrenamiento else None

    dataset = dataset.map(
        lambda ruta, etiqueta: preprocesar_ejemplo(
            ruta,
            etiqueta,
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


if __name__ == "__main__":
    import a_Configuracion.config as cfg

    print("\n=== DEBUG PREPROCESADO ===")
    print("CWD:", Path.cwd())
    print("config importado desde:", Path(cfg.__file__).resolve())
    print("TAMANO_IMG:", cfg.TAMANO_IMG)
    print("CLAVES_CLASES:", cfg.CLAVES_CLASES)

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
    label_guess = Path(img_path).parent.name

    print("Ejemplo imagen:", img_path)
    print("Etiqueta inferida:", label_guess)

    df_prueba = pd.DataFrame([{"filepath": img_path, "label": label_guess}])

    print("\n--- Probando dataset (sin aumento) ---")
    ds = crear_dataset_tf(df_prueba, batch_size=1, entrenamiento=False)

    for x, y in ds.take(1):
        print("Batch imagen shape:", x.shape, "dtype:", x.dtype)
        print("Batch label:", y.numpy(), "dtype:", y.dtype)

    print("\n--- Probando dataset (con aumento) ---")
    ds_aug = crear_dataset_tf(df_prueba, batch_size=1, entrenamiento=True)

    for x_aug, y_aug in ds_aug.take(1):
        print("Batch imagen shape:", x_aug.shape, "dtype:", x_aug.dtype)
        print("Batch label:", y_aug.numpy(), "dtype:", y_aug.dtype)

    print("\n✅ DEBUG OK: preprocesado funciona con este ejemplo.")