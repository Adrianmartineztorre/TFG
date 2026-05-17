"""
Utilidades de soporte para el entrenamiento del modelo.
Gestiona la reproducibilidad mediante seed global.
Define callbacks estándar para control del entrenamiento.
Centraliza lógica reutilizable relacionada con TensorFlow.
Se utiliza principalmente desde el módulo de entrenamiento.
"""

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from a_Configuracion.config import (
    SEED,
    MONITOR_METRIC,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
)


def fijar_seed() -> None:
    """
    Fija la semilla global para garantizar reproducibilidad.
    """
    tf.keras.utils.set_random_seed(SEED)


def construir_callbacks(
    ruta_ckpt: Path,
) -> List[tf.keras.callbacks.Callback]:
    """
    Construye los callbacks estándar:
    - ModelCheckpoint: guarda el mejor checkpoint
    - EarlyStopping: detiene entrenamiento si no mejora
    - ReduceLROnPlateau: reduce LR automáticamente

    Para modelos EfficientNet, guarda solo los pesos para evitar
    problemas de serialización del modelo completo.
    """

    ruta_ckpt.parent.mkdir(parents=True, exist_ok=True)

    nombre_modelo = str(ruta_ckpt).lower()
    es_efficientnet = "efficientnet" in nombre_modelo

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_ckpt),
            monitor=MONITOR_METRIC,
            save_best_only=True,
            save_weights_only=es_efficientnet,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=MONITOR_METRIC,
            patience=REDUCE_LR_PATIENCE,
            factor=REDUCE_LR_FACTOR,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    return callbacks


def convertir_a_json_serializable(obj):
    """
    Convierte tensores, arrays NumPy y tipos NumPy
    a tipos nativos de Python para poder guardarlos en JSON.
    """

    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, dict):
        return {k: convertir_a_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [convertir_a_json_serializable(v) for v in obj]

    return obj