# SRC/tf_utils.py

from pathlib import Path
from typing import List

import tensorflow as tf

from config import (
    SEED,
    MODELOS_DIR,
    MONITOR_METRIC,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
)


def fijar_seed() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def construir_callbacks(nombre_modelo: str) -> List[tf.keras.callbacks.Callback]:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_ckpt: Path = MODELOS_DIR / f"{nombre_modelo}_best.keras"

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta_ckpt),
            monitor=MONITOR_METRIC,
            save_best_only=True,
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
