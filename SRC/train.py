# SRC/train.py

import tensorflow as tf

# ===============================
# 🔹 Configuración GPU (MUY IMPORTANTE)
# ===============================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"🚀 GPU(s) detectada(s): {gpus}")
else:
    print("⚠️ No se ha detectado GPU, se usará CPU.")


import argparse
import json
from datetime import datetime
import importlib

from config import (
    MODELOS_DIR,
    BATCH_SIZE,
    EPOCHS_BASELINE,
)

from data import get_tf_datasets
from utils import fijar_seed, construir_callbacks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ProgresoPorcentaje(tf.keras.callbacks.Callback):
    """Muestra el progreso total del entrenamiento en %."""
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        porcentaje = ((epoch + 1) / self.total_epochs) * 100
        print(f"📊 Progreso total: {porcentaje:.2f}%")


def cargar_modelo_dinamico(nombre_modelo: str):
    """
    Carga dinámicamente un modelo desde SRC/modelos/<nombre_modelo>.py
    y devuelve la función construir_modelo()
    """
    try:
        modulo = importlib.import_module(f"modelos.{nombre_modelo}")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"No existe el archivo modelos/{nombre_modelo}.py"
        )

    if not hasattr(modulo, "construir_modelo"):
        raise AttributeError(
            f"El módulo modelos/{nombre_modelo}.py no tiene la función construir_modelo()"
        )

    return modulo.construir_modelo


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar un modelo CNN y guardar SOLO el mejor modelo."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_BASELINE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    # Seed y determinismo
    fijar_seed()

    # ===============================
    # 🔹 Elegir modelo por input
    # ===============================
    print("\n📦 Modelos disponibles (SRC/modelos):")
    print("  - cnn_simple")
    print("  - cnn_bn")
    print("  - cnn_bn_dropout")
    print("  - cnn_sepconv")

    nombre_modelo = input("\n👉 Escribe el nombre del modelo a entrenar (sin .py): ").strip()

    construir_modelo = cargar_modelo_dinamico(nombre_modelo)

    # ===============================
    # 1) Datos
    # ===============================
    train_ds, val_ds, _test_ds, *_ = get_tf_datasets(
        batch_size=args.batch_size,
        cache=args.cache,
    )

    # ===============================
    # 2) Modelo
    # ===============================
    model = construir_modelo()

    if getattr(model, "optimizer", None) is None:
        raise RuntimeError(
            "El modelo no está compilado. "
            "Debe compilarse dentro de construir_modelo()."
        )

    # ===============================
    # 3) Carpeta del experimento
    # ===============================
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    exp_name = args.exp_name or f"{model.name}_{_timestamp()}"
    out_dir = MODELOS_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # 4) Ruta del modelo (solo el mejor)
    # ===============================
    model_path = out_dir / "model.keras"

    # ===============================
    # 5) Callbacks
    # ===============================
    callbacks = construir_callbacks(model_path)
    callbacks.append(ProgresoPorcentaje(total_epochs=args.epochs))

    # ===============================
    # 6) Entrenamiento
    # ===============================
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ===============================
    # 7) Guardar history en RUNS
    # ===============================
    RUNS_DIR = MODELOS_DIR.parent / "RUNS"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    history_path = RUNS_DIR / f"history_{exp_name}.json"
    history_path.write_text(
        json.dumps(history.history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n✅ Entrenamiento finalizado.")
    print(f"🧠 Modelo: {model.name}")
    print(f"📦 Modelo guardado en: {model_path}")
    print(f"🧾 History guardado en: {history_path}")


if __name__ == "__main__":
    main()
