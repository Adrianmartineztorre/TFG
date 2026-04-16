"""
Entrenamiento específico del modelo efficientnet_b0.
Utiliza EfficientNetB0 con backbone congelado y una cabeza de
clasificación entrenable. Aplica el preprocesado específico del
modelo en el pipeline de entrenamiento y validación.
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from a_Configuracion.config import (
    BATCH_SIZE,
    EPOCHS_TRANSFER,
    MODELOS_DIR,
    MONITOR_METRIC,
)
from b_Preprocesado.utils import (
    construir_callbacks,
    fijar_seed,
    convertir_a_json_serializable,
)
from c_Data.data import get_tf_datasets
from d_Modelos.efficientnet_b0 import (
    construir_modelo_efficientnet_b0 as construir_modelo,
)


# ===============================
# Configuración GPU
# ===============================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"🚀 GPU(s) detectada(s): {gpus}")
else:
    print("⚠️ No se ha detectado GPU, se usará CPU.")


# ===============================
# Callback progreso
# ===============================
class ProgresoPorcentaje(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        porcentaje = ((epoch + 1) / self.total_epochs) * 100
        print(f"📊 Progreso total: {porcentaje:.2f}%")


# ===============================
# Extraer mejor run
# ===============================
def _extraer_mejor_run(history_dict: dict, monitor_metric: str) -> dict:
    if not history_dict:
        return {}

    metricas = {
        k: v for k, v in history_dict.items()
        if isinstance(v, list) and len(v) > 0
    }

    if monitor_metric not in metricas:
        return {
            "error": f"No existe {monitor_metric} en history",
            "available_metrics": list(metricas.keys()),
        }

    valores = metricas[monitor_metric]

    if "loss" in monitor_metric.lower():
        best_idx = int(np.argmin(valores))
        mode = "min"
    else:
        best_idx = int(np.argmax(valores))
        mode = "max"

    best_epoch = best_idx + 1

    resumen = {
        "monitor_metric": monitor_metric,
        "mode": mode,
        "best_epoch": best_epoch,
        "best_value": float(valores[best_idx]),
        "epochs_ejecutados": len(valores),
        "metrics": {
            k: float(v[best_idx])
            for k, v in metricas.items()
            if best_idx < len(v)
        },
    }

    return resumen


# ===============================
# Preprocesado EfficientNet
# ===============================
def aplicar_preprocess_efficientnet(dataset: tf.data.Dataset) -> tf.data.Dataset:
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    return dataset.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento efficientnet_b0"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_TRANSFER)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    fijar_seed()

    # ===============================
    # Datos
    # ===============================
    print("\n📦 Cargando datasets...")
    train_ds, val_ds, _test_ds, train_df, val_df, _test_df = get_tf_datasets(
        batch_size=args.batch_size,
        cache=args.cache,
    )

    print(f"✅ Train samples: {len(train_df)}")
    print(f"✅ Val samples:   {len(val_df)}")

    # ===============================
    # Preprocesado específico EfficientNet
    # ===============================
    train_ds = aplicar_preprocess_efficientnet(train_ds)
    val_ds = aplicar_preprocess_efficientnet(val_ds)

    # ===============================
    # Modelo
    # ===============================
    print("\n🧠 Construyendo modelo efficientnet_b0...")
    model = construir_modelo()

    if getattr(model, "optimizer", None) is None:
        raise RuntimeError("❌ El modelo no está compilado.")

    # ===============================
    # Carpetas
    # ===============================
    modelo_dir = MODELOS_DIR / "efficientnet_b0"
    modelo_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = MODELOS_DIR.parent / "RUNS" / "efficientnet_b0"
    runs_dir.mkdir(parents=True, exist_ok=True)

    model_path = modelo_dir / "model.best.weights.h5"

    # ===============================
    # Callbacks
    # ===============================
    callbacks = construir_callbacks(model_path)
    callbacks.append(ProgresoPorcentaje(total_epochs=args.epochs))

    # ===============================
    # Entrenamiento
    # ===============================
    print("\n🚀 Iniciando entrenamiento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    history_dict = history.history

    # ===============================
    # Guardar history
    # ===============================
    history_path = runs_dir / "history.json"
    history_path.write_text(
        json.dumps(
            convertir_a_json_serializable(history_dict),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ===============================
    # Guardar best run
    # ===============================
    best_run = _extraer_mejor_run(history_dict, MONITOR_METRIC)
    best_run["model_name"] = "efficientnet_b0"
    best_run["model_path"] = str(model_path)

    best_run_path = runs_dir / "best_run.json"
    best_run_path.write_text(
        json.dumps(
            convertir_a_json_serializable(best_run),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ===============================
    # Resumen final
    # ===============================
    print("\n✅ Entrenamiento finalizado.")
    print(f"🧠 Modelo: {model.name}")
    print(f"📦 Mejor modelo: {model_path}")

    if "best_epoch" in best_run:
        print(
            f"🏆 Mejor epoch: {best_run['best_epoch']} "
            f"({best_run['monitor_metric']} = {best_run['best_value']:.5f})"
        )


if __name__ == "__main__":
    main()