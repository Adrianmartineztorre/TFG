"""
Entrenamiento específico del modelo cnn_vgg_opt.
Carga los datasets de entrenamiento y validación desde el pipeline de datos.
Construye el modelo cnn_vgg_opt, ejecuta el entrenamiento y guarda el mejor checkpoint.
Registra el history del entrenamiento y el mejor run en formato JSON.
Si se interrumpe manualmente con Ctrl + C, guarda también el estado parcial.
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from a_Configuracion.config import (
    BATCH_SIZE_CNN_VGG,
    EPOCHS_CNN_VGG,
    MODELOS_DIR,
    MONITOR_METRIC,
)
from a_Configuracion.utils import construir_callbacks, fijar_seed
from c_Data.data import get_tf_datasets
from d_Modelos.cnn_vgg_opt import construir_modelo_cnn_vgg_opt as construir_modelo


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
# Callback para guardar history parcial
# ===============================
class HistorialParcialCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))


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
    }

    resumen["metrics"] = {
        k: float(v[best_idx])
        for k, v in metricas.items()
        if best_idx < len(v)
    }

    resumen["epochs_ejecutados"] = len(valores)

    return resumen


# ===============================
# Guardado común
# ===============================
def _guardar_resultados(history_dict: dict, runs_dir, model_path, nombre_modelo="cnn_vgg_opt"):
    history_path = runs_dir / "history.json"
    history_path.write_text(
        json.dumps(history_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_run = _extraer_mejor_run(history_dict, MONITOR_METRIC)
    best_run["model_name"] = nombre_modelo
    best_run["model_path"] = str(model_path)

    best_run_path = runs_dir / "best_run.json"
    best_run_path.write_text(
        json.dumps(best_run, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return history_path, best_run_path, best_run


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento cnn_vgg_opt")
    parser.add_argument("--epochs", type=int, default=EPOCHS_CNN_VGG)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_CNN_VGG)
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
    print(f"⚙️ Epochs configuradas: {args.epochs}")
    print(f"⚙️ Batch size configurado: {args.batch_size}")

    # ===============================
    # Modelo
    # ===============================
    print("\n🧠 Construyendo modelo cnn_vgg_opt...")
    model = construir_modelo()

    if getattr(model, "optimizer", None) is None:
        raise RuntimeError("❌ El modelo no está compilado.")

    # ===============================
    # Carpetas
    # ===============================
    modelo_dir = MODELOS_DIR / "cnn_vgg_opt"
    modelo_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = MODELOS_DIR.parent / "RUNS" / "cnn_vgg_opt"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # Rutas modelo
    # ===============================
    model_path = modelo_dir / "model.best.keras"
    interrupted_model_path = modelo_dir / "model.interrupted.keras"

    # ===============================
    # Callbacks
    # ===============================
    historial_parcial_cb = HistorialParcialCallback()
    callbacks = construir_callbacks(model_path)
    callbacks.append(ProgresoPorcentaje(total_epochs=args.epochs))
    callbacks.append(historial_parcial_cb)

    # ===============================
    # Entrenamiento
    # ===============================
    print("\n🚀 Iniciando entrenamiento...")

    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        history_dict = {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        }

        history_path, best_run_path, best_run = _guardar_resultados(
            history_dict=history_dict,
            runs_dir=runs_dir,
            model_path=model_path,
            nombre_modelo="cnn_vgg_opt",
        )

        print("\n✅ Entrenamiento finalizado.")
        print(f"🧠 Modelo: {model.name}")
        print(f"📦 Mejor modelo: {model_path}")
        print(f"🧾 History: {history_path}")
        print(f"⭐ Best run: {best_run_path}")
        print(f"📉 Monitor usado: {MONITOR_METRIC}")
        print(f"🏁 Epochs ejecutados: {best_run.get('epochs_ejecutados', 'N/A')}")
        print(f"🥇 Mejor época: {best_run.get('best_epoch', 'N/A')}")
        print(f"📊 Mejor valor: {best_run.get('best_value', 'N/A')}")

    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido manualmente con Ctrl + C")
        print("💾 Guardando resultados parciales...")

        history_dict = {
            k: [float(x) for x in v]
            for k, v in historial_parcial_cb.history.items()
        }

        history_path, best_run_path, best_run = _guardar_resultados(
            history_dict=history_dict,
            runs_dir=runs_dir,
            model_path=model_path,
            nombre_modelo="cnn_vgg_opt",
        )

        model.save(interrupted_model_path)

        print("\n✅ Guardado parcial completado.")
        print(f"🧠 Modelo interrumpido: {interrupted_model_path}")
        print(f"🧾 History parcial: {history_path}")
        print(f"⭐ Best run parcial: {best_run_path}")


if __name__ == "__main__":
    main()