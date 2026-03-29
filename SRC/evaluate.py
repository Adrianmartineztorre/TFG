# SRC/evaluar.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # Quita aviso oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # Oculta logs INFO y WARNING

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    SEED,
    MODELOS_DIR,
    FIGURAS_DIR,
    BATCH_SIZE,
    CLAVES_CLASES,
    NUM_CLASES,
)

from data import get_splits
from preprocess import crear_dataset_tf  # <-- si tu módulo se llama preprocess, cambia esto a: from preprocess import crear_dataset_tf


def _asegurar_carpetas():
    # Sin subcarpetas
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    metricas_dir = FIGURAS_DIR.parent / "METRICAS"
    metricas_dir.mkdir(parents=True, exist_ok=True)
    return metricas_dir


def _recoger_predicciones(modelo, dataset):
    y_true_list, y_proba_list = [], []

    for x_batch, y_batch in dataset:
        proba = modelo.predict(x_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_proba_list.append(proba)

    y_true = np.concatenate(y_true_list).astype(int)
    y_proba = np.concatenate(y_proba_list)
    y_pred = np.argmax(y_proba, axis=1).astype(int)

    return y_true, y_pred


def guardar_matriz_confusion(y_true, y_pred, nombres_clases, ruta_png: Path, cmap="viridis"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        linewidths=0.5,
        linecolor="black",
    )
    plt.xlabel("Etiquetas predichas")
    plt.ylabel("Etiquetas reales")
    plt.title("Matriz de confusión (Multiclase)")
    plt.yticks(rotation=0)
    plt.tight_layout()

    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_png, dpi=200)
    plt.close()

    return cm


def _cargar_history_desde_runs(runs_dir: Path, run_json: str):
    """
    Carga el history desde OUTPUT/RUNS/<run_json>.
    run_json debe ser un nombre tipo "history_xxx.json".
    Devuelve dict o None.
    """
    path = runs_dir / run_json
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Si alguien lo guardó como {"history": {...}}, lo soportamos
        if "history" in data and isinstance(data["history"], dict):
            return data["history"]
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extraer_info_training(history: dict):
    """
    Devuelve:
      - epochs_ejecutados
      - best_epoch_val_loss (1-indexed) y best_val_loss si existe val_loss
      - best_epoch_val_acc  (1-indexed) y best_val_acc si existe val_accuracy/val_acc
    """
    if not history or not isinstance(history, dict):
        return {
            "epochs_ejecutados": "N/A",
            "best_epoch_val_loss": "N/A",
            "best_val_loss": "N/A",
            "best_epoch_val_acc": "N/A",
            "best_val_acc": "N/A",
        }

    loss = history.get("loss", [])
    epochs_ejecutados = len(loss) if isinstance(loss, list) else "N/A"

    # val_loss
    val_loss = history.get("val_loss", None)
    if isinstance(val_loss, list) and len(val_loss) > 0:
        best_idx = int(np.argmin(val_loss))
        best_epoch_val_loss = best_idx + 1
        best_val_loss = float(val_loss[best_idx])
    else:
        best_epoch_val_loss = "N/A"
        best_val_loss = "N/A"

    # val_accuracy / val_acc
    val_acc = history.get("val_accuracy", history.get("val_acc", None))
    if isinstance(val_acc, list) and len(val_acc) > 0:
        best_idx = int(np.argmax(val_acc))
        best_epoch_val_acc = best_idx + 1
        best_val_acc = float(val_acc[best_idx])
    else:
        best_epoch_val_acc = "N/A"
        best_val_acc = "N/A"

    return {
        "epochs_ejecutados": epochs_ejecutados,
        "best_epoch_val_loss": best_epoch_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch_val_acc": best_epoch_val_acc,
        "best_val_acc": best_val_acc,
    }


def _tabla_metricas_imagen(
    reporte_dict: dict,
    accuracy: float,
    info_extra: dict,
    ruta_png: Path,
):
    """
    Genera una tabla bonita en imagen con:
      - filas por clase (precision/recall/f1/support)
      - macro avg
      - weighted avg
      - accuracy
    + bloque info extra (incluye epochs/best epoch si hay history).
    """
    filas = []
    nombres_clases = info_extra["nombres_clases"]

    def fmt3(x):
        return f"{x:.3f}"

    for c in nombres_clases:
        d = reporte_dict.get(c, {})
        filas.append([
            c,
            fmt3(float(d.get("precision", 0.0))),
            fmt3(float(d.get("recall", 0.0))),
            fmt3(float(d.get("f1-score", 0.0))),
            str(int(d.get("support", 0))),
        ])

    for key, label in [("macro avg", "macro avg"), ("weighted avg", "weighted avg")]:
        if key in reporte_dict:
            d = reporte_dict[key]
            filas.append([
                label,
                fmt3(float(d.get("precision", 0.0))),
                fmt3(float(d.get("recall", 0.0))),
                fmt3(float(d.get("f1-score", 0.0))),
                str(int(d.get("support", 0))),
            ])

    filas.append([
        "accuracy",
        "-",
        "-",
        fmt3(accuracy),
        str(int(info_extra["num_muestras"])),
    ])

    columnas = ["Clase", "Precision", "Recall", "F1", "Support"]

    plt.figure(figsize=(12, 7))
    plt.axis("off")

    titulo = f"Tabla de métricas ({info_extra['split']}) — {info_extra['modelo_base']}"
    plt.text(0.01, 0.97, titulo, fontsize=16, weight="bold", va="top")

    # Info extra (incluye training)
    extra_txt = (
        f"Ruta modelo: {info_extra['ruta_modelo']}\n"
        f"Batch size: {info_extra['batch_size']}\n"
        f"Nº muestras: {info_extra['num_muestras']}\n"
        f"Distribución (y_true): {info_extra['distribucion_true']}\n"
        f"Distribución (y_pred): {info_extra['distribucion_pred']}\n"
        f"Parámetros modelo: {info_extra['num_params']}\n"
        f"Epochs ejecutados: {info_extra.get('epochs_ejecutados', 'N/A')}\n"
        f"Mejor epoch (val_loss): {info_extra.get('best_epoch_val_loss', 'N/A')} | val_loss: {info_extra.get('best_val_loss', 'N/A')}\n"
        f"Mejor epoch (val_acc):  {info_extra.get('best_epoch_val_acc', 'N/A')} | val_acc:  {info_extra.get('best_val_acc', 'N/A')}\n"
    )
    plt.text(0.01, 0.86, extra_txt, fontsize=10, family="monospace", va="top")

    table = plt.table(
        cellText=filas,
        colLabels=columnas,
        loc="lower left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.05, 0.98, 0.55],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluación multiclase de modelos entrenados")

    # Solo pedimos carpeta (el nombre de la carpeta del experimento)
    parser.add_argument("--carpeta", type=str, default=None, help="Carpeta dentro de MODELOS_DIR (experimento)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap para la matriz de confusión")

    args = parser.parse_args()

    METRICAS_DIR = _asegurar_carpetas()
    tf.random.set_seed(SEED)

    # ✅ Solo preguntar por el nombre de la carpeta (experimento)
    carpeta = args.carpeta or input("Nombre del modelo, sin extensión: ").strip()

    # Modelo y run se derivan automáticamente:
    # - modelo: <carpeta>.keras
    # - run: history_<carpeta>.json
    nombre_modelo_archivo = f"{carpeta}.keras"
    run_json = f"history_{carpeta}.json"

    modelo_base = Path(nombre_modelo_archivo).stem

    model_dir = MODELOS_DIR / carpeta
    model_path = model_dir / nombre_modelo_archivo
    if not model_path.exists():
        # fallback por si tu train guarda "model.keras" (tu caso actual)
        fallback = model_dir / "model.keras"
        if fallback.exists():
            model_path = fallback
            nombre_modelo_archivo = "model.keras"
            modelo_base = carpeta  # para que las salidas usen el nombre del experimento
        else:
            raise FileNotFoundError(
                f"No existe el modelo en: {model_path}\n"
                f"Ni tampoco en fallback: {fallback}"
            )

    # Cargar modelo
    modelo = tf.keras.models.load_model(model_path)

    # Dataset
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    train_df, val_df, test_df = get_splits(create_if_missing=True)

    if args.split == "train":
        df_eval = train_df
    elif args.split == "val":
        df_eval = val_df
    else:
        df_eval = test_df

    ds_eval = crear_dataset_tf(
        df_eval,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=args.cache,
    )

    # Predicciones
    y_true, y_pred = _recoger_predicciones(modelo, ds_eval)

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    unique_true, counts_true = np.unique(y_true, return_counts=True)

    distrib_true = {int(k): int(v) for k, v in zip(unique_true, counts_true)}
    distrib_pred = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}

    # Matriz confusión en FIGURAS_DIR (sin subcarpetas)
    ruta_cm = FIGURAS_DIR / f"matriz_confusion_{modelo_base}.png"
    guardar_matriz_confusion(y_true, y_pred, CLAVES_CLASES, ruta_cm, cmap=args.cmap)

    # Métricas base
    reporte_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLAVES_CLASES,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(accuracy_score(y_true, y_pred))

    # ✅ Cargar history desde OUTPUT/RUNS (sin subcarpetas)
    RUNS_DIR = FIGURAS_DIR.parent / "RUNS"
    history = _cargar_history_desde_runs(RUNS_DIR, run_json)
    info_train = _extraer_info_training(history)

    # Extras
    num_params = int(modelo.count_params())

    info_extra = {
        "modelo_base": modelo_base,
        "ruta_modelo": str(model_path),
        "split": args.split,
        "batch_size": batch_size,
        "num_muestras": int(len(y_true)),
        "distribucion_true": distrib_true,
        "distribucion_pred": distrib_pred,
        "num_params": num_params,
        "nombres_clases": CLAVES_CLASES,
        **info_train,  # epochs_ejecutados + best epochs
    }

    # Tabla de métricas en imagen
    ruta_tabla = METRICAS_DIR / f"tabla_metricas_{modelo_base}.png"
    _tabla_metricas_imagen(
        reporte_dict=reporte_dict,
        accuracy=accuracy,
        info_extra=info_extra,
        ruta_png=ruta_tabla,
    )

    # Prints
    print("\n✅ Evaluación completada")
    print(f"📁 Carpeta: {carpeta}")
    print(f"🧠 Modelo cargado: {nombre_modelo_archivo}")
    print(f"📌 Ruta modelo: {model_path}")
    print(f"🧾 Run (history): {RUNS_DIR / run_json} {'(OK)' if history else '(NO ENCONTRADO)'}")
    print(f"🧪 Split: {args.split}")
    print(f"🎯 Accuracy: {accuracy:.3f}")
    print(f"🖼️ Matriz de confusión: {ruta_cm}")
    print(f"🧾 Tabla métricas (PNG): {ruta_tabla}")

    if len(unique_pred) == 1:
        print("\n⚠️ OJO: el modelo está prediciendo UNA sola clase.")
        print(f"   Clase predicha: {unique_pred[0]}  | Conteo: {counts_pred[0]}")


if __name__ == "__main__":
    main()
