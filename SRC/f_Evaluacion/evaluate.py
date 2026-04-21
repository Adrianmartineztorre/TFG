# SRC/e_Evaluacion/evaluar.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

from a_Configuracion.config import (
    SEED,
    MODELOS_DIR,
    FIGURAS_DIR,
    METRICAS_DIR,
    BATCH_SIZE,
    CLAVES_CLASES,
)

from c_Data.data import get_splits
from b_Preprocesado.preprocess import crear_dataset_tf


# ===============================
# Modelos permitidos
# ===============================
MODELOS_DISPONIBLES = [
    "baseline_cnn",
    "cnn_bn",
    "cnn_bn_dropout",
    "efficientnet_b0",
    "cnn_vgg",
    "cnn_vgg_opt",
]

NOMBRE_MODELO_ARCHIVO = "model.best.keras"


def _asegurar_carpetas_modelo(nombre_modelo: str):
    """
    Crea carpetas separadas por modelo para guardar resultados:
    - FIGURAS/<nombre_modelo>/
    - METRICAS/<nombre_modelo>/
    """
    figuras_modelo_dir = FIGURAS_DIR / nombre_modelo
    metricas_modelo_dir = METRICAS_DIR / nombre_modelo

    figuras_modelo_dir.mkdir(parents=True, exist_ok=True)
    metricas_modelo_dir.mkdir(parents=True, exist_ok=True)

    return figuras_modelo_dir, metricas_modelo_dir


def _elegir_modelo(modelo_argumento: str | None = None) -> str:
    """
    Obliga a escoger uno de los modelos definidos en MODELOS_DISPONIBLES.
    """
    if modelo_argumento is not None:
        modelo_argumento = modelo_argumento.strip()
        if modelo_argumento not in MODELOS_DISPONIBLES:
            disponibles = ", ".join(MODELOS_DISPONIBLES)
            raise ValueError(
                f"Modelo no válido: {modelo_argumento}\n"
                f"Modelos disponibles: {disponibles}"
            )
        return modelo_argumento

    print("\nModelos disponibles:")
    for i, nombre in enumerate(MODELOS_DISPONIBLES, start=1):
        print(f"  {i}. {nombre}")

    while True:
        eleccion = input("\nEscoge el número del modelo a evaluar: ").strip()

        if not eleccion.isdigit():
            print("❌ Introduce un número válido.")
            continue

        idx = int(eleccion)
        if 1 <= idx <= len(MODELOS_DISPONIBLES):
            return MODELOS_DISPONIBLES[idx - 1]

        print("❌ Opción fuera de rango.")


def _resolver_ruta_modelo(nombre_modelo: str) -> Path:
    """
    Busca el modelo en:
    MODELOS_DIR/<nombre_modelo>/model.best.keras
    """
    model_path = MODELOS_DIR / nombre_modelo / NOMBRE_MODELO_ARCHIVO

    if model_path.exists():
        return model_path

    raise FileNotFoundError(
        f"No se encontró el modelo en:\n{model_path}"
    )


def _cargar_modelo(nombre_modelo: str, model_path: Path):
    """
    Carga el modelo normalmente.
    Si falla para EfficientNetB0, reconstruye la arquitectura y carga pesos
    desde el mismo archivo model.best.keras.
    """
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        if nombre_modelo == "efficientnet_b0":
            print("⚠️ No se pudo cargar EfficientNetB0 como modelo completo.")
            print(f"   Motivo: {e}")
            print("🔄 Reconstruyendo arquitectura y cargando pesos...")

            from d_Modelos.efficientnet_b0 import (
                construir_modelo_efficientnet_b0 as construir_modelo
            )

            modelo = construir_modelo()
            modelo.load_weights(model_path)
            print("✅ EfficientNetB0 cargado como arquitectura + pesos.")
            return modelo

        raise


def _buscar_history_para_modelo(nombre_modelo: str) -> Path | None:
    """
    Busca un history relacionado con el modelo dentro de OUTPUT/RUNS.
    """
    runs_dir = FIGURAS_DIR.parent / "RUNS"

    if not runs_dir.exists():
        return None

    candidatos = []

    subdir_modelo = runs_dir / nombre_modelo
    if subdir_modelo.exists() and subdir_modelo.is_dir():
        candidatos.extend(sorted(subdir_modelo.glob("history*.json")))

    candidatos.extend(sorted(runs_dir.glob(f"history_{nombre_modelo}*.json")))

    for p in runs_dir.glob("**/history*.json"):
        if nombre_modelo.lower() in str(p).lower():
            candidatos.append(p)

    unicos = []
    vistos = set()
    for c in candidatos:
        if c.exists() and c not in vistos:
            unicos.append(c)
            vistos.add(c)

    if not unicos:
        return None

    return unicos[-1]


def _buscar_best_run_para_modelo(nombre_modelo: str) -> Path | None:
    """
    Busca el best_run.json dentro de OUTPUT/RUNS/<modelo>/.
    """
    runs_dir_modelo = FIGURAS_DIR.parent / "RUNS" / nombre_modelo

    if not runs_dir_modelo.exists():
        return None

    best_run_path = runs_dir_modelo / "best_run.json"

    if best_run_path.exists():
        return best_run_path

    return None


def _cargar_history(path_history: Path | None):
    if path_history is None or not path_history.exists():
        return None

    try:
        data = json.loads(path_history.read_text(encoding="utf-8"))
        if "history" in data and isinstance(data["history"], dict):
            return data["history"]
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cargar_best_run(path_best_run: Path | None):
    if path_best_run is None or not path_best_run.exists():
        return None

    try:
        data = json.loads(path_best_run.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extraer_info_training(history: dict | None):
    if not history or not isinstance(history, dict):
        return {
            "epochs_ejecutados": "N/A",
            "best_epoch_val_loss": "N/A",
            "best_val_loss": "N/A",
            "best_epoch_val_acc": "N/A",
            "best_val_acc": "N/A",
            "best_epoch": "N/A",
            "best_value": "N/A",
            "monitor_metric": "N/A",
        }

    loss = history.get("loss", [])
    epochs_ejecutados = len(loss) if isinstance(loss, list) else "N/A"

    val_loss = history.get("val_loss", None)
    if isinstance(val_loss, list) and len(val_loss) > 0:
        best_idx_loss = int(np.argmin(val_loss))
        best_epoch_val_loss = best_idx_loss + 1
        best_val_loss = float(val_loss[best_idx_loss])
    else:
        best_epoch_val_loss = "N/A"
        best_val_loss = "N/A"

    val_acc = history.get("val_accuracy", history.get("val_acc", None))
    if isinstance(val_acc, list) and len(val_acc) > 0:
        best_idx_acc = int(np.argmax(val_acc))
        best_epoch_val_acc = best_idx_acc + 1
        best_val_acc = float(val_acc[best_idx_acc])
    else:
        best_epoch_val_acc = "N/A"
        best_val_acc = "N/A"

    return {
        "epochs_ejecutados": epochs_ejecutados,
        "best_epoch_val_loss": best_epoch_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch_val_acc": best_epoch_val_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": "N/A",
        "best_value": "N/A",
        "monitor_metric": "N/A",
    }


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


def guardar_matriz_confusion(
    y_true,
    y_pred,
    nombres_clases,
    ruta_png: Path,
    cmap="viridis",
):
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


def _tabla_metricas_imagen(
    reporte_dict: dict,
    accuracy: float,
    info_extra: dict,
    ruta_png: Path,
):
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

    plt.figure(figsize=(14, 9))
    plt.axis("off")

    titulo = f"Tabla de métricas ({info_extra['split']}) — {info_extra['modelo_base']}"
    plt.text(0.01, 0.98, titulo, fontsize=18, weight="bold", va="top")

    extra_txt = (
        f"Ruta modelo: {info_extra['ruta_modelo']}\n"
        f"Batch size: {info_extra['batch_size']}\n"
        f"Nº muestras: {info_extra['num_muestras']}\n"
        f"Distribución (y_true): {info_extra['distribucion_true']}\n"
        f"Distribución (y_pred): {info_extra['distribucion_pred']}\n"
        f"Parámetros modelo: {info_extra['num_params']}\n"
        f"Epochs ejecutados: {info_extra.get('epochs_ejecutados', 'N/A')}\n"
        f"Mejor epoch (val_loss): {info_extra.get('best_epoch_val_loss', 'N/A')} | val_loss: {info_extra.get('best_val_loss', 'N/A')}\n"
        f"Mejor epoch (val_acc): {info_extra.get('best_epoch_val_acc', 'N/A')} | val_acc: {info_extra.get('best_val_acc', 'N/A')}\n"
        f"Best run metric: {info_extra.get('monitor_metric', 'N/A')}\n"
        f"Best epoch (global): {info_extra.get('best_epoch', 'N/A')} | value: {info_extra.get('best_value', 'N/A')}\n"
    )
    plt.text(0.01, 0.88, extra_txt, fontsize=11, family="monospace", va="top")

    table = plt.table(
        cellText=filas,
        colLabels=columnas,
        loc="lower left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.10, 0.98, 0.52]
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
    parser = argparse.ArgumentParser(
        description="Evaluación multiclase de modelos entrenados"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=None,
        choices=MODELOS_DISPONIBLES,
        help="Nombre del modelo a evaluar",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cmap", type=str, default="viridis")

    args = parser.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    nombre_modelo = _elegir_modelo(args.modelo)
    figuras_modelo_dir, metricas_modelo_dir = _asegurar_carpetas_modelo(nombre_modelo)

    model_path = _resolver_ruta_modelo(nombre_modelo)
    modelo = _cargar_modelo(nombre_modelo, model_path)

    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    train_df, val_df, test_df = get_splits(create_if_missing=True)

    if args.split == "train":
        df_eval = train_df.copy()
    elif args.split == "val":
        df_eval = val_df.copy()
    else:
        df_eval = test_df.copy()

    ds_eval = crear_dataset_tf(
        df=df_eval,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=args.cache,
    )

    y_true, y_pred = _recoger_predicciones(modelo, ds_eval)

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    unique_true, counts_true = np.unique(y_true, return_counts=True)

    distrib_true = {int(k): int(v) for k, v in zip(unique_true, counts_true)}
    distrib_pred = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}

    ruta_cm = figuras_modelo_dir / f"matriz_confusion_{nombre_modelo}_{args.split}.png"
    guardar_matriz_confusion(
        y_true,
        y_pred,
        CLAVES_CLASES,
        ruta_cm,
        cmap=args.cmap,
    )

    reporte_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLAVES_CLASES,
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(accuracy_score(y_true, y_pred))

    path_history = _buscar_history_para_modelo(nombre_modelo)
    history = _cargar_history(path_history)

    path_best_run = _buscar_best_run_para_modelo(nombre_modelo)
    best_run = _cargar_best_run(path_best_run)

    info_train = _extraer_info_training(history)

    if best_run and isinstance(best_run, dict):
        metrics_run = best_run.get("metrics", {})

        info_train.update({
            "epochs_ejecutados": best_run.get(
                "epochs_ejecutados",
                info_train.get("epochs_ejecutados", "N/A")
            ),
            "best_epoch": best_run.get("best_epoch", "N/A"),
            "best_value": best_run.get("best_value", "N/A"),
            "monitor_metric": best_run.get("monitor_metric", "N/A"),
            "best_epoch_val_loss": best_run.get(
                "best_epoch",
                info_train.get("best_epoch_val_loss", "N/A")
            ),
            "best_val_loss": metrics_run.get(
                "val_loss",
                info_train.get("best_val_loss", "N/A")
            ),
            "best_epoch_val_acc": best_run.get(
                "best_epoch",
                info_train.get("best_epoch_val_acc", "N/A")
            ),
            "best_val_acc": metrics_run.get(
                "val_accuracy",
                info_train.get("best_val_acc", "N/A")
            ),
        })

    num_params = int(modelo.count_params())

    info_extra = {
        "modelo_base": nombre_modelo,
        "ruta_modelo": str(model_path),
        "split": args.split,
        "batch_size": batch_size,
        "num_muestras": int(len(y_true)),
        "distribucion_true": distrib_true,
        "distribucion_pred": distrib_pred,
        "num_params": num_params,
        "nombres_clases": CLAVES_CLASES,
        **info_train,
    }

    ruta_tabla = metricas_modelo_dir / f"tabla_metricas_{nombre_modelo}_{args.split}.png"
    _tabla_metricas_imagen(
        reporte_dict=reporte_dict,
        accuracy=accuracy,
        info_extra=info_extra,
        ruta_png=ruta_tabla,
    )

    resumen_json = metricas_modelo_dir / f"metricas_{nombre_modelo}_{args.split}.json"
    resumen = {
        "modelo": nombre_modelo,
        "ruta_modelo": str(model_path),
        "split": args.split,
        "batch_size": batch_size,
        "accuracy": accuracy,
        "num_muestras": int(len(y_true)),
        "distribucion_true": distrib_true,
        "distribucion_pred": distrib_pred,
        "num_params": num_params,
        "classification_report": reporte_dict,
        "history_path": str(path_history) if path_history else None,
        "best_run_path": str(path_best_run) if path_best_run else None,
        "matriz_confusion_png": str(ruta_cm),
        "tabla_metricas_png": str(ruta_tabla),
        **info_train,
    }
    resumen_json.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n✅ Evaluación completada")
    print(f"🧠 Modelo: {nombre_modelo}")
    print(f"📌 Ruta modelo: {model_path}")
    print(f"🧾 History: {path_history if path_history else 'NO ENCONTRADO'}")
    print(f"⭐ Best run: {path_best_run if path_best_run else 'NO ENCONTRADO'}")
    print(f"🧪 Split: {args.split}")
    print(f"🎯 Accuracy: {accuracy:.3f}")
    print(f"🖼️ Matriz de confusión: {ruta_cm}")
    print(f"🧾 Tabla métricas (PNG): {ruta_tabla}")
    print(f"📄 JSON métricas: {resumen_json}")

    if best_run and "best_epoch" in best_run:
        best_value = best_run.get("best_value", "N/A")
        monitor_metric = best_run.get("monitor_metric", "N/A")

        if isinstance(best_value, (int, float)):
            print(
                f"🏆 Best run → epoch {best_run['best_epoch']} "
                f"({monitor_metric} = {best_value:.5f})"
            )
        else:
            print(
                f"🏆 Best run → epoch {best_run['best_epoch']} "
                f"({monitor_metric} = {best_value})"
            )

    if len(unique_pred) == 1:
        print("\n⚠️ OJO: el modelo está prediciendo UNA sola clase.")
        print(f"   Clase predicha: {unique_pred[0]} | Conteo: {counts_pred[0]}")


if __name__ == "__main__":
    main()