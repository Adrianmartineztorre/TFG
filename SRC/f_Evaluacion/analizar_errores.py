# SRC/f_Evaluacion/analizar_errores.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from a_Configuracion.config import (
    SEED,
    MODELOS_DIR,
    FIGURAS_DIR,
    RUTA_OUTPUTS,
    BATCH_SIZE,
    CLAVES_CLASES,
)

from c_Data.data import get_splits
from b_Preprocesado.preprocess import crear_dataset_tf


# =========================================================
# GPU
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


# =========================================================
# Utils
# =========================================================
def fijar_seed(seed=SEED):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _inferir_columna_ruta(df):
    for c in ["filepath", "ruta", "path", "file_path", "imagen", "image_path"]:
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna de ruta. Columnas: {list(df.columns)}")


def _inferir_columna_etiqueta(df):
    for c in ["label", "etiqueta", "clase", "target", "y"]:
        if c in df.columns:
            return c
    raise ValueError(f"No encuentro columna de etiqueta. Columnas: {list(df.columns)}")


def _pasar_a_indices(etiquetas_raw):
    if all(isinstance(x, (int, np.integer)) for x in etiquetas_raw):
        return np.array(etiquetas_raw, dtype=np.int32)

    mapa = {clase: i for i, clase in enumerate(CLAVES_CLASES)}
    return np.array([mapa[str(x)] for x in etiquetas_raw], dtype=np.int32)


def _crear_dataset(df, col_ruta, col_etiqueta, batch_size=32, cache=False):
    df_modelo = df[[col_ruta, col_etiqueta]].copy()
    df_modelo.columns = ["filepath", "label"]

    ds = crear_dataset_tf(
        df=df_modelo,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=cache,
    )

    rutas = df_modelo["filepath"].astype(str).tolist()
    etiquetas = _pasar_a_indices(df_modelo["label"].tolist())

    return ds, rutas, etiquetas


def _top_k(pred_vector, k=3):
    idxs = np.argsort(pred_vector)[::-1][:k]
    return [(CLAVES_CLASES[i], float(pred_vector[i])) for i in idxs]


def _copiar_muestras_error(df_errores, out_dir: Path, max_por_confusion=5):
    try:
        import shutil

        out_dir.mkdir(parents=True, exist_ok=True)
        conteo = {}

        for _, row in df_errores.iterrows():
            clave = f"{row['era_en_verdad']}__a__{row['predijo']}"
            conteo.setdefault(clave, 0)

            if conteo[clave] >= max_por_confusion:
                continue

            src = Path(row["ruta_imagen"])
            if not src.exists():
                continue

            subdir = out_dir / clave
            subdir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, subdir / src.name)
            conteo[clave] += 1

    except Exception as e:
        print(f"⚠️ No se pudieron copiar muestras de error: {e}")


def _guardar_tabla_confusiones_academica(
    confusiones,
    save_path,
    accuracy,
    total_errores,
    total_muestras,
    tasa_error,
    top_n=10,
):
    tabla = confusiones.head(top_n).copy()

    tabla = tabla.rename(
        columns={
            "era_en_verdad": "Clase real",
            "predijo": "Clase predicha",
            "n": "Nº errores",
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "Confusiones más frecuentes registradas en el conjunto de prueba",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
    )

    texto_metricas = (
        f"Accuracy:   {accuracy:.4f}\n"
        f"Nº errores: {total_errores}/{total_muestras} ({tasa_error:.4%})"
    )

    ax.text(
        0.02,
        0.82,
        texto_metricas,
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        family="monospace",
    )

    tabla_plot = ax.table(
        cellText=tabla.values,
        colLabels=tabla.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.66],
    )

    tabla_plot.auto_set_font_size(False)
    tabla_plot.set_fontsize(11)
    tabla_plot.scale(1, 1.75)

    for (fila, columna), celda in tabla_plot.get_celld().items():
        celda.set_edgecolor("#333333")
        celda.set_linewidth(0.6)
        celda.set_facecolor("white")
        celda.PAD = 0.15

        if fila == 0:
            celda.set_text_props(
                weight="bold",
                fontsize=11.5,
                ha="center",
                va="center",
                fontfamily="sans-serif",
            )
            celda.set_linewidth(0.9)
        else:
            celda.set_text_props(
                weight="normal",
                fontsize=11,
                ha="center",
                va="center",
                fontfamily="sans-serif",
            )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Analizar fallos del modelo")
    parser.add_argument("--modelo", type=str, default="cnn_vgg_optimizado")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--guardar_imagenes_error", action="store_true")
    args = parser.parse_args()

    fijar_seed()

    train_df, val_df, test_df = get_splits()

    if args.split == "train":
        df = train_df.copy()
    elif args.split == "val":
        df = val_df.copy()
    else:
        df = test_df.copy()

    col_ruta = _inferir_columna_ruta(df)
    col_etiqueta = _inferir_columna_etiqueta(df)

    print(f"📦 Split analizado: {args.split}")
    print(f"✅ Nº muestras: {len(df)}")
    print(f"✅ Columna ruta: {col_ruta}")
    print(f"✅ Columna etiqueta: {col_etiqueta}")

    ds, rutas, y_true = _crear_dataset(
        df,
        col_ruta=col_ruta,
        col_etiqueta=col_etiqueta,
        batch_size=args.batch_size,
        cache=args.cache,
    )

    model_path = MODELOS_DIR / args.modelo / "model.best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print(f"🧠 Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("🚀 Generando predicciones...")
    y_prob = model.predict(ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Desajuste entre predicciones ({len(y_pred)}) y etiquetas reales ({len(y_true)})."
        )

    acc = float((y_pred == y_true).mean())
    total = len(y_true)
    total_errores = int((y_pred != y_true).sum())
    tasa_error = float(total_errores / total) if total > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred)

    filas_error = []

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue

        top3 = _top_k(y_prob[i], k=3)

        filas_error.append({
            "indice": i,
            "ruta_imagen": rutas[i],
            "era_en_verdad": CLAVES_CLASES[y_true[i]],
            "predijo": CLAVES_CLASES[y_pred[i]],
            "probabilidad_clase_real": float(y_prob[i][y_true[i]]),
            "probabilidad_clase_predicha": float(y_prob[i][y_pred[i]]),
            "confianza_prediccion": float(np.max(y_prob[i])),
            "top1_clase": top3[0][0],
            "top1_prob": top3[0][1],
            "top2_clase": top3[1][0],
            "top2_prob": top3[1][1],
            "top3_clase": top3[2][0],
            "top3_prob": top3[2][1],
            "margen_top1_top2": float(top3[0][1] - top3[1][1]),
        })

    df_errores = pd.DataFrame(filas_error)

    if total_errores > 0:
        confusiones = (
            df_errores
            .groupby(["era_en_verdad", "predijo"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )

        errores_alta_confianza = int((df_errores["confianza_prediccion"] >= 0.80).sum())
        errores_baja_separacion = int((df_errores["margen_top1_top2"] <= 0.15).sum())
    else:
        confusiones = pd.DataFrame(columns=["era_en_verdad", "predijo", "n"])
        errores_alta_confianza = 0
        errores_baja_separacion = 0

    hipotesis = []

    if total_errores == 0:
        hipotesis.append(
            "No se observaron errores en el split analizado; conviene confirmar con un conjunto externo o validación más exigente."
        )
    else:
        if errores_alta_confianza / total_errores > 0.30:
            hipotesis.append(
                "Una parte relevante de los errores ocurre con alta confianza, lo que podría indicar regiones visuales ambiguas, sesgos del dataset o ejemplos atípicos no bien representados."
            )

        if errores_baja_separacion / total_errores > 0.40:
            hipotesis.append(
                "Muchos errores presentan poca diferencia entre top-1 y top-2, lo que sugiere fronteras de decisión próximas entre clases histológicamente similares."
            )

        top_conf = confusiones.iloc[0]
        hipotesis.append(
            f"La confusión más frecuente es '{top_conf['era_en_verdad']}' → '{top_conf['predijo']}', por lo que esa pareja debería revisarse visualmente en trabajos futuros."
        )

        hipotesis.append(
            "Sería recomendable revisar manualmente los casos fallados para identificar artefactos de tinción, calidad de imagen, zonas con poco tejido representativo o similitud morfológica entre clases."
        )

        hipotesis.append(
            "Como trabajo futuro, sería útil incorporar análisis por paciente, por slide o por origen de parche para comprobar si ciertos fallos están asociados a distribución de datos o a variabilidad biológica."
        )

    ERRORES_DIR = RUTA_OUTPUTS / "ERRORES"
    out_dir = ERRORES_DIR / args.modelo / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    informe_path = out_dir / f"informe_errores_{args.modelo}_{args.split}.txt"
    tabla_confusiones_img_path = out_dir / f"tabla_confusiones_frecuentes_{args.modelo}_{args.split}.png"

    if total_errores > 0:
        _guardar_tabla_confusiones_academica(
            confusiones=confusiones,
            save_path=tabla_confusiones_img_path,
            accuracy=acc,
            total_errores=total_errores,
            total_muestras=total,
            tasa_error=tasa_error,
            top_n=10,
        )

    with open(informe_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy:   {acc:.4f}\n")
        f.write(f"Nº errores: {total_errores}/{total} ({tasa_error:.4%})\n\n")

        f.write("Confusiones más frecuentes:\n")
        if total_errores > 0:
            f.write(
                confusiones
                .rename(columns={
                    "era_en_verdad": "clase_real",
                    "predijo": "clase_predicha",
                })
                .to_string(index=False)
            )
        else:
            f.write("No se observaron errores en el split analizado.")

        f.write("\n\n")
        f.write("Hipótesis para discusión / trabajo futuro:\n")

        for h in hipotesis:
            f.write(f"- {h}\n")

    if args.guardar_imagenes_error and total_errores > 0:
        fallos_dir = ERRORES_DIR / args.modelo / args.split / "imagenes_falladas"
        _copiar_muestras_error(df_errores, fallos_dir, max_por_confusion=5)
        print(f"🖼️ Imágenes de fallos copiadas en: {fallos_dir}")

    print("\n" + "=" * 70)
    print("✅ ANÁLISIS DE FALLOS FINALIZADO")
    print("=" * 70)
    print(f"📄 Informe generado: {informe_path}")

    if total_errores > 0:
        print(f"🖼️ Tabla académica generada: {tabla_confusiones_img_path}")

    print()
    print(f"Accuracy:   {acc:.4f}")
    print(f"Nº errores: {total_errores}/{total} ({tasa_error:.4%})")

    if total_errores > 0:
        print("\nConfusiones más frecuentes:")
        print(
            confusiones
            .rename(columns={
                "era_en_verdad": "clase_real",
                "predijo": "clase_predicha",
            })
            .head(10)
            .to_string(index=False)
        )

        print("\nHipótesis para discusión / trabajo futuro:")
        for h in hipotesis:
            print(f"- {h}")


if __name__ == "__main__":
    main()