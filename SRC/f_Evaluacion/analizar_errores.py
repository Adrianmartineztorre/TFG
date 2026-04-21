# SRC/f_Evaluacion/analizar_errores.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from a_Configuracion.config import (
    SEED,
    MODELOS_DIR,
    FIGURAS_DIR,
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
    """
    Adapta el dataframe al formato esperado por preprocess.crear_dataset_tf:
    columnas obligatorias -> filepath, label
    """
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


def _guardar_json(path: Path, data: dict):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _copiar_muestras_error(df_errores, out_dir: Path, max_por_confusion=5):
    """
    Copia unas pocas imágenes mal clasificadas para revisión visual.
    """
    try:
        import shutil

        out_dir.mkdir(parents=True, exist_ok=True)

        conteo = {}

        for _, row in df_errores.iterrows():
            clave = f"{row['clase_real']}__a__{row['clase_predicha']}"
            conteo.setdefault(clave, 0)

            if conteo[clave] >= max_por_confusion:
                continue

            src = Path(row["ruta"])
            if not src.exists():
                continue

            subdir = out_dir / clave
            subdir.mkdir(parents=True, exist_ok=True)

            dst = subdir / src.name
            shutil.copy2(src, dst)
            conteo[clave] += 1

    except Exception as e:
        print(f"⚠️ No se pudieron copiar muestras de error: {e}")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Analizar fallos del modelo")
    parser.add_argument("--modelo", type=str, default="cnn_vgg_opt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--guardar_imagenes_error", action="store_true")
    args = parser.parse_args()

    fijar_seed()

    # -----------------------------------------------------
    # Cargar splits
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    ds, rutas, y_true = _crear_dataset(
        df,
        col_ruta=col_ruta,
        col_etiqueta=col_etiqueta,
        batch_size=args.batch_size,
        cache=args.cache,
    )

    # -----------------------------------------------------
    # Modelo
    # -----------------------------------------------------
    model_path = MODELOS_DIR / args.modelo / "model.best.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print(f"🧠 Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # -----------------------------------------------------
    # Predicción
    # -----------------------------------------------------
    print("🚀 Generando predicciones...")
    y_prob = model.predict(ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Desajuste entre predicciones ({len(y_pred)}) y etiquetas reales ({len(y_true)})."
        )

    # -----------------------------------------------------
    # Métricas globales
    # -----------------------------------------------------
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLAVES_CLASES,
        output_dict=True,
        zero_division=0,
    )

    # -----------------------------------------------------
    # Errores detallados
    # -----------------------------------------------------
    filas_error = []

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue

        prob_real = float(y_prob[i][y_true[i]])
        prob_pred = float(y_prob[i][y_pred[i]])
        top3 = _top_k(y_prob[i], k=3)

        filas_error.append({
            "indice": i,
            "ruta": rutas[i],
            "clase_real_idx": int(y_true[i]),
            "clase_real": CLAVES_CLASES[y_true[i]],
            "clase_predicha_idx": int(y_pred[i]),
            "clase_predicha": CLAVES_CLASES[y_pred[i]],
            "prob_clase_real": prob_real,
            "prob_clase_predicha": prob_pred,
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

    # -----------------------------------------------------
    # Resumen de fallos
    # -----------------------------------------------------
    total = len(y_true)
    total_errores = len(df_errores)
    tasa_error = float(total_errores / total) if total > 0 else 0.0

    if total_errores > 0:
        confusiones = (
            df_errores
            .groupby(["clase_real", "clase_predicha"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )

        errores_por_clase_real = (
            df_errores
            .groupby("clase_real")
            .size()
            .reset_index(name="n_errores")
            .sort_values("n_errores", ascending=False)
        )

        confianza_media_error = float(df_errores["confianza_prediccion"].mean())
        margen_medio = float(df_errores["margen_top1_top2"].mean())

        errores_alta_confianza = int((df_errores["confianza_prediccion"] >= 0.80).sum())
        errores_baja_separacion = int((df_errores["margen_top1_top2"] <= 0.15).sum())
    else:
        confusiones = pd.DataFrame(columns=["clase_real", "clase_predicha", "n"])
        errores_por_clase_real = pd.DataFrame(columns=["clase_real", "n_errores"])
        confianza_media_error = 0.0
        margen_medio = 0.0
        errores_alta_confianza = 0
        errores_baja_separacion = 0

    # -----------------------------------------------------
    # Hipótesis automáticas para discusión
    # -----------------------------------------------------
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

        if len(confusiones) > 0:
            top_conf = confusiones.iloc[0]
            hipotesis.append(
                f"La confusión más frecuente es '{top_conf['clase_real']}' → '{top_conf['clase_predicha']}', por lo que esa pareja debería revisarse visualmente en trabajos futuros."
            )

        hipotesis.append(
            "Sería recomendable revisar manualmente los casos fallados para identificar artefactos de tinción, calidad de imagen, zonas con poco tejido representativo o similitud morfológica entre clases."
        )

        hipotesis.append(
            "Como trabajo futuro, sería útil incorporar análisis por paciente, por slide o por origen de parche para comprobar si ciertos fallos están asociados a distribución de datos o a variabilidad biológica."
        )

    # -----------------------------------------------------
    # Guardado único
    # -----------------------------------------------------
    out_dir = Path("OUTPUT") / "ERRORES" / args.modelo
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"analisis_errores_{args.split}.json"

    output = {
        "info_general": {
            "modelo": args.modelo,
            "split": args.split,
            "model_path": str(model_path),
            "n_muestras": int(total),
            "accuracy": acc,
            "n_errores": int(total_errores),
            "tasa_error": tasa_error,
        },
        "metricas_error": {
            "confianza_media_en_errores": confianza_media_error,
            "margen_top1_top2_medio_en_errores": margen_medio,
            "errores_alta_confianza": errores_alta_confianza,
            "errores_baja_separacion_top1_top2": errores_baja_separacion,
        },
        "clasificacion": {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "analisis_fallos": {
            "top_confusiones": confusiones.to_dict(orient="records"),
            "errores_por_clase_real": errores_por_clase_real.to_dict(orient="records"),
        },
        "errores_detallados": df_errores.to_dict(orient="records"),
        "hipotesis_trabajo_futuro": hipotesis,
    }

    _guardar_json(output_path, output)

    # -----------------------------------------------------
    # Copia opcional de imágenes falladas
    # -----------------------------------------------------
    if args.guardar_imagenes_error and total_errores > 0:
        fallos_dir = FIGURAS_DIR / args.modelo / f"fallos_{args.split}"
        _copiar_muestras_error(df_errores, fallos_dir, max_por_confusion=5)
        print(f"🖼️ Imágenes de fallos copiadas en: {fallos_dir}")

    # -----------------------------------------------------
    # Print final
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS DE FALLOS FINALIZADO")
    print("=" * 70)
    print(f"📄 Archivo generado: {output_path}")
    print()
    print(f"Accuracy:   {acc:.4f}")
    print(f"Nº errores: {total_errores}/{total} ({tasa_error:.4%})")

    if total_errores > 0:
        print("\nConfusiones más frecuentes:")
        print(confusiones.head(10).to_string(index=False))

        print("\nHipótesis para discusión / trabajo futuro:")
        for h in hipotesis:
            print(f"- {h}")


if __name__ == "__main__":
    main()