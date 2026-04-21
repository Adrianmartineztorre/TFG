# SRC/e_Evaluacion/auditar_modelo.py

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from a_Configuracion.config import (
    SEED,
    MODELOS_DIR,
    RUTA_OUTPUTS,
    BATCH_SIZE,
    CLAVES_CLASES,
)

from c_Data.data import get_splits
from b_Preprocesado.preprocess import crear_dataset_tf


# =========================================================
# CONFIG FIJA
# =========================================================
MODEL_NAME = "cnn_vgg_opt"
MODEL_FILENAME = "model.best.keras"
AUDIT_FOLDER_NAME = "cnn_vgg_opt"


# =========================================================
# CONFIG GPU
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"🚀 GPU(s) detectada(s): {gpus}")
else:
    print("⚠️ No se ha detectado GPU, se usará CPU.")


# =========================================================
# UTILIDADES
# =========================================================
def fijar_seed(seed=SEED):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _inferir_columna_ruta(df: pd.DataFrame) -> str:
    posibles = ["filepath", "ruta", "path", "file_path", "imagen", "image_path"]
    for col in posibles:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró columna de ruta en el DataFrame. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def _inferir_columna_etiqueta(df: pd.DataFrame) -> str:
    posibles = ["label", "etiqueta", "clase", "target", "y"]
    for col in posibles:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró columna de etiqueta en el DataFrame. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def _normalizar_ruta(p):
    return str(Path(p)).replace("\\", "/").strip().lower()


def _basename(p):
    return Path(p).name.lower().strip()


def _sha256_archivo(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hashes_archivos(paths):
    hashes = {}
    for p in paths:
        try:
            hashes[p] = _sha256_archivo(p)
        except Exception as e:
            print(f"⚠️ No se pudo hashear {p}: {e}")
    return hashes


def _buscar_interseccion(conj_a, conj_b):
    return sorted(list(set(conj_a).intersection(set(conj_b))))


def _analizar_solapes_por_ruta(train_df, val_df, test_df, col_ruta):
    train_paths = [_normalizar_ruta(x) for x in train_df[col_ruta].tolist()]
    val_paths = [_normalizar_ruta(x) for x in val_df[col_ruta].tolist()]
    test_paths = [_normalizar_ruta(x) for x in test_df[col_ruta].tolist()]

    train_val = _buscar_interseccion(train_paths, val_paths)
    train_test = _buscar_interseccion(train_paths, test_paths)
    val_test = _buscar_interseccion(val_paths, test_paths)

    return {
        "train_val_overlap_count": len(train_val),
        "train_test_overlap_count": len(train_test),
        "val_test_overlap_count": len(val_test),
        "train_val_examples": train_val[:20],
        "train_test_examples": train_test[:20],
        "val_test_examples": val_test[:20],
    }


def _analizar_solapes_por_nombre(train_df, val_df, test_df, col_ruta):
    train_names = [_basename(x) for x in train_df[col_ruta].tolist()]
    val_names = [_basename(x) for x in val_df[col_ruta].tolist()]
    test_names = [_basename(x) for x in test_df[col_ruta].tolist()]

    train_val = _buscar_interseccion(train_names, val_names)
    train_test = _buscar_interseccion(train_names, test_names)
    val_test = _buscar_interseccion(val_names, test_names)

    return {
        "train_val_same_filename_count": len(train_val),
        "train_test_same_filename_count": len(train_test),
        "val_test_same_filename_count": len(val_test),
        "train_val_examples": train_val[:20],
        "train_test_examples": train_test[:20],
        "val_test_examples": val_test[:20],
    }


def _analizar_duplicados_binarios(train_df, val_df, test_df, col_ruta):
    print("\n🔍 Calculando hashes SHA256 para detectar duplicados exactos entre splits...")
    train_paths = [str(Path(x)) for x in train_df[col_ruta].tolist()]
    val_paths = [str(Path(x)) for x in val_df[col_ruta].tolist()]
    test_paths = [str(Path(x)) for x in test_df[col_ruta].tolist()]

    train_hashes = _hashes_archivos(train_paths)
    val_hashes = _hashes_archivos(val_paths)
    test_hashes = _hashes_archivos(test_paths)

    hash_to_train = defaultdict(list)
    hash_to_val = defaultdict(list)
    hash_to_test = defaultdict(list)

    for p, h in train_hashes.items():
        hash_to_train[h].append(p)
    for p, h in val_hashes.items():
        hash_to_val[h].append(p)
    for p, h in test_hashes.items():
        hash_to_test[h].append(p)

    train_val_dup = []
    train_test_dup = []
    val_test_dup = []

    for h in set(hash_to_train).intersection(set(hash_to_val)):
        for a in hash_to_train[h]:
            for b in hash_to_val[h]:
                train_val_dup.append({"hash": h, "train": a, "val": b})

    for h in set(hash_to_train).intersection(set(hash_to_test)):
        for a in hash_to_train[h]:
            for b in hash_to_test[h]:
                train_test_dup.append({"hash": h, "train": a, "test": b})

    for h in set(hash_to_val).intersection(set(hash_to_test)):
        for a in hash_to_val[h]:
            for b in hash_to_test[h]:
                val_test_dup.append({"hash": h, "val": a, "test": b})

    return {
        "train_val_exact_duplicates_count": len(train_val_dup),
        "train_test_exact_duplicates_count": len(train_test_dup),
        "val_test_exact_duplicates_count": len(val_test_dup),
        "train_val_examples": train_val_dup[:20],
        "train_test_examples": train_test_dup[:20],
        "val_test_examples": val_test_dup[:20],
    }


def _crear_dataset_desde_df(df, col_ruta, col_etiqueta, batch_size=32, cache=False):
    df_tmp = df[[col_ruta, col_etiqueta]].copy()
    df_tmp = df_tmp.rename(columns={
        col_ruta: "filepath",
        col_etiqueta: "label",
    })

    etiquetas_raw = df_tmp["label"].tolist()

    if all(isinstance(x, (int, np.integer)) for x in etiquetas_raw):
        y_true = np.array(etiquetas_raw, dtype=np.int32)
        df_tmp["label"] = [CLAVES_CLASES[i] for i in y_true]
    else:
        mapa = {clase: i for i, clase in enumerate(CLAVES_CLASES)}
        y_true = np.array([mapa[str(x)] for x in etiquetas_raw], dtype=np.int32)
        df_tmp["label"] = df_tmp["label"].astype(str)

    ds = crear_dataset_tf(
        df=df_tmp,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=cache,
    )

    return ds, y_true


def _evaluar_split(model, ds, y_true, nombre_split):
    print(f"\n📊 Evaluando split: {nombre_split} ...")
    y_prob = model.predict(ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Desajuste en split {nombre_split}: "
            f"{len(y_pred)} predicciones vs {len(y_true)} etiquetas"
        )

    acc = accuracy_score(y_true, y_pred)

    return {
        "split": nombre_split,
        "accuracy": float(acc),
        "n_samples": int(len(y_true)),
    }


def _guardar_grafica_overfitting(resultados, save_path):
    splits = [r["split"] for r in resultados]
    accs = [r["accuracy"] for r in resultados]

    plt.figure(figsize=(8, 5))
    plt.bar(splits, accs)
    plt.ylim(0, 1)
    plt.title("Comparación de accuracy por split")
    plt.ylabel("Accuracy")
    plt.xlabel("Split")

    for i, v in enumerate(accs):
        plt.text(i, min(v + 0.02, 0.98), f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _interpretar_riesgo_overfitting(train_acc, val_acc, test_acc):
    gap_train_val = train_acc - val_acc
    gap_train_test = train_acc - test_acc
    gap_val_test = abs(val_acc - test_acc)

    conclusiones = []

    if gap_train_val > 0.05:
        conclusiones.append(
            f"Posible overfitting: train_acc - val_acc = {gap_train_val:.4f} (> 0.05)"
        )
    else:
        conclusiones.append(
            f"No se observa overfitting fuerte por train/val: gap = {gap_train_val:.4f}"
        )

    if gap_train_test > 0.05:
        conclusiones.append(
            f"Posible overfitting: train_acc - test_acc = {gap_train_test:.4f} (> 0.05)"
        )
    else:
        conclusiones.append(
            f"Generalización razonable frente a test: gap = {gap_train_test:.4f}"
        )

    if gap_val_test > 0.03:
        conclusiones.append(
            f"Val y test difieren bastante: |val_acc - test_acc| = {gap_val_test:.4f}"
        )
    else:
        conclusiones.append(
            f"Val y test son consistentes: gap = {gap_val_test:.4f}"
        )

    return {
        "train_val_gap": float(gap_train_val),
        "train_test_gap": float(gap_train_test),
        "val_test_gap": float(gap_val_test),
        "conclusiones": conclusiones,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Auditoría de leakage y overfitting para cnn_vgg_opt"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help=f"Ruta manual al modelo .keras. Si se deja vacío, usa MODELOS_DIR/{MODEL_NAME}/{MODEL_FILENAME}"
    )
    args = parser.parse_args()

    fijar_seed()

    # -----------------------------------------------------
    # Cargar splits
    # -----------------------------------------------------
    print("\n📦 Cargando splits...")
    train_df, val_df, test_df = get_splits()

    col_ruta = _inferir_columna_ruta(train_df)
    col_etiqueta = _inferir_columna_etiqueta(train_df)

    print(f"✅ Columna ruta detectada: {col_ruta}")
    print(f"✅ Columna etiqueta detectada: {col_etiqueta}")

    print(f"✅ Train samples: {len(train_df)}")
    print(f"✅ Val samples:   {len(val_df)}")
    print(f"✅ Test samples:  {len(test_df)}")

    # -----------------------------------------------------
    # Auditoría de leakage
    # -----------------------------------------------------
    print("\n🔎 Auditando posibles fugas de datos...")
    overlap_paths = _analizar_solapes_por_ruta(train_df, val_df, test_df, col_ruta)
    overlap_names = _analizar_solapes_por_nombre(train_df, val_df, test_df, col_ruta)
    exact_dups = _analizar_duplicados_binarios(train_df, val_df, test_df, col_ruta)

    # -----------------------------------------------------
    # Cargar datasets
    # -----------------------------------------------------
    print("\n🧪 Construyendo datasets de evaluación...")
    train_ds, y_train = _crear_dataset_desde_df(
        train_df, col_ruta, col_etiqueta, batch_size=args.batch_size, cache=args.cache
    )
    val_ds, y_val = _crear_dataset_desde_df(
        val_df, col_ruta, col_etiqueta, batch_size=args.batch_size, cache=args.cache
    )
    test_ds, y_test = _crear_dataset_desde_df(
        test_df, col_ruta, col_etiqueta, batch_size=args.batch_size, cache=args.cache
    )

    # -----------------------------------------------------
    # Cargar modelo
    # -----------------------------------------------------
    if args.model_path.strip():
        model_path = Path(args.model_path)
    else:
        model_path = MODELOS_DIR / MODEL_NAME / MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"❌ No se encontró el modelo en: {model_path}")

    print(f"\n🧠 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # -----------------------------------------------------
    # Evaluar splits
    # -----------------------------------------------------
    resultados = []
    resultados.append(_evaluar_split(model, train_ds, y_train, "train"))
    resultados.append(_evaluar_split(model, val_ds, y_val, "val"))
    resultados.append(_evaluar_split(model, test_ds, y_test, "test"))

    train_acc = resultados[0]["accuracy"]
    val_acc = resultados[1]["accuracy"]
    test_acc = resultados[2]["accuracy"]

    overfitting = _interpretar_riesgo_overfitting(train_acc, val_acc, test_acc)

    # -----------------------------------------------------
    # Carpeta de salida
    # -----------------------------------------------------
    audit_dir = RUTA_OUTPUTS / "Auditacion" / AUDIT_FOLDER_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Guardar única gráfica
    # -----------------------------------------------------
    plot_path = audit_dir / "comparacion_accuracy_splits.png"
    _guardar_grafica_overfitting(resultados, plot_path)

    leakage_detectado = (
        overlap_paths["train_val_overlap_count"] > 0
        or overlap_paths["train_test_overlap_count"] > 0
        or overlap_paths["val_test_overlap_count"] > 0
        or overlap_names["train_val_same_filename_count"] > 0
        or overlap_names["train_test_same_filename_count"] > 0
        or overlap_names["val_test_same_filename_count"] > 0
        or exact_dups["train_val_exact_duplicates_count"] > 0
        or exact_dups["train_test_exact_duplicates_count"] > 0
        or exact_dups["val_test_exact_duplicates_count"] > 0
    )

    possible_overfitting = bool(
        overfitting["train_val_gap"] > 0.05 or overfitting["train_test_gap"] > 0.05
    )

    # -----------------------------------------------------
    # Resumen final
    # -----------------------------------------------------
    resumen = {
        "model_name": MODEL_NAME,
        "model_path": str(model_path),
        "output_dir": str(audit_dir),
        "json_path": str(audit_dir / "auditoria_leakage_overfitting.json"),
        "plot_path": str(plot_path),

        "n_samples": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },

        "leakage": {
            "solape_train_val_por_ruta": int(overlap_paths["train_val_overlap_count"]),
            "solape_train_test_por_ruta": int(overlap_paths["train_test_overlap_count"]),
            "solape_val_test_por_ruta": int(overlap_paths["val_test_overlap_count"]),

            "mismo_nombre_train_val": int(overlap_names["train_val_same_filename_count"]),
            "mismo_nombre_train_test": int(overlap_names["train_test_same_filename_count"]),
            "mismo_nombre_val_test": int(overlap_names["val_test_same_filename_count"]),

            "duplicados_exactos_train_val": int(exact_dups["train_val_exact_duplicates_count"]),
            "duplicados_exactos_train_test": int(exact_dups["train_test_exact_duplicates_count"]),
            "duplicados_exactos_val_test": int(exact_dups["val_test_exact_duplicates_count"]),
        },

        "accuracy": {
            "train": float(train_acc),
            "val": float(val_acc),
            "test": float(test_acc),
        },

        "overfitting": {
            "train_val_gap": float(overfitting["train_val_gap"]),
            "train_test_gap": float(overfitting["train_test_gap"]),
            "val_test_gap": float(overfitting["val_test_gap"]),
            "conclusiones": overfitting["conclusiones"],
        },

        "summary": {
            "possible_data_leakage": bool(leakage_detectado),
            "possible_overfitting": bool(possible_overfitting),
        },
    }

    out_json = audit_dir / "auditoria_leakage_overfitting.json"
    out_json.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Print final
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ AUDITORÍA FINALIZADA")
    print("=" * 70)
    print(f"📄 Informe JSON: {out_json}")
    print(f"🖼️ Gráfica generada en: {plot_path}")
    print()
    print("----- Leakage -----")
    print(f"Solape train-val por ruta:   {resumen['leakage']['solape_train_val_por_ruta']}")
    print(f"Solape train-test por ruta:  {resumen['leakage']['solape_train_test_por_ruta']}")
    print(f"Solape val-test por ruta:    {resumen['leakage']['solape_val_test_por_ruta']}")
    print(f"Mismo nombre train-val:      {resumen['leakage']['mismo_nombre_train_val']}")
    print(f"Mismo nombre train-test:     {resumen['leakage']['mismo_nombre_train_test']}")
    print(f"Mismo nombre val-test:       {resumen['leakage']['mismo_nombre_val_test']}")
    print(f"Dup exactos train-val:       {resumen['leakage']['duplicados_exactos_train_val']}")
    print(f"Dup exactos train-test:      {resumen['leakage']['duplicados_exactos_train_test']}")
    print(f"Dup exactos val-test:        {resumen['leakage']['duplicados_exactos_val_test']}")
    print()
    print("----- Accuracy -----")
    print(f"Train: {resumen['accuracy']['train']:.4f}")
    print(f"Val:   {resumen['accuracy']['val']:.4f}")
    print(f"Test:  {resumen['accuracy']['test']:.4f}")
    print()
    print("----- Conclusión overfitting -----")
    for c in resumen["overfitting"]["conclusiones"]:
        print(f"- {c}")
    print()
    print(f"🚨 Posible data leakage: {resumen['summary']['possible_data_leakage']}")
    print(f"🚨 Posible overfitting:  {resumen['summary']['possible_overfitting']}")


if __name__ == "__main__":
    main()