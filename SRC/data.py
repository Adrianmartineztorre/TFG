# SRC/data.py

from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    SEED,
    RUTA_COLON,
    RUTA_LUNG,
    RUTA_SPLITS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    BATCH_SIZE,
    CLAVES_CLASES,
)

from preprocess import crear_dataset_tf


EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def construir_dataframe() -> pd.DataFrame:
    registros = []

    mapeo_carpetas = {}
    for clase in CLAVES_CLASES:
        if clase.startswith("lung_"):
            mapeo_carpetas[clase] = RUTA_LUNG / clase
        elif clase.startswith("colon_"):
            mapeo_carpetas[clase] = RUTA_COLON / clase
        else:
            raise ValueError(f"No se reconoce el prefijo de la clase: {clase}")

    for clase, carpeta in mapeo_carpetas.items():
        if not carpeta.exists():
            raise FileNotFoundError(f"No existe la carpeta para {clase}: {carpeta}")

        for img_path in carpeta.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in EXTS:
                registros.append(
                    {"filepath": str(img_path.resolve()), "label": clase}
                )

    df = pd.DataFrame(registros)
    if df.empty:
        raise RuntimeError("No se encontraron imágenes válidas.")

    return df


def crear_splits_estratificados(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_df = df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = df.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def verificar_fuga_datos(train_df, val_df, test_df) -> None:
    if (
        not set(train_df["filepath"]).isdisjoint(val_df["filepath"])
        or not set(train_df["filepath"]).isdisjoint(test_df["filepath"])
        or not set(val_df["filepath"]).isdisjoint(test_df["filepath"])
    ):
        raise ValueError("Fuga de datos detectada entre splits.")


def guardar_splits(train_df, val_df, test_df, out_dir: Path = RUTA_SPLITS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def cargar_splits(splits_dir: Path = RUTA_SPLITS):
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError("No existen los CSV de splits.")

    return (
        pd.read_csv(train_csv),
        pd.read_csv(val_csv),
        pd.read_csv(test_csv),
    )


def crear_splits(df: pd.DataFrame | None = None):
    if df is None:
        df = construir_dataframe()

    train_df, val_df, test_df = crear_splits_estratificados(
        df,
        TRAIN_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT,
        SEED,
    )

    verificar_fuga_datos(train_df, val_df, test_df)
    guardar_splits(train_df, val_df, test_df)

    return train_df, val_df, test_df


def get_splits(create_if_missing: bool = True):
    try:
        return cargar_splits()
    except FileNotFoundError:
        if not create_if_missing:
            raise
        return crear_splits()


def get_tf_datasets(batch_size: int = BATCH_SIZE, cache: bool = False):
    train_df, val_df, test_df = get_splits()

    train_ds = crear_dataset_tf(train_df, batch_size, entrenamiento=True, usar_cache=cache)
    val_ds = crear_dataset_tf(val_df, batch_size, entrenamiento=False, usar_cache=cache)
    test_ds = crear_dataset_tf(test_df, batch_size, entrenamiento=False, usar_cache=cache)

    return train_ds, val_ds, test_ds, train_df, val_df, test_df

