"""
Módulo de gestión de datos para el pipeline de entrenamiento.
Construye el dataframe a partir de las carpetas del dataset original.
Genera divisiones estratificadas de entrenamiento, validación y test.
Guarda y recupera los splits en formato CSV para reutilizarlos.
Se utiliza como puente entre la configuración global y el preprocesado.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from a_Configuracion.config_antiguo import (
    BATCH_SIZE,
    CLAVES_CLASES,
    RUTA_COLON,
    RUTA_LUNG,
    RUTA_SPLITS,
    SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from b_Preprocesado.preprocess import crear_dataset_tf


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalizar_ruta_entorno(path_str: str) -> str:
    """
    Normaliza rutas para que funcionen en el entorno actual.

    - Si recibe una ruta de Windows tipo C:\\Users\\... y estamos en WSL/Linux,
      la convierte a /mnt/c/Users/...
    - Si ya viene en formato Linux/WSL, la deja como está.
    """
    if not isinstance(path_str, str):
        return path_str

    path_str = path_str.strip()

    return str(Path(path_str))


def normalizar_dataframe_rutas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la normalización de rutas a la columna filepath.
    """
    df = df.copy()
    if "filepath" not in df.columns:
        raise KeyError("El dataframe no contiene la columna 'filepath'.")
    df["filepath"] = df["filepath"].apply(normalizar_ruta_entorno)
    return df


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
            raise FileNotFoundError(
                f"No existe la carpeta para la clase '{clase}': {carpeta}"
            )

        for ruta_imagen in carpeta.rglob("*"):
            if ruta_imagen.is_file() and ruta_imagen.suffix.lower() in EXTENSIONES_VALIDAS:
                registros.append(
                    {
                        "filepath": str(ruta_imagen.resolve()),
                        "label": clase,
                    }
                )

    df = pd.DataFrame(registros)

    if df.empty:
        raise RuntimeError("No se encontraron imágenes válidas en las carpetas del dataset.")

    df = normalizar_dataframe_rutas(df)
    return df


def crear_splits_estratificados(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Los ratios de train, val y test deben sumar 1.0.")

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for _, grupo in df.groupby("label"):
        indices = grupo.index.to_numpy()
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    train_df = df.loc[train_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = df.loc[val_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def verificar_fuga_datos(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_paths = set(train_df["filepath"])
    val_paths = set(val_df["filepath"])
    test_paths = set(test_df["filepath"])

    if (
        not train_paths.isdisjoint(val_paths)
        or not train_paths.isdisjoint(test_paths)
        or not val_paths.isdisjoint(test_paths)
    ):
        raise ValueError("Fuga de datos detectada entre los splits.")


def verificar_existencia_archivos(df: pd.DataFrame, nombre_split: str) -> None:
    """
    Comprueba que todos los filepaths del split existan en disco.
    """
    inexistentes = [p for p in df["filepath"] if not Path(p).exists()]

    if inexistentes:
        ejemplo = inexistentes[0]
        raise FileNotFoundError(
            f"Se han encontrado {len(inexistentes)} rutas inexistentes en el split '{nombre_split}'.\n"
            f"Ejemplo: {ejemplo}"
        )


def guardar_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path = RUTA_SPLITS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = normalizar_dataframe_rutas(train_df)
    val_df = normalizar_dataframe_rutas(val_df)
    test_df = normalizar_dataframe_rutas(test_df)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def cargar_splits(
    splits_dir: Path = RUTA_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError("No existen todos los archivos CSV de splits.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_df = normalizar_dataframe_rutas(train_df)
    val_df = normalizar_dataframe_rutas(val_df)
    test_df = normalizar_dataframe_rutas(test_df)

    return train_df, val_df, test_df


def crear_splits(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = construir_dataframe()

    train_df, val_df, test_df = crear_splits_estratificados(
        df=df,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT,
        seed=SEED,
    )

    verificar_fuga_datos(train_df, val_df, test_df)
    verificar_existencia_archivos(train_df, "train")
    verificar_existencia_archivos(val_df, "val")
    verificar_existencia_archivos(test_df, "test")

    guardar_splits(train_df, val_df, test_df)

    return train_df, val_df, test_df


def get_splits(
    create_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        train_df, val_df, test_df = cargar_splits()

        # Validar que no haya rutas rotas en los splits cargados
        verificar_existencia_archivos(train_df, "train")
        verificar_existencia_archivos(val_df, "val")
        verificar_existencia_archivos(test_df, "test")

        return train_df, val_df, test_df

    except (FileNotFoundError, OSError):
        if not create_if_missing:
            raise
        return crear_splits()


def get_tf_datasets(
    batch_size: int = BATCH_SIZE,
    cache: bool = False,
):
    train_df, val_df, test_df = get_splits()

    train_ds = crear_dataset_tf(
        train_df,
        batch_size=batch_size,
        entrenamiento=True,
        usar_cache=cache,
    )
    val_ds = crear_dataset_tf(
        val_df,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=cache,
    )
    test_ds = crear_dataset_tf(
        test_df,
        batch_size=batch_size,
        entrenamiento=False,
        usar_cache=cache,
    )

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def main():
    print("=== DEBUG DATA: INICIO ===")
    print(f"RUTA_COLON: {RUTA_COLON}")
    print(f"RUTA_LUNG: {RUTA_LUNG}")
    print(f"RUTA_SPLITS: {RUTA_SPLITS}")

    print("\n📦 Generando o cargando splits...")
    train_df, val_df, test_df = get_splits(create_if_missing=True)

    print("\n✅ Splits listos")
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print("\n🔎 Ejemplos de rutas:")
    print(f"Train ejemplo: {train_df['filepath'].iloc[0]}")
    print(f"Val ejemplo:   {val_df['filepath'].iloc[0]}")
    print(f"Test ejemplo:  {test_df['filepath'].iloc[0]}")

    print("=== DEBUG DATA: FIN ===")


if __name__ == "__main__":
    main()