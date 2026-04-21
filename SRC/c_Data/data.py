"""
Módulo de gestión de datos para el pipeline de entrenamiento.

Responsabilidades:
- Construir el DataFrame a partir del dataset en disco.
- Calcular SHA256 para agrupar duplicados exactos.
- Generar splits train/val/test sin fuga entre duplicados exactos.
- Guardar y cargar los splits en CSV para reutilizarlos.

IMPORTANTE:
- La prevención del data leakage real se hace aquí, al crear los splits.
- Las imágenes idénticas (mismo SHA256) siempre se asignan al mismo split.
- Este módulo no hace preprocesado TensorFlow; eso vive en preprocess.py.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from a_Configuracion.config import (
    SEED,
    RUTA_COLON,
    RUTA_LUNG,
    RUTA_SPLITS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    CLAVES_CLASES,
)


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _sha256_archivo(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Calcula el hash SHA256 del contenido de un archivo.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalizar_ruta_entorno(path_str: str) -> str:
    """
    Normaliza la ruta para el entorno actual.
    """
    return str(Path(path_str).resolve())


def _buscar_imagenes_en_clase(base_dir: Path, clase: str) -> list[str]:
    """
    Busca imágenes válidas dentro de la carpeta de una clase.
    """
    class_dir = base_dir / clase
    if not class_dir.exists():
        return []

    rutas = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS:
            rutas.append(str(p.resolve()))
    return rutas


def construir_dataframe() -> pd.DataFrame:
    """
    Construye un DataFrame con filepath, label y sha256 a partir del dataset.
    """
    registros = []

    for clase in CLAVES_CLASES:
        if clase.startswith("colon_"):
            rutas = _buscar_imagenes_en_clase(RUTA_COLON, clase)
        elif clase.startswith("lung_"):
            rutas = _buscar_imagenes_en_clase(RUTA_LUNG, clase)
        else:
            rutas = []

        for ruta in rutas:
            registros.append(
                {
                    "filepath": _normalizar_ruta_entorno(ruta),
                    "label": clase,
                }
            )

    df = pd.DataFrame(registros)

    if df.empty:
        raise FileNotFoundError(
            "No se encontraron imágenes al construir el dataframe. "
            "Revisa las rutas del dataset en config.py."
        )

    print(f"✅ Imágenes encontradas: {len(df)}")

    print("🔍 Calculando SHA256 para agrupar duplicados exactos...")
    df["sha256"] = df["filepath"].apply(_sha256_archivo)

    return df


def _asignar_split_por_hash(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hace el split por grupos de SHA256, para que imágenes idénticas
    nunca queden repartidas entre train/val/test.
    """
    if "sha256" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'sha256'.")

    grupos = (
        df.groupby("sha256", as_index=False)
        .agg(
            label=("label", "first"),
            n_copias=("filepath", "count"),
        )
    )

    train_hashes, temp_hashes = train_test_split(
        grupos,
        test_size=(1.0 - TRAIN_SPLIT),
        random_state=SEED,
        stratify=grupos["label"],
    )

    proporcion_test_relativa = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    val_hashes, test_hashes = train_test_split(
        temp_hashes,
        test_size=proporcion_test_relativa,
        random_state=SEED,
        stratify=temp_hashes["label"],
    )

    train_sha = set(train_hashes["sha256"].tolist())
    val_sha = set(val_hashes["sha256"].tolist())
    test_sha = set(test_hashes["sha256"].tolist())

    train_df = df[df["sha256"].isin(train_sha)].copy()
    val_df = df[df["sha256"].isin(val_sha)].copy()
    test_df = df[df["sha256"].isin(test_sha)].copy()

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df


def _verificar_sin_fuga(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Verifica que no haya hashes compartidos entre splits.
    """
    train_hashes = set(train_df["sha256"].tolist())
    val_hashes = set(val_df["sha256"].tolist())
    test_hashes = set(test_df["sha256"].tolist())

    assert train_hashes.isdisjoint(val_hashes), "Hay hashes compartidos entre train y val"
    assert train_hashes.isdisjoint(test_hashes), "Hay hashes compartidos entre train y test"
    assert val_hashes.isdisjoint(test_hashes), "Hay hashes compartidos entre val y test"


def crear_y_guardar_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construye el dataframe completo y genera splits sin data leakage
    entre duplicados exactos.
    """
    RUTA_SPLITS.mkdir(parents=True, exist_ok=True)

    df = construir_dataframe()
    train_df, val_df, test_df = _asignar_split_por_hash(df)

    _verificar_sin_fuga(train_df, val_df, test_df)

    train_path = RUTA_SPLITS / "train.csv"
    val_path = RUTA_SPLITS / "val.csv"
    test_path = RUTA_SPLITS / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("\n✅ Splits generados sin fuga entre duplicados exactos")
    print(f"Train: {len(train_df)} imágenes | {train_df['sha256'].nunique()} hashes únicos")
    print(f"Val:   {len(val_df)} imágenes | {val_df['sha256'].nunique()} hashes únicos")
    print(f"Test:  {len(test_df)} imágenes | {test_df['sha256'].nunique()} hashes únicos")

    print(f"\n💾 Guardado en: {RUTA_SPLITS}")

    return train_df, val_df, test_df


def get_splits(
    create_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los splits desde disco. Si no existen, los genera.
    """
    train_path = RUTA_SPLITS / "train.csv"
    val_path = RUTA_SPLITS / "val.csv"
    test_path = RUTA_SPLITS / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        if not create_if_missing:
            raise FileNotFoundError("No existen los CSV de splits.")
        return crear_y_guardar_splits()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    _verificar_sin_fuga(train_df, val_df, test_df)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = crear_y_guardar_splits()

    print("\n=== RESUMEN ===")
    print(train_df.head())
    print(val_df.head())
    print(test_df.head())