# SRC/f_Evaluacion/tabla_dataset.py

import pandas as pd
import matplotlib.pyplot as plt

from c_Data.data import get_splits
from a_Configuracion.config import RUTA_OUTPUTS, CLAVES_CLASES


def detectar_columna_label(df):
    for col in ["label", "etiqueta", "clase", "target"]:
        if col in df.columns:
            return col
    raise ValueError("No se encontró columna de etiquetas")


def contar_por_clase(df, col_label):
    conteo = df[col_label].value_counts().to_dict()
    return {clase: conteo.get(clase, 0) for clase in CLAVES_CLASES}


def formatear_numero(n):
    return f"{n:,}".replace(",", ".")


def main():
    train_df, val_df, test_df = get_splits()

    col_label = detectar_columna_label(train_df)

    filas = []

    conteo_train = contar_por_clase(train_df, col_label)
    conteo_val = contar_por_clase(val_df, col_label)
    conteo_test = contar_por_clase(test_df, col_label)

    for clase in CLAVES_CLASES:
        train = conteo_train[clase]
        val = conteo_val[clase]
        test = conteo_test[clase]

        filas.append({
            "Clase": clase,
            "Train": formatear_numero(train),
            "Val": formatear_numero(val),
            "Test": formatear_numero(test),
            "Total": formatear_numero(train + val + test),
        })

    total_train = sum(conteo_train.values())
    total_val = sum(conteo_val.values())
    total_test = sum(conteo_test.values())
    total_global = total_train + total_val + total_test

    fila_total = {
        "Clase": "Total",
        "Train": (
            f"{formatear_numero(total_train)} "
            f"({total_train / total_global:.1%})"
        ),
        "Val": (
            f"{formatear_numero(total_val)} "
            f"({total_val / total_global:.1%})"
        ),
        "Test": (
            f"{formatear_numero(total_test)} "
            f"({total_test / total_global:.1%})"
        ),
        "Total": formatear_numero(total_global),
    }

    filas.append(fila_total)

    df = pd.DataFrame(filas)

    salida = RUTA_OUTPUTS / "dataset"
    salida.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.axis("off")

    tabla = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    num_filas = len(df)

    for (r, c), cell in tabla.get_celld().items():

        if r == 0:
            cell.set_text_props(weight="bold")

        if r == num_filas:
            cell.set_text_props(weight="bold")

    plt.title(
        "Distribución final de muestras por clase "
        "en los subconjuntos de entrenamiento, validación y prueba",
        fontsize=10,
        weight="bold",
        pad=10,
    )

    plt.tight_layout()

    ruta_salida = salida / "tabla_dataset.png"

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"✅ Tabla guardada en: {ruta_salida}")


if __name__ == "__main__":
    main()