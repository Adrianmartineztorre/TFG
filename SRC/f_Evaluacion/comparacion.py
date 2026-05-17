#comparacion.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from a_Configuracion.config import RUTA_OUTPUTS

RUNS_DIR = RUTA_OUTPUTS / "RUNS"
METRICAS_DIR = RUTA_OUTPUTS / "METRICAS"

MODELOS = [
    "baseline_cnn",
    "cnn_bn",
    "cnn_bn_dropout",
    "cnn_vgg_optimizado",
]

NOMBRES = {
    "baseline_cnn": "CNN base",
    "cnn_bn": "CNN + Batch Normalization",
    "cnn_bn_dropout": "CNN + Batch Normalization + Dropout",
    "cnn_vgg_optimizado": "VGG optimizado",
}

COLOR_PALETTE = {
    "baseline_cnn": "#4C72B0",
    "cnn_bn": "#DD8452",
    "cnn_bn_dropout": "#55A868",
    "cnn_vgg_optimizado": "#C44E52",
}

VGG_KEY = "cnn_vgg_optimizado"


def cargar_json(ruta):
    if not ruta.exists():
        print(f"⚠️  No existe: {ruta}")
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def obtener(history, claves):
    for clave in claves:
        if clave in history:
            return history[clave]
    return None


def obtener_valor_en_epoch(serie, idx):
    if serie is None or idx >= len(serie):
        return None
    return serie[idx]


def cargar_metricas_test(modelo):
    ruta = METRICAS_DIR / modelo / f"metricas_{modelo}_test.json"
    data = cargar_json(ruta)

    if data is None:
        return {"precision": None, "recall": None, "f1": None, "accuracy": None}

    report = data.get("classification_report", {})
    macro = report.get("macro avg", {})
    accuracy = report.get("accuracy", data.get("accuracy", None))

    return {
        "accuracy": accuracy,
        "precision": macro.get("precision", None),
        "recall": macro.get("recall", None),
        "f1": macro.get("f1-score", None),
    }


def procesar_modelo(modelo):
    carpeta = RUNS_DIR / modelo
    history = cargar_json(carpeta / "history.json")
    best_run = cargar_json(carpeta / "best_run.json")

    if history is None:
        return None

    loss = obtener(history, ["loss"])
    accuracy = obtener(history, ["accuracy", "acc"])
    val_loss = obtener(history, ["val_loss"])
    val_accuracy = obtener(history, ["val_accuracy", "val_acc"])
    learning_rate = obtener(history, ["learning_rate", "lr"])

    if val_loss is None or val_accuracy is None:
        return None

    epochs = len(val_loss)

    if best_run and "best_epoch" in best_run:
        mejor_idx = int(best_run["best_epoch"]) - 1
    elif best_run and "best_epoch_val_loss" in best_run:
        mejor_idx = int(best_run["best_epoch_val_loss"]) - 1
    else:
        mejor_idx = val_loss.index(min(val_loss))

    lr_final = learning_rate[-1] if (learning_rate and len(learning_rate) > 0) else None

    return {
        "nombre": NOMBRES[modelo],
        "modelo_key": modelo,
        "epochs": epochs,
        "mejor_epoch": mejor_idx + 1,
        "mejor_idx": mejor_idx,
        "loss": obtener_valor_en_epoch(loss, mejor_idx),
        "accuracy": obtener_valor_en_epoch(accuracy, mejor_idx),
        "val_loss": obtener_valor_en_epoch(val_loss, mejor_idx),
        "val_accuracy": obtener_valor_en_epoch(val_accuracy, mejor_idx),
        "lr_final": lr_final,
        "val_loss_hist": val_loss,
        "val_acc_hist": val_accuracy,
    }


def f3(x):
    return "N/A" if (x is None or pd.isnull(x)) else f"{float(x):.3f}"


def f4(x):
    return "N/A" if (x is None or pd.isnull(x)) else f"{float(x):.4f}"


def f_lr(x):
    return "N/A" if (x is None or pd.isnull(x)) else f"{float(x):.0e}"


def aplicar_estilo_academico(tabla, col_widths, fontsize_body=18, fontsize_header=18):
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(fontsize_body)

    # Celdas más altas para texto grande
    tabla.scale(1, 4.35)

    for (r, c), cell in tabla.get_celld().items():
        cell.set_linewidth(0.55)
        cell.set_edgecolor("#444444")
        cell.set_facecolor("white")
        cell.PAD = 0.34

        if r == 0:
            cell.set_text_props(
                weight="bold",
                fontsize=fontsize_header,
                ha="center",
                va="center",
                fontfamily="sans-serif",
            )
        else:
            cell.set_text_props(
                weight="normal",
                fontsize=fontsize_body,
                ha="center",
                va="center",
                fontfamily="sans-serif",
            )

        if c < len(col_widths):
            cell.set_width(col_widths[c])


def graficar_curvas(
    datos,
    hist_key,
    ylabel,
    titulo,
    nombre_archivo,
    salida,
    ylim=None,
    anotaciones_abajo=False,
):
    fig, ax = plt.subplots(figsize=(15, 8.4))

    for d in datos:
        es_vgg = d["modelo_key"] == VGG_KEY
        hist = d[hist_key]
        x = list(range(1, len(hist) + 1))
        color = COLOR_PALETTE.get(d["modelo_key"], None)

        ax.plot(
            x,
            hist,
            label=d["nombre"],
            color=color,
            linewidth=4.0 if es_vgg else 2.1,
            alpha=1.0 if es_vgg else 0.58,
            zorder=3 if es_vgg else 2,
        )

        i = d["mejor_idx"]
        mejor_x = i + 1
        mejor_y = hist[i]

        ax.scatter(
            mejor_x,
            mejor_y,
            color=color,
            s=155 if es_vgg else 115,
            zorder=6,
            alpha=1.0,
            edgecolor="white",
            linewidth=1.6,
        )

        if anotaciones_abajo:
            xytext = (0, -44 if es_vgg else -40)
            va = "top"
        else:
            xytext = (0, 32 if es_vgg else 28)
            va = "bottom"

        ax.annotate(
            f"e{mejor_x}\n{mejor_y:.3f}",
            xy=(mejor_x, mejor_y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=17 if es_vgg else 16,
            weight="bold" if es_vgg else "normal",
            color=color,
            ha="center",
            va=va,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                alpha=0.95,
            ),
            zorder=8,
        )

    ax.set_xlabel(
        "Epoch",
        fontsize=19,
        fontfamily="sans-serif",
        labelpad=22,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=19,
        fontfamily="sans-serif",
        labelpad=26,
    )

    ax.set_title(
        titulo,
        fontsize=21,
        weight="bold",
        fontfamily="sans-serif",
        pad=34,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.34),
    ncol=4,
    fontsize=12,
    frameon=True,
    framealpha=0.90,
    edgecolor="lightgrey",
    borderpad=0.8,
    labelspacing=0.8,
    handlelength=2.8,
    columnspacing=1.6,
    )

    ax.grid(True, alpha=0.10, linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.margins(x=0.06, y=0.16)

    plt.tight_layout()
    plt.savefig(
        salida / nombre_archivo,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def main():
    salida = RUTA_OUTPUTS / "comparacion"
    salida.mkdir(parents=True, exist_ok=True)

    datos = [procesar_modelo(m) for m in MODELOS]
    datos = [d for d in datos if d]

    filas_principal = []

    for d in datos:
        mt = cargar_metricas_test(d["modelo_key"])
        filas_principal.append({
            "Arquitectura": d["nombre"],
            "Accuracy": f4(mt["accuracy"]),
            "Precision (Macro)": f4(mt["precision"]),
            "Recall (Macro)": f4(mt["recall"]),
            "F1-score (Macro)": f4(mt["f1"]),
        })

    df_principal = pd.DataFrame(filas_principal)

    fig, ax = plt.subplots(figsize=(18.2, 6.8))
    ax.axis("off")

    tabla_p = ax.table(
        cellText=df_principal.values,
        colLabels=df_principal.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    aplicar_estilo_academico(
        tabla_p,
        col_widths=[0.32, 0.16, 0.20, 0.18, 0.18],
        fontsize_body=18,
        fontsize_header=18,
    )

    ax.set_title(
        "Comparativa de rendimiento de las arquitecturas CNN evaluadas",
        fontsize=23,
        weight="bold",
        pad=42,
        fontfamily="sans-serif",
        loc="center",
    )

    plt.tight_layout()
    plt.savefig(
        salida / "tabla_comparativa_principal.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    filas_epoch = []

    for d in datos:
        filas_epoch.append({
            "Arquitectura": d["nombre"],
            "Mejor epoch": str(d["mejor_epoch"]),
            "Accuracy": f3(d["accuracy"]),
            "Val. Accuracy": f3(d["val_accuracy"]),
            "Loss": f3(d["loss"]),
            "Val. Loss": f3(d["val_loss"]),
            "LR final": f_lr(d["lr_final"]),
        })

    df_epoch = pd.DataFrame(filas_epoch)

    fig, ax = plt.subplots(figsize=(20.2, 6.8))
    ax.axis("off")

    tabla_e = ax.table(
        cellText=df_epoch.values,
        colLabels=df_epoch.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    aplicar_estilo_academico(
        tabla_e,
        col_widths=[0.26, 0.14, 0.12, 0.16, 0.12, 0.14, 0.13],
        fontsize_body=18,
        fontsize_header=18,
    )

    ax.set_title(
        "Resumen del entrenamiento en el epoch óptimo",
        fontsize=23,
        weight="bold",
        pad=42,
        fontfamily="sans-serif",
        loc="center",
    )

    plt.tight_layout()
    plt.savefig(
        salida / "tabla_mejor_epoch.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    graficar_curvas(
        datos,
        hist_key="val_acc_hist",
        ylabel="Val Accuracy",
        titulo="Evolución de la accuracy de validación durante el entrenamiento",
        nombre_archivo="val_accuracy_curvas.png",
        salida=salida,
        anotaciones_abajo=True,
    )

    graficar_curvas(
        datos,
        hist_key="val_loss_hist",
        ylabel="Val Loss",
        titulo="Evolución de la pérdida de validación durante el entrenamiento",
        nombre_archivo="val_loss_curvas.png",
        salida=salida,
        ylim=(0, 1.45),
        anotaciones_abajo=False,
    )

    print("✅ Todo correcto")


if __name__ == "__main__":
    main()