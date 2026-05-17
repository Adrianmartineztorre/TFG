"""
Grad-CAM para el modelo cnn_vgg_optimizado entrenado.

Este script:
- Carga el modelo guardado en OUTPUT/MODELOS/cnn_vgg_optimizado/model.best.keras
- Pide al usuario la ruta de una imagen
- Predice la clase
- Genera Grad-CAM usando la última capa convolucional
- Detecta la región más activada del heatmap
- Guarda:
    1) imagen original
    2) heatmap superpuesto
    3) imagen con bounding box
    4) imagen con contorno
    5) figura combinada académica con las 4 imágenes
    6) txt con resultados
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from a_Configuracion.config import (
    CLAVES_CLASES,
    DESV_IMAGEN,
    MEDIA_IMAGEN,
    MODELOS_DIR,
    RUTA_OUTPUTS,
    TAMANO_IMG,
)


MODELO_PATH = MODELOS_DIR / "cnn_vgg_optimizado" / "model.best.keras"
SALIDA_DIR = RUTA_OUTPUTS / "gradcam"


def cargar_y_preprocesar_imagen(ruta_imagen: str) -> tuple[np.ndarray, np.ndarray]:
    ruta = Path(ruta_imagen)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    img = Image.open(ruta).convert("RGB")
    img = img.resize(TAMANO_IMG)

    img_visual = np.array(img).astype("uint8")

    img_modelo = img_visual.astype("float32") / 255.0
    media = np.array(MEDIA_IMAGEN, dtype=np.float32)
    desv = np.array(DESV_IMAGEN, dtype=np.float32)

    img_modelo = (img_modelo - media) / desv
    img_modelo = np.expand_dims(img_modelo, axis=0)

    return img_visual, img_modelo


def encontrar_ultima_capa_conv(modelo: tf.keras.Model) -> str:
    for capa in reversed(modelo.layers):
        if isinstance(capa, tf.keras.layers.Conv2D):
            return capa.name

    raise ValueError("No se encontró ninguna capa Conv2D en el modelo.")


def predecir_clase(
    modelo: tf.keras.Model,
    img_batch: np.ndarray,
) -> tuple[int, str, np.ndarray]:
    probs = modelo.predict(img_batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    etiqueta = CLAVES_CLASES[idx]

    return idx, etiqueta, probs


def generar_gradcam(
    modelo: tf.keras.Model,
    img_batch: np.ndarray,
    nombre_capa_conv: str,
    class_index: int | None = None,
) -> np.ndarray:
    img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    entradas = tf.keras.Input(shape=img_batch.shape[1:])
    x = entradas
    salida_conv = None

    for capa in modelo.layers:
        x = capa(x)
        if capa.name == nombre_capa_conv:
            salida_conv = x

    if salida_conv is None:
        raise ValueError(
            f"No se pudo localizar la salida de la capa convolucional '{nombre_capa_conv}'."
        )

    grad_model = tf.keras.models.Model(
        inputs=entradas,
        outputs=[salida_conv, x],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        if class_index is None:
            class_index = int(tf.argmax(preds[0]))

        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError(
            f"No se pudieron calcular gradientes para Grad-CAM usando la capa "
            f"'{nombre_capa_conv}'."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def superponer_heatmap(
    img_original: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> tuple[np.ndarray, np.ndarray]:
    heatmap_resized = cv2.resize(
        heatmap,
        (img_original.shape[1], img_original.shape[0]),
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superpuesta = np.clip(
        (1 - alpha) * img_original.astype("float32")
        + alpha * heatmap_color.astype("float32"),
        0,
        255,
    ).astype("uint8")

    return superpuesta, heatmap_resized


def extraer_region_principal(
    heatmap_resized: np.ndarray,
    threshold: float = 0.60,
    min_area: int = 80,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray | None]:
    mask = (heatmap_resized >= threshold).astype("uint8") * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask, None, None

    mejor_idx = None
    mejor_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area and area > mejor_area:
            mejor_area = area
            mejor_idx = i

    if mejor_idx is None:
        return mask, None, None

    mask_principal = np.where(labels == mejor_idx, 255, 0).astype("uint8")

    contours, _ = cv2.findContours(
        mask_principal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return mask_principal, None, None

    contorno_principal = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_principal)

    return mask_principal, (x, y, w, h), contorno_principal


def dibujar_bounding_box(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    salida = img.copy()

    if bbox is None:
        return salida

    x, y, w, h = bbox

    cv2.rectangle(
        salida,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        5,
    )

    return salida


def dibujar_contorno(
    img: np.ndarray,
    contorno: np.ndarray | None,
) -> np.ndarray:
    salida = img.copy()

    if contorno is None:
        return salida

    cv2.drawContours(
        salida,
        [contorno],
        contourIdx=-1,
        color=(255, 0, 0),
        thickness=5,
    )

    return salida


def guardar_imagen(ruta_salida: Path, imagen: np.ndarray) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagen).save(ruta_salida)


def guardar_resultados(
    nombre_base: str,
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_bbox: np.ndarray,
    img_contorno: np.ndarray,
) -> tuple[Path, Path, Path, Path, Path]:
    carpeta = SALIDA_DIR / nombre_base
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_original = carpeta / f"{nombre_base}_original.png"
    ruta_gradcam = carpeta / f"{nombre_base}_gradcam.png"
    ruta_bbox = carpeta / f"{nombre_base}_gradcam_bounding_box.png"
    ruta_contorno = carpeta / f"{nombre_base}_gradcam_contorno.png"

    guardar_imagen(ruta_original, img_original)
    guardar_imagen(ruta_gradcam, img_gradcam)
    guardar_imagen(ruta_bbox, img_bbox)
    guardar_imagen(ruta_contorno, img_contorno)

    return ruta_original, ruta_gradcam, ruta_bbox, ruta_contorno, carpeta


def guardar_txt_resultados(
    carpeta: Path,
    nombre_base: str,
    etiqueta_predicha: str,
    probs: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> Path:
    ruta_txt = carpeta / f"{nombre_base}_resultados.txt"

    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("=== RESULTADO PREDICCIÓN ===\n")
        f.write(f"Clase predicha: {etiqueta_predicha}\n")

        if bbox is not None:
            x, y, w, h = bbox
            f.write(f"Bounding box región activa: x={x}, y={y}, w={w}, h={h}\n")
        else:
            f.write(
                "No se detectó una región principal suficientemente grande "
                "con el threshold actual.\n"
            )

        f.write("\nProbabilidades por clase:\n")
        for clase, prob in zip(CLAVES_CLASES, probs):
            f.write(f"  - {clase}: {prob:.6f}\n")

    return ruta_txt


def _formatear_probabilidades_panel(
    etiqueta_predicha: str,
    probs: np.ndarray,
) -> str:
    top_idxs = np.argsort(probs)[::-1][:2]
    top_probs = [(CLAVES_CLASES[i], float(probs[i])) for i in top_idxs]

    texto = (
        r"$\bf{Clase\ predicha:}$ "
        f"{etiqueta_predicha} | "
        r"$\bf{Confianza\ principal:}$ "
        f"{top_probs[0][0]} = {top_probs[0][1]:.4f} | "
        f"2ª probabilidad: {top_probs[1][0]} = {top_probs[1][1]:.4f}"
    )

    return texto


def guardar_panel_4_imagenes(
    carpeta: Path,
    nombre_base: str,
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_bbox: np.ndarray,
    img_contorno: np.ndarray,
    etiqueta_predicha: str,
    probs: np.ndarray,
) -> Path:
    ruta_panel = carpeta / f"{nombre_base}_panel_4imagenes.png"

    texto_inferior = _formatear_probabilidades_panel(
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
    )

    fig, axes = plt.subplots(1, 4, figsize=(28, 10.5))

    fig.suptitle(
        "Visualización Grad-CAM de la predicción del modelo",
        fontsize=32,
        fontweight="bold",
        y=0.985,
    )

    imagenes = [
        img_original,
        img_gradcam,
        img_bbox,
        img_contorno,
    ]

    titulos = [
        "Imagen original",
        "Mapa Grad-CAM",
        "Región activa (Bounding box)",
        "Región activa (Contorno)",
    ]

    for ax, imagen, titulo in zip(axes, imagenes, titulos):
        ax.imshow(imagen)
        ax.set_title(
            titulo,
            fontsize=24,
            pad=16,
            fontweight="bold",
        )
        ax.axis("off")

    fig.text(
        0.5,
        0.045,
        texto_inferior,
        ha="center",
        va="center",
        fontsize=30,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.91])

    fig.savefig(
        ruta_panel,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return ruta_panel


def mostrar_resultados(
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_bbox: np.ndarray,
    img_contorno: np.ndarray,
    etiqueta_predicha: str,
    probs: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> None:
    print("\n=== RESULTADO PREDICCIÓN ===")
    print(f"Clase predicha: {etiqueta_predicha}")

    if bbox is not None:
        x, y, w, h = bbox
        print(f"Bounding box región activa: x={x}, y={y}, w={w}, h={h}")
    else:
        print("No se detectó una región principal suficientemente grande con el threshold actual.")

    print("\nProbabilidades por clase:")
    for clase, prob in zip(CLAVES_CLASES, probs):
        print(f"  - {clase}: {prob:.6f}")

    texto_inferior = _formatear_probabilidades_panel(
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 8.2))

    fig.suptitle(
        "Visualización Grad-CAM de la predicción del modelo",
        fontsize=26,
        fontweight="bold",
        y=0.98,
    )

    imagenes = [
        img_original,
        img_gradcam,
        img_bbox,
        img_contorno,
    ]

    titulos = [
        "Imagen original",
        "Mapa Grad-CAM",
        "Región activa (Bounding box)",
        "Región activa (Contorno)",
    ]

    for ax, imagen, titulo in zip(axes, imagenes, titulos):
        ax.imshow(imagen)
        ax.set_title(
            titulo,
            fontsize=22,
            pad=16,
            fontweight="bold",
        )
        ax.axis("off")

    fig.text(
        0.5,
        0.045,
        texto_inferior,
        ha="center",
        va="center",
        fontsize=20,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.91])
    plt.show()


def main() -> None:
    print("=== GRAD-CAM CNN_VGG_OPTIMIZADO ===")
    print(f"Modelo esperado: {MODELO_PATH}")

    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo entrenado en:\n{MODELO_PATH}\n"
            "Asegúrate de haber entrenado cnn_vgg_optimizado y guardado model.best.keras"
        )

    ruta_imagen = input("\nIntroduce la ruta completa de la imagen: ").strip().strip('"')

    if not ruta_imagen:
        raise ValueError("No se ha introducido ninguna ruta de imagen.")

    print("\n📦 Cargando modelo...")
    modelo = tf.keras.models.load_model(MODELO_PATH)

    print("🧠 Buscando última capa convolucional...")
    nombre_capa_conv = encontrar_ultima_capa_conv(modelo)
    print(f"Última capa convolucional detectada: {nombre_capa_conv}")

    print("🖼️ Cargando imagen...")
    img_original, img_batch = cargar_y_preprocesar_imagen(ruta_imagen)

    print("🔎 Realizando predicción...")
    class_idx, etiqueta_predicha, probs = predecir_clase(modelo, img_batch)

    print("🔥 Generando Grad-CAM...")
    heatmap = generar_gradcam(
        modelo=modelo,
        img_batch=img_batch,
        nombre_capa_conv=nombre_capa_conv,
        class_index=class_idx,
    )

    img_gradcam, heatmap_resized = superponer_heatmap(img_original, heatmap)

    _, bbox, contorno = extraer_region_principal(
        heatmap_resized=heatmap_resized,
        threshold=0.60,
        min_area=80,
    )

    img_bbox = dibujar_bounding_box(img_gradcam, bbox)
    img_contorno = dibujar_contorno(img_gradcam, contorno)

    nombre_base = Path(ruta_imagen).stem

    ruta_original, ruta_gradcam, ruta_bbox, ruta_contorno, carpeta = guardar_resultados(
        nombre_base=nombre_base,
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_bbox=img_bbox,
        img_contorno=img_contorno,
    )

    ruta_txt = guardar_txt_resultados(
        carpeta=carpeta,
        nombre_base=nombre_base,
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
        bbox=bbox,
    )

    ruta_panel = guardar_panel_4_imagenes(
        carpeta=carpeta,
        nombre_base=nombre_base,
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_bbox=img_bbox,
        img_contorno=img_contorno,
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
    )

    mostrar_resultados(
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_bbox=img_bbox,
        img_contorno=img_contorno,
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
        bbox=bbox,
    )

    print("\n✅ Resultados guardados en:")
    print(f" - Original:      {ruta_original}")
    print(f" - Grad-CAM:      {ruta_gradcam}")
    print(f" - Bounding box:  {ruta_bbox}")
    print(f" - Contorno:      {ruta_contorno}")
    print(f" - Panel:         {ruta_panel}")
    print(f" - TXT:           {ruta_txt}")


if __name__ == "__main__":
    main()