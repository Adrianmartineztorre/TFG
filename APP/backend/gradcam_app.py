from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from config_app import (
    CLASES,
    CLAVES_CLASES,
    GRADCAM_ALPHA,
    GRADCAM_DIR,
    GRADCAM_MIN_AREA,
    GRADCAM_THRESHOLD,
)
from predict import cargar_modelo, cargar_y_preprocesar_imagen, obtener_probabilidades


def encontrar_ultima_capa_conv(modelo: tf.keras.Model) -> str:
    """
    Busca automáticamente la última capa convolucional del modelo.
    """
    for capa in reversed(modelo.layers):
        if isinstance(capa, tf.keras.layers.Conv2D):
            return capa.name
    raise ValueError("No se encontró ninguna capa Conv2D en el modelo.")


def generar_heatmap_gradcam(
    modelo: tf.keras.Model,
    img_batch: np.ndarray,
    nombre_capa_conv: str,
    class_index: int,
) -> np.ndarray:
    """
    Genera el heatmap Grad-CAM para una clase concreta.
    """
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
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError(
            f"No se pudieron calcular gradientes para Grad-CAM con la capa '{nombre_capa_conv}'."
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
    alpha: float = GRADCAM_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Superpone el heatmap sobre la imagen original.
    """
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
    threshold: float = GRADCAM_THRESHOLD,
    min_area: int = GRADCAM_MIN_AREA,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, np.ndarray | None]:
    """
    Extrae la región principal de activación del heatmap.
    """
    mask = (heatmap_resized >= threshold).astype("uint8") * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

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
    contours, _ = cv2.findContours(mask_principal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask_principal, None, None

    contorno_principal = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contorno_principal)

    return mask_principal, (x, y, w, h), contorno_principal


def dibujar_bounding_box(
    img: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """
    Dibuja la bounding box sobre la imagen.
    """
    salida = img.copy()

    if bbox is None:
        return salida

    x, y, w, h = bbox
    cv2.rectangle(salida, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return salida


def dibujar_contorno(
    img: np.ndarray,
    contorno: np.ndarray | None,
) -> np.ndarray:
    """
    Dibuja el contorno principal sobre la imagen.
    """
    salida = img.copy()

    if contorno is None:
        return salida

    cv2.drawContours(salida, [contorno], contourIdx=-1, color=(255, 0, 0), thickness=3)
    return salida


def guardar_imagen(ruta_salida: Path, imagen: np.ndarray) -> None:
    """
    Guarda una imagen en disco.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagen).save(ruta_salida)


def guardar_panel_4_imagenes(
    carpeta: Path,
    nombre_base: str,
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_bbox: np.ndarray,
    img_contorno: np.ndarray,
    etiqueta_predicha: str,
) -> Path:
    """
    Guarda un panel con las 4 visualizaciones.
    """
    ruta_panel = carpeta / f"{nombre_base}_panel_4imagenes.png"

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(img_original)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(img_gradcam)
    plt.title(f"Grad-CAM\n{etiqueta_predicha}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(img_bbox)
    plt.title("Región activa (BBox)")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_contorno)
    plt.title("Región activa (Contorno)")
    plt.axis("off")

    plt.tight_layout()
    fig.savefig(ruta_panel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return ruta_panel


def generar_gradcam_app(
    ruta_imagen: str | Path,
    class_index: int | None = None,
) -> dict:
    """
    Genera Grad-CAM para una imagen de la app.

    Si class_index no se indica, usa la clase predicha por el modelo.
    Si se indica, fuerza Grad-CAM sobre esa clase.
    """
    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    modelo = cargar_modelo()
    nombre_capa_conv = encontrar_ultima_capa_conv(modelo)

    img_original, img_batch = cargar_y_preprocesar_imagen(ruta_imagen)
    probs = obtener_probabilidades(modelo, img_batch)

    if class_index is None:
        class_index = int(np.argmax(probs))

    etiqueta_predicha = CLAVES_CLASES[class_index]
    etiqueta_bonita = CLASES.get(etiqueta_predicha, etiqueta_predicha)

    heatmap = generar_heatmap_gradcam(
        modelo=modelo,
        img_batch=img_batch,
        nombre_capa_conv=nombre_capa_conv,
        class_index=class_index,
    )

    img_gradcam, heatmap_resized = superponer_heatmap(img_original, heatmap)

    _, bbox, contorno = extraer_region_principal(
        heatmap_resized=heatmap_resized,
        threshold=GRADCAM_THRESHOLD,
        min_area=GRADCAM_MIN_AREA,
    )

    img_bbox = dibujar_bounding_box(img_gradcam, bbox)
    img_contorno = dibujar_contorno(img_gradcam, contorno)

    nombre_base = ruta_imagen.stem
    carpeta = GRADCAM_DIR / nombre_base
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_original = carpeta / f"{nombre_base}_original.png"
    ruta_gradcam = carpeta / f"{nombre_base}_gradcam.png"
    ruta_bbox = carpeta / f"{nombre_base}_gradcam_bbox.png"
    ruta_contorno = carpeta / f"{nombre_base}_gradcam_contorno.png"

    guardar_imagen(ruta_original, img_original)
    guardar_imagen(ruta_gradcam, img_gradcam)
    guardar_imagen(ruta_bbox, img_bbox)
    guardar_imagen(ruta_contorno, img_contorno)

    ruta_panel = guardar_panel_4_imagenes(
        carpeta=carpeta,
        nombre_base=nombre_base,
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_bbox=img_bbox,
        img_contorno=img_contorno,
        etiqueta_predicha=etiqueta_bonita,
    )

    return {
        "ruta_imagen": str(ruta_imagen),
        "clase_gradcam": etiqueta_predicha,
        "clase_gradcam_nombre": etiqueta_bonita,
        "indice_clase": class_index,
        "nombre_capa_conv": nombre_capa_conv,
        "bbox": bbox,
        "rutas": {
            "original": str(ruta_original),
            "gradcam": str(ruta_gradcam),
            "bbox": str(ruta_bbox),
            "contorno": str(ruta_contorno),
            "panel": str(ruta_panel),
        },
    }


if __name__ == "__main__":
    print("=== GRAD-CAM APP ===")

    ruta = input("\nIntroduce la ruta completa de la imagen: ").strip().strip('"')

    if not ruta:
        raise ValueError("No se ha introducido ninguna ruta de imagen.")

    resultado = generar_gradcam_app(ruta)

    print("\n=== RESULTADO GRAD-CAM ===")
    print(f"Clase usada para Grad-CAM: {resultado['clase_gradcam_nombre']}")
    print(f"Clave interna: {resultado['clase_gradcam']}")
    print(f"Índice clase: {resultado['indice_clase']}")
    print(f"Última capa convolucional: {resultado['nombre_capa_conv']}")
    print(f"Bounding box: {resultado['bbox']}")

    print("\nArchivos generados:")
    for clave, ruta_salida in resultado["rutas"].items():
        print(f" - {clave}: {ruta_salida}")