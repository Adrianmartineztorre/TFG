"""
Grad-CAM + Guided Grad-CAM para el modelo cnn_vgg entrenado.

Este script:
- Carga el modelo guardado en OUTPUT/MODELOS/cnn_vgg/model.best.keras
- Pide al usuario la ruta de una imagen
- Predice la clase
- Genera:
    1) Grad-CAM clásico (heatmap superpuesto)
    2) Guided Grad-CAM
- Guarda:
    1) imagen original
    2) imagen con Grad-CAM
    3) imagen con Guided Grad-CAM
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from a_Configuracion.config_antiguo import (
    CLAVES_CLASES,
    DESV_IMAGEN,
    FIGURAS_DIR,
    MEDIA_IMAGEN,
    MODELOS_DIR,
    TAMANO_IMG,
)


MODELO_PATH = MODELOS_DIR / "cnn_vgg" / "model.best.keras"
SALIDA_DIR = FIGURAS_DIR / "cnn_vgg" / "gradcam"


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
    class_index: int,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=modelo.inputs,
        outputs=[
            modelo.get_layer(nombre_capa_conv).output,
            modelo.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_batch, training=False)
        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
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


def generar_saliency_positiva(
    modelo: tf.keras.Model,
    img_batch: np.ndarray,
    class_index: int,
) -> np.ndarray:
    entrada = tf.cast(img_batch, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(entrada)
        preds = modelo(entrada, training=False)
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, entrada)[0]
    grads = tf.maximum(grads, 0)
    grads = grads.numpy()

    grads_min = grads.min()
    grads_max = grads.max()

    if grads_max > grads_min:
        grads = (grads - grads_min) / (grads_max - grads_min)
    else:
        grads = np.zeros_like(grads, dtype=np.float32)

    return grads.astype(np.float32)


def redimensionar_heatmap_a_imagen(
    heatmap: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_img = heatmap_img.resize(target_size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img).astype(np.float32) / 255.0
    return heatmap_resized


def generar_guided_gradcam(
    img_original: np.ndarray,
    heatmap: np.ndarray,
    saliency: np.ndarray,
) -> np.ndarray:
    h, w = img_original.shape[:2]
    heatmap_resized = redimensionar_heatmap_a_imagen(heatmap, (w, h))
    heatmap_resized = np.expand_dims(heatmap_resized, axis=-1)

    guided = saliency * heatmap_resized

    guided_min = guided.min()
    guided_max = guided.max()

    if guided_max > guided_min:
        guided = (guided - guided_min) / (guided_max - guided_min)
    else:
        guided = np.zeros_like(guided, dtype=np.float32)

    guided_uint8 = np.uint8(guided * 255)
    return guided_uint8


def guardar_imagen(ruta_salida: Path, imagen: np.ndarray) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagen).save(ruta_salida)


def guardar_resultados(
    nombre_base: str,
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_guided: np.ndarray,
) -> tuple[Path, Path, Path]:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    ruta_original = SALIDA_DIR / f"{nombre_base}_original.png"
    ruta_gradcam = SALIDA_DIR / f"{nombre_base}_gradcam.png"
    ruta_guided = SALIDA_DIR / f"{nombre_base}_guided_gradcam.png"

    guardar_imagen(ruta_original, img_original)
    guardar_imagen(ruta_gradcam, img_gradcam)
    guardar_imagen(ruta_guided, img_guided)

    return ruta_original, ruta_gradcam, ruta_guided


def mostrar_resultados(
    img_original: np.ndarray,
    img_gradcam: np.ndarray,
    img_guided: np.ndarray,
    etiqueta_predicha: str,
    probs: np.ndarray,
) -> None:
    print("\n=== RESULTADO PREDICCIÓN ===")
    print(f"Clase predicha: {etiqueta_predicha}")

    print("\nProbabilidades por clase:")
    for clase, prob in zip(CLAVES_CLASES, probs):
        print(f"  - {clase}: {prob:.6f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_original)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_gradcam)
    plt.title(f"Grad-CAM\n{etiqueta_predicha}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_guided)
    plt.title(f"Guided Grad-CAM\n{etiqueta_predicha}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    print("=== GRAD-CAM + GUIDED GRAD-CAM CNN_VGG ===")
    print(f"Modelo esperado: {MODELO_PATH}")

    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo entrenado en:\n{MODELO_PATH}\n"
            "Asegúrate de haber entrenado cnn_vgg y guardado model.best.keras"
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

    print("🌡️ Generando heatmap superpuesto...")
    img_gradcam, _ = superponer_heatmap(img_original, heatmap)

    print("✨ Generando saliency positiva...")
    saliency = generar_saliency_positiva(
        modelo=modelo,
        img_batch=img_batch,
        class_index=class_idx,
    )

    print("🧩 Generando Guided Grad-CAM...")
    img_guided = generar_guided_gradcam(
        img_original=img_original,
        heatmap=heatmap,
        saliency=saliency,
    )

    nombre_base = Path(ruta_imagen).stem
    ruta_original, ruta_gradcam, ruta_guided = guardar_resultados(
        nombre_base=nombre_base,
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_guided=img_guided,
    )

    mostrar_resultados(
        img_original=img_original,
        img_gradcam=img_gradcam,
        img_guided=img_guided,
        etiqueta_predicha=etiqueta_predicha,
        probs=probs,
    )

    print("\n✅ Resultados guardados en:")
    print(f" - Original:          {ruta_original}")
    print(f" - Grad-CAM:          {ruta_gradcam}")
    print(f" - Guided Grad-CAM:   {ruta_guided}")


if __name__ == "__main__":
    main()