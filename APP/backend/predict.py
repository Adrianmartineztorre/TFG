from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config_app import (
    CLASES,
    CLAVES_CLASES,
    DESV_IMAGEN,
    MAX_RESULTADOS,
    MEDIA_IMAGEN,
    MODELO_PATH,
    TAMANO_IMG,
    UMBRAL_PROBABILIDAD,
)

_modelo = None


def cargar_modelo() -> tf.keras.Model:
    """
    Carga el modelo una sola vez y lo reutiliza.
    """
    global _modelo

    if _modelo is None:
        if not MODELO_PATH.exists():
            raise FileNotFoundError(f"No existe el modelo en:\n{MODELO_PATH}")

        _modelo = tf.keras.models.load_model(MODELO_PATH)

    return _modelo


def cargar_y_preprocesar_imagen(ruta_imagen: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga una imagen desde disco y aplica el mismo preprocesado
    usado durante el entrenamiento.

    Devuelve:
    - img_visual: imagen uint8 para mostrar
    - img_batch: imagen preparada para el modelo con shape (1, H, W, C)
    """
    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize(TAMANO_IMG)

    img_visual = np.array(img).astype("uint8")

    img_modelo = img_visual.astype("float32") / 255.0
    media = np.array(MEDIA_IMAGEN, dtype=np.float32)
    desv = np.array(DESV_IMAGEN, dtype=np.float32)
    img_modelo = (img_modelo - media) / desv
    img_batch = np.expand_dims(img_modelo, axis=0)

    return img_visual, img_batch


def obtener_probabilidades(modelo: tf.keras.Model, img_batch: np.ndarray) -> np.ndarray:
    """
    Devuelve el vector de probabilidades del modelo.
    """
    probs = modelo.predict(img_batch, verbose=0)[0]
    return probs


def construir_resultados_probabilidades(probs: np.ndarray) -> dict:
    """
    Construye la distribución de porcentajes:
    - elimina valores menores del umbral
    - muestra como máximo N resultados
    - siempre conserva la predicción principal
    """
    idx_pred = int(np.argmax(probs))
    clave_pred = CLAVES_CLASES[idx_pred]
    nombre_pred = CLASES.get(clave_pred, clave_pred)
    porcentaje_pred = round(float(probs[idx_pred]) * 100, 2)

    distribucion = []

    for i, (clase, prob) in enumerate(zip(CLAVES_CLASES, probs)):
        porcentaje = round(float(prob) * 100, 2)

        if porcentaje >= UMBRAL_PROBABILIDAD or i == idx_pred:
            distribucion.append(
                {
                    "clave": clase,
                    "nombre": CLASES.get(clase, clase),
                    "porcentaje": porcentaje,
                }
            )

    distribucion.sort(key=lambda x: x["porcentaje"], reverse=True)
    distribucion = distribucion[:MAX_RESULTADOS]

    if not any(item["clave"] == clave_pred for item in distribucion):
        distribucion.append(
            {
                "clave": clave_pred,
                "nombre": nombre_pred,
                "porcentaje": porcentaje_pred,
            }
        )
        distribucion.sort(key=lambda x: x["porcentaje"], reverse=True)
        distribucion = distribucion[:MAX_RESULTADOS]

    return {
        "prediccion": {
            "clave": clave_pred,
            "nombre": nombre_pred,
            "porcentaje": porcentaje_pred,
            "indice": idx_pred,
        },
        "distribucion": distribucion,
    }


def predecir_imagen(ruta_imagen: str | Path) -> dict:
    """
    Función principal de inferencia.
    Devuelve un diccionario listo para usar en la app.
    """
    modelo = cargar_modelo()
    _, img_batch = cargar_y_preprocesar_imagen(ruta_imagen)
    probs = obtener_probabilidades(modelo, img_batch)
    resultados = construir_resultados_probabilidades(probs)

    return {
        "ruta_imagen": str(ruta_imagen),
        "prediccion": resultados["prediccion"]["nombre"],
        "prediccion_clave": resultados["prediccion"]["clave"],
        "prediccion_indice": resultados["prediccion"]["indice"],
        "porcentaje_prediccion": resultados["prediccion"]["porcentaje"],
        "distribucion": resultados["distribucion"],
    }


if __name__ == "__main__":
    print("=== PREDICCIÓN CNN_VGG_OPT ===")
    print(f"Modelo esperado en: {MODELO_PATH}")

    ruta = input("\nIntroduce la ruta completa de la imagen: ").strip().strip('"')

    if not ruta:
        raise ValueError("No se ha introducido ninguna ruta de imagen.")

    resultado = predecir_imagen(ruta)

    print("\n=== RESULTADO ===")
    print(f"Predicción final: {resultado['prediccion']}")
    print(f"Clase interna: {resultado['prediccion_clave']}")
    print(f"Índice clase: {resultado['prediccion_indice']}")
    print(f"Porcentaje principal: {resultado['porcentaje_prediccion']:.2f}%")

    print("\nDistribución:")
    for item in resultado["distribucion"]:
        print(f" - {item['nombre']}: {item['porcentaje']:.2f}%")