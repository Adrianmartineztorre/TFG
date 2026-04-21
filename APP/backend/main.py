from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_app import (
    ANATOMIA_DIR,
    IMAGENES_ENTRADA_DIR,
    MAPA_ANATOMIA,
)
from predict import predecir_imagen
from gradcam_app import generar_gradcam_app


app = FastAPI()

# =========================================================
# CORS (para frontend)
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SERVIR ARCHIVOS ESTÁTICOS
# =========================================================
app.mount("/anatomia", StaticFiles(directory=ANATOMIA_DIR), name="anatomia")
app.mount("/imagenes", StaticFiles(directory=IMAGENES_ENTRADA_DIR), name="imagenes")
app.mount("/resultados", StaticFiles(directory="resultados"), name="resultados")


# =========================================================
# MODELO DE PETICIÓN
# =========================================================
class ImagenRequest(BaseModel):
    nombre_imagen: str


# =========================================================
# ENDPOINT: LISTAR IMÁGENES (máx 5)
# =========================================================
@app.get("/imagenes")
def listar_imagenes():
    imagenes = [
        f.name
        for f in IMAGENES_ENTRADA_DIR.iterdir()
        if f.is_file()
    ]

    imagenes = sorted(imagenes)[:5]

    return {"imagenes": imagenes}


# =========================================================
# ENDPOINT: ANALIZAR IMAGEN
# =========================================================
@app.post("/analizar")
def analizar_imagen(req: ImagenRequest):
    ruta_imagen = IMAGENES_ENTRADA_DIR / req.nombre_imagen

    if not ruta_imagen.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    # 🔹 1. Predicción
    resultado_pred = predecir_imagen(ruta_imagen)

    # 🔹 2. Grad-CAM (usando índice de predicción)
    resultado_gradcam = generar_gradcam_app(
        ruta_imagen,
        class_index=resultado_pred["prediccion_indice"],
    )

    # 🔹 3. Imagen anatómica
    clave = resultado_pred["prediccion_clave"]
    nombre_anatomia = MAPA_ANATOMIA.get(clave, "base.png")

    return {
        "imagen": req.nombre_imagen,
        "prediccion": resultado_pred["prediccion"],
        "prediccion_clave": resultado_pred["prediccion_clave"],
        "porcentaje_prediccion": resultado_pred["porcentaje_prediccion"],
        "distribucion": resultado_pred["distribucion"],
        "imagen_anatomica": f"/anatomia/{nombre_anatomia}",
        "gradcam": {
            "original": f"/resultados/{Path(resultado_gradcam['rutas']['original']).name}",
            "gradcam": f"/resultados/{Path(resultado_gradcam['rutas']['gradcam']).name}",
            "bbox": f"/resultados/{Path(resultado_gradcam['rutas']['bbox']).name}",
            "contorno": f"/resultados/{Path(resultado_gradcam['rutas']['contorno']).name}",
            "panel": f"/resultados/{Path(resultado_gradcam['rutas']['panel']).name}",
        },
    }