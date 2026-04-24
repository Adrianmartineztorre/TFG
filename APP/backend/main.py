from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config_app import (
    ARCHIVOS_EXCLUIDOS,
    ASSETS_DIR,
    EXTENSIONES_VALIDAS,
    FRONTEND_DIR,
    IMAGENES_ENTRADA_DIR,
    MODELOS_RNA_DIR,
    RESULTADOS_DIR,
)
from predict import predecir_imagen
from gradcam_app import generar_gradcam_app
from rna_app import obtener_perfil_rna


app = FastAPI()

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ARCHIVOS ESTÁTICOS
# =========================================================
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount(
    "/static/imagenes_entrada",
    StaticFiles(directory=IMAGENES_ENTRADA_DIR),
    name="imagenes_entrada",
)
app.mount("/static/resultados", StaticFiles(directory=RESULTADOS_DIR), name="resultados")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount(
    "/static/interpretabilidad",
    StaticFiles(directory=MODELOS_RNA_DIR),
    name="interpretabilidad",
)


# =========================================================
# MODELO DE PETICIÓN
# =========================================================
class ImagenRequest(BaseModel):
    nombre_imagen: str


# =========================================================
# HELPERS
# =========================================================
def obtener_imagenes_validas() -> list[str]:
    imagenes = [
        f.name
        for f in IMAGENES_ENTRADA_DIR.iterdir()
        if f.is_file()
        and f.suffix.lower() in EXTENSIONES_VALIDAS
        and f.name not in ARCHIVOS_EXCLUIDOS
    ]
    return sorted(imagenes)[:5]


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def inicio():
    ruta_index = FRONTEND_DIR / "index.html"
    if not ruta_index.exists():
        raise HTTPException(status_code=404, detail="No se encontró frontend/index.html")
    return FileResponse(ruta_index)


# =========================================================
# FAVICON
# =========================================================
@app.get("/favicon.ico")
def favicon():
    ruta_favicon = ASSETS_DIR / "flavicon.png"
    if not ruta_favicon.exists():
        raise HTTPException(status_code=404, detail="No se encontró assets/flavicon.png")
    return FileResponse(ruta_favicon)


# =========================================================
# ENDPOINT: LISTAR IMÁGENES
# =========================================================
@app.get("/imagenes_disponibles")
def listar_imagenes():
    imagenes = obtener_imagenes_validas()
    return {
        "carpeta": str(IMAGENES_ENTRADA_DIR),
        "total": len(imagenes),
        "imagenes": imagenes,
    }


# =========================================================
# ENDPOINT: ANALIZAR IMAGEN
# =========================================================
@app.post("/analizar")
def analizar_imagen(req: ImagenRequest):
    ruta_imagen = IMAGENES_ENTRADA_DIR / req.nombre_imagen

    if not ruta_imagen.exists():
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {req.nombre_imagen}")

    if req.nombre_imagen in ARCHIVOS_EXCLUIDOS:
        raise HTTPException(status_code=400, detail="Ese archivo no es una muestra válida")

    if ruta_imagen.suffix.lower() not in EXTENSIONES_VALIDAS:
        raise HTTPException(status_code=400, detail="Formato de imagen no permitido")

    resultado_pred = predecir_imagen(ruta_imagen)

    resultado_gradcam = generar_gradcam_app(
        ruta_imagen,
        class_index=resultado_pred["prediccion_indice"],
    )

    nombre_base = ruta_imagen.stem

    return {
        "imagen": req.nombre_imagen,
        "ruta_imagen": f"/static/imagenes_entrada/{req.nombre_imagen}",
        "prediccion": resultado_pred["prediccion"],
        "prediccion_clave": resultado_pred["prediccion_clave"],
        "porcentaje_prediccion": resultado_pred["porcentaje_prediccion"],
        "distribucion": resultado_pred["distribucion"],
        "gradcam": {
            "original": f"/static/resultados/gradcam/{nombre_base}/{nombre_base}_original.png",
            "gradcam": f"/static/resultados/gradcam/{nombre_base}/{nombre_base}_gradcam.png",
            "bbox": f"/static/resultados/gradcam/{nombre_base}/{nombre_base}_gradcam_bbox.png",
            "contorno": f"/static/resultados/gradcam/{nombre_base}/{nombre_base}_gradcam_contorno.png",
            "panel": f"/static/resultados/gradcam/{nombre_base}/{nombre_base}_panel_4imagenes.png",
        },
    }


# =========================================================
# ENDPOINT: PERFIL RNA
# =========================================================
@app.get("/perfil_rna/{prediccion_clave}")
def perfil_rna(prediccion_clave: str):
    claves_validas = ["lung_aca", "lung_scc", "colon_aca", "lung_n", "colon_n"]
    if prediccion_clave not in claves_validas:
        raise HTTPException(
            status_code=400,
            detail=f"Clave no válida: {prediccion_clave}"
        )
    return obtener_perfil_rna(prediccion_clave)


# =========================================================
# TEST RÁPIDO
# =========================================================
@app.get("/test")
def test_rapido():
    imagenes = obtener_imagenes_validas()
    if not imagenes:
        raise HTTPException(status_code=404, detail="No hay imágenes válidas en imagenes_entrada")
    nombre = imagenes[0]
    return analizar_imagen(ImagenRequest(nombre_imagen=nombre))


# =========================================================
# TEST CON NOMBRE DIRECTO
# =========================================================
@app.get("/test/{nombre_imagen}")
def test_imagen(nombre_imagen: str):
    return analizar_imagen(ImagenRequest(nombre_imagen=nombre_imagen))
