from pathlib import Path

# =========================================================
# RUTAS DE LA APP
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODELO_DIR = BASE_DIR / "modelo"
IMAGENES_ENTRADA_DIR = BASE_DIR / "imagenes_entrada"
ANATOMIA_DIR = BASE_DIR / "anatomia"
RESULTADOS_DIR = BASE_DIR / "resultados"
GRADCAM_DIR = RESULTADOS_DIR / "gradcam"

MODELO_PATH = MODELO_DIR / "model.best.keras"

# Crear carpetas automáticamente
for carpeta in [
    MODELO_DIR,
    IMAGENES_ENTRADA_DIR,
    ANATOMIA_DIR,
    RESULTADOS_DIR,
    GRADCAM_DIR,
]:
    carpeta.mkdir(parents=True, exist_ok=True)

# =========================================================
# CLASES DEL MODELO
# =========================================================
CLASES = {
    "lung_n": "Tejido benigno de pulmón",
    "lung_aca": "Adenocarcinoma de pulmón",
    "lung_scc": "Carcinoma escamoso de pulmón",
    "colon_n": "Tejido benigno de colon",
    "colon_aca": "Adenocarcinoma de colon",
}

CLAVES_CLASES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]

# Para frontend (nombres más cortos si quieres)
CLASES_CORTAS = {
    "lung_n": "Pulmón normal",
    "lung_aca": "Pulmón AC",
    "lung_scc": "Pulmón SCC",
    "colon_n": "Colon normal",
    "colon_aca": "Colon AC",
}

# =========================================================
# CONFIGURACIÓN DE IMÁGENES
# =========================================================
TAMANO_IMG = (224, 224)
N_CANALES = 3
IMAGEN_SHAPE = (*TAMANO_IMG, N_CANALES)

MEDIA_IMAGEN = [0.485, 0.456, 0.406]
DESV_IMAGEN = [0.229, 0.224, 0.225]

# =========================================================
# GRADCAM CONFIG
# =========================================================
GRADCAM_ALPHA = 0.40
GRADCAM_THRESHOLD = 0.60
GRADCAM_MIN_AREA = 80

# =========================================================
# FILTRO DE PROBABILIDADES (UI)
# =========================================================
UMBRAL_PROBABILIDAD = 0.1  # %
MAX_RESULTADOS = 3

# =========================================================
# MAPEO A IMÁGENES ANATÓMICAS
# =========================================================
MAPA_ANATOMIA = {
    # COLON
    "colon_aca": "colon_coloreado.png",   # cáncer
    "colon_n": "pulmon_colon_grises",    # benigno

    # PULMÓN
    "lung_aca": "pulmon_coloreado.png",     # cáncer
    "lung_scc": "pulmon_coloreado.png",     # cáncer
    "lung_n": "pulmon_colon_grises",      # benigno
}

ANATOMIA_DEFAULT = "pulmon_colon_grises"



# =========================================================
# EXTENSIONES PERMITIDAS
# =========================================================
EXTENSIONES_VALIDAS = [".png", ".jpg", ".jpeg"]

# =========================================================
# DEBUG / CONTROL
# =========================================================
DEBUG = True