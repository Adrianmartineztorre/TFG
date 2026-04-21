# config.py
# Configuración global del TFG
# ==============================
# ⚠️ Este archivo NO importa TensorFlow
# Solo constantes, rutas y parámetros globales
# ==============================

from pathlib import Path

# ==============================
# Reproducibilidad (seed global)
# ==============================
SEED = 42

# ==============================
# Rutas del proyecto
# ==============================
RUTA_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATA = RUTA_PROYECTO / "DATA"
RUTA_DATA_RAW = RUTA_DATA / "RAW"
RUTA_DATASET_RAW = RUTA_DATA_RAW / "lung_colon_image_set"

RUTA_COLON = RUTA_DATASET_RAW / "colon_image_sets"
RUTA_LUNG = RUTA_DATASET_RAW / "lung_image_sets"

RUTA_DATA_PROCESADA = RUTA_DATA / "PROCESADA"
RUTA_SPLITS = RUTA_DATA_PROCESADA / "SPLITS"

RUTA_OUTPUTS = RUTA_PROYECTO / "OUTPUT"
MODELOS_DIR = RUTA_OUTPUTS / "MODELOS"
FIGURAS_DIR = RUTA_OUTPUTS / "FIGURAS"
METRICAS_DIR = RUTA_OUTPUTS / "METRICAS"

# Crear carpetas automáticamente
for p in [RUTA_DATA_PROCESADA, RUTA_SPLITS, RUTA_OUTPUTS, MODELOS_DIR, FIGURAS_DIR, METRICAS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ==============================
# Dataset y clases
# ==============================
CLASES = {
    "lung_n": "Tejido benigno de pulmón",
    "lung_aca": "Adenocarcinoma de pulmón",
    "lung_scc": "Carcinoma escamoso de pulmón",
    "colon_n": "Tejido benigno de colon",
    "colon_aca": "Adenocarcinoma de colon",
}

# Orden fijo para evitar cambios accidentales en el mapeo etiqueta->id
CLAVES_CLASES = ["lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"]
assert set(CLAVES_CLASES) == set(CLASES.keys()), "CLAVES_CLASES no coincide con CLASES"

NUM_CLASES = len(CLAVES_CLASES)

# ==============================
# Configuración de imágenes
# ==============================
TAMANO_IMG = (224, 224)
N_CANALES = 3
IMAGEN_SHAPE = (*TAMANO_IMG, N_CANALES)

# ==============================
# División de datos
# ==============================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-9, "Los splits deben sumar 1"

# ==============================
# Parámetros de entrenamiento
# ==============================
BATCH_SIZE = 32

EPOCHS_BASELINE = 20
EPOCHS_TRANSFER = 20

TASA_APRENDIZAJE_BASELINE = 1e-3
TASA_APRENDIZAJE_TRANSFER = 1e-4

# ==============================
# Parámetros específicos cnn_vgg
# ==============================
BATCH_SIZE_CNN_VGG = 16
EPOCHS_CNN_VGG = 20
TASA_APRENDIZAJE_CNN_VGG = 1e-4

# ==============================
# Aumento de datos (solo entrenamiento)
# ==============================
PARAMETROS_AUMENTO = dict(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.9, 1.1),
    fill_mode="nearest",
)

# ==============================
# Normalización
# ==============================
REESCALADO = 1.0 / 255.0

MEDIA_IMAGEN = [0.485, 0.456, 0.406]
DESV_IMAGEN = [0.229, 0.224, 0.225]

# ==============================
# Parámetros de callbacks (sin TF aquí)
# ==============================
MONITOR_METRIC = "val_loss"

EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4

REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.3
MIN_LR = 1e-6