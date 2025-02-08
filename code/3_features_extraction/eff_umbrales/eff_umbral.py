# ============================
# LIBRERIAS
# ============================
# Importar datos
import os
import zipfile
import pandas as pd
import numpy as np
import pickle

# Procesamiento de imagenes
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# MODELOS
from tensorflow.keras.applications import (
    EfficientNetV2B0,
    MobileNetV3Large,
    DenseNet121,
    VGG16,
)
from timm import create_model
import timm
from torchinfo import summary
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Importar df
# ruta = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Codigo\pdf_etiquetas.csv"
# pdf_labels = pd.read_csv(ruta)

# =================================
# 1.IMPORTAR IMAGENES Y ETIQUETAS
# ================================
ruta_guardado = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Codigo\imagenes_y_etiquetas_balanceadas4.pkl"
with open(ruta_guardado, "rb") as f:
    datos_cargados = pickle.load(f)


# =================================
# 2.MODELOS
# ================================
# Acceder a imagenes y etiquetas cargadas
imagenes = datos_cargados["imagenes"]
etiquetas = datos_cargados["etiquetas"]
input_shape = (224, 224, 3)  # tamaño de la entrada

# Importar modelos:
eff = EfficientNetV2B0(weights="imagenet", input_shape=input_shape)
extractor_eff = Model(inputs=eff.input, outputs=eff.get_layer("avg_pool").output)  # extraer 1280 caracteristicas

# Extraer caracteristicas de las imagenes
caracteristicas_eff = extractor_eff.predict(imagenes, batch_size=32, verbose=1)
print(caracteristicas_eff.shape)

# GUARDAR CARACTERISTICAS EXTRAUDAS
ruta_eff = r"C:\Users\grupo\OneDrive\Escritorio\Nueva carpeta\prueba_umbrales\eff_umbrales\caracteristicas_efficientnetV2B0.pkl"
with open(ruta_eff, "wb") as f:
    pickle.dump(caracteristicas_eff, f)
print("Caracteristicas guardadas en:", ruta_eff)


# =================================
# 3.NORMALIZAR CARACTERISTICAS PARA IM
# ================================
scaler = StandardScaler()
features_eff_norm = scaler.fit_transform(caracteristicas_eff)


# =================================
# 4.CALCULAR INFORMACION MUTUA
# ================================
# Calcular la IM entre caracteristicas y las clases BI-RADS
im = mutual_info_classif(features_eff_norm, etiquetas, random_state=42)
# DF
df_im = pd.DataFrame(
    {"Caracteristica": [f"Feature_{i}" for i in range(caracteristicas_eff.shape[1])], "IM": im}
)

print(df_im)


# =================================
# 4.PRUEBA DE UMBRALES
# ================================
umbrales = [0.01, 0.02, 0.03, 0.05, 0.06, 0.1]
resultados = {}

for umbral in umbrales:
    # Seleccionar caracteristicas con IM mayor al umbral
    seleccion_caracteristicas = df_im[df_im["IM"] > umbral]["Caracteristica"]
    indices_seleccionados = [int(f.split("_")[1]) for f in seleccion_caracteristicas]
    caracteristicas_reducidas = features_eff_norm[
        :, indices_seleccionados
    ]  # selecciona solo las filas
    print(
        f"Número de características seleccionadas: {caracteristicas_reducidas.shape[1]}"
    )

    # ===============================
    # 5.EVALUACUON CON LAZYPREDICT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        caracteristicas_reducidas,
        etiquetas,
        test_size=0.3,
        random_state=42,
        stratify=etiquetas,
    )

    # aplicar lazypredict
    lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = lazy.fit(X_train, X_test, y_train, y_test)

    # Guardar resultados
    resultados[umbral] = models
    print(models)

# =================================
# 6. SELECCIÓN FINAL DEL MODELO
# =================================
for umbral, models in resultados.items():
    print(f"\n=== Resultados para umbral: {umbral} ===")
    print(models.sort_values(by="Accuracy", ascending=False).head(5))

