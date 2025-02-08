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
import torch
import torch.nn as nn
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

# Crear modelo resnext50_32x4d preentrenado
res = create_model("resnext50_32x4d", pretrained=True)

# Remover la parte superior del modelo y usar la capa `flatten` como salida
extractor_res = nn.Sequential(*(list(res.children())[:-1]))
extractor_res.add_module("flatten", nn.Flatten())

print(f"Modelo configurado con capa de extracción: {list(extractor_res.children())[-1]}")  # Verificar la capa de extracción

# Configurar el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor_res = extractor_res.to(device)


# =================================
# 2. EXTRAER CARACTERÍSTICAS
# =================================
def extraer_caracteristicas(modelo, imagenes, batch_size=32):
    modelo.eval()
    caracteristicas = []
    with torch.no_grad():
        for i in range(0, len(imagenes), batch_size):
            batch = torch.tensor(imagenes[i : i + batch_size], dtype=torch.float32).to(
                device
            )
            salida = modelo(batch)
            caracteristicas.append(salida.cpu().numpy())
    return np.vstack(caracteristicas)

# Cargar imágenes
imagenes_tensor = torch.tensor(imagenes.transpose(0, 3, 1, 2))  # Convertir a formato PyTorch (N, C, H, W)
caracteristicas_res = extraer_caracteristicas(extractor_res, imagenes_tensor)

print(f"Shape de características extraídas: {caracteristicas_res.shape}")

#GUARDAR CARACTERISTICAS EXTRAIDAS
ruta = r"C:\Users\grupo\OneDrive\Escritorio\Nueva carpeta\prueba_umbrales\res_umbrales\caracteristicas_resnet18.pkl"
with open(ruta, 'wb') as f:
    pickle.dump(caracteristicas_res, f)
print('Caracteristicas guardadas en:', ruta)

"""
# =================================
# 3. NORMALIZAR CARACTERÍSTICAS
# =================================
scaler = StandardScaler()
caracteristicas_res_norm = scaler.fit_transform(caracteristicas_res)

# =================================
# 4. CALCULAR INFORMACIÓN MUTUA
# =================================
im = mutual_info_classif(caracteristicas_res_norm, etiquetas, random_state=42)
df_im = pd.DataFrame(
    {
        "Característica": [f"Feature_{i}" for i in range(caracteristicas_res.shape[1])],
        "IM": im,
    }
)

df_im = df_im.sort_values(by="IM", ascending=False)
print(df_im)

# =================================
# 5. PRUEBA DE UMBRALES
# =================================
umbrales = [0.01, 0.02, 0.03, 0.05, 0.06, 0.1]
resultados = {}

for umbral in umbrales:
    print(f"\n=== Evaluando con umbral: {umbral} ===")

    # Seleccionar características con IM mayor al umbral
    seleccion_caracteristicas = df_im[df_im["IM"] > umbral]["Característica"]
    indices_seleccionados = [int(f.split("_")[1]) for f in seleccion_caracteristicas]
    caracteristicas_reducidas = caracteristicas_res_norm[:, indices_seleccionados]

    print(
        f"Número de características seleccionadas: {caracteristicas_reducidas.shape[1]}"
    )

    # =================================
    # 6. EVALUACIÓN CON LAZYPREDICT
    # =================================
    X_train, X_test, y_train, y_test = train_test_split(
        caracteristicas_reducidas,
        etiquetas,
        test_size=0.3,
        random_state=42,
        stratify=etiquetas,
    )

    lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = lazy.fit(X_train, X_test, y_train, y_test)

    # Guardar resultados
    resultados[umbral] = models
    print(models.head(10))

# =================================
# 7. SELECCIÓN FINAL DEL MODELO
# =================================
for umbral, models in resultados.items():
    print(f"\n=== Resultados para umbral: {umbral} ===")
    print(models.sort_values(by="Accuracy", ascending=False).head(5))
    print(
        f"Número de características seleccionadas: {caracteristicas_reducidas.shape[1]}"
    )
"""