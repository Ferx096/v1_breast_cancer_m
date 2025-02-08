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


# =================================
# 1.IMPORTAR IMAGENES Y ETIQUETAS
# ================================
ruta_guardado = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\arch_descomprimidos\imagenes_y_etiquetas_balanceadas4.pkl"
with open(ruta_guardado, "rb") as f:
    datos_cargados = pickle.load(f)


# =================================
# 2.IMAGENES Y ETIQUETAS
# ================================
# Acceder a imagenes y etiquetas cargadas
imagenes = datos_cargados["imagenes"]
etiquetas = datos_cargados["etiquetas"]
input_shape = (224, 224, 3)  # tamaño de la entrada

# =================================
# 3. CARGAR CARACTERISTICAS
# ================================
# solo cargare las caracteristicas que me superan el 75%
ruta_inc = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\extraccion_caracteriscas_modelos\inc_umbrales\caracteristicas_inceptionv8.pkl"
ruta_res = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\extraccion_caracteriscas_modelos\res_umbrales\caracteristicas_resnet18.pkl"
ruta_den = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\extraccion_caracteriscas_modelos\den_umbrales\caracteristicas_densenet21.pkl"

# cargar caracteristicas
with open(ruta_inc, "rb") as f_inc:
    caracteristicas_inc = pickle.load(f_inc)
with open(ruta_res, "rb") as f_res:
    caracteristicas_res = pickle.load(f_res)
with open(ruta_den, "rb") as f_den:
    caracteristicas_den = pickle.load(f_den)



# Convertir características a NumPy arrays
caracteristicas_inc = np.array(caracteristicas_inc)
caracteristicas_res = np.array(caracteristicas_res)
caracteristicas_den = np.array(caracteristicas_den)
concatenado_umbral_1 = np.concatenate(
    [caracteristicas_inc, caracteristicas_res], axis=1
)
concatenado_umbral_2 = np.concatenate(
    [caracteristicas_den], axis=1
)
# normalizar caracteristicas
scaler = StandardScaler()
caracteristicas_umb_1 = scaler.fit_transform(concatenado_umbral_1)
caracteristicas_umb_2 = scaler.fit_transform(concatenado_umbral_2)

# =================================
# 4.CALCULAR IM POR MODELO
# ================================
# Calcular IM para cada conjunto de caracteristicas
im_den = mutual_info_classif(caracteristicas_umb_1, etiquetas, random_state=42)
im_con = mutual_info_classif(caracteristicas_umb_2, etiquetas, random_state=42)

# Filtrar las caracteristicas segun el valor de IM para cada conjunto
# Dense
caracteristicas_umb1_seleccionadas = caracteristicas_umb_1[:, im_den >= 0.01]
caracteristicas_umb2_seleccionadas = caracteristicas_umb_2[:, im_con >= 0.02]

# =================================
# 5.CONCATEANAR CARACTERISTICAS
# =================================
caracteristicas_concatenadas = np.concatenate(
    [caracteristicas_umb1_seleccionadas, caracteristicas_umb2_seleccionadas],
    axis=1,
)

# normalizar
caracteristicas_concantenadas_norm = scaler.fit_transform(caracteristicas_concatenadas)

print("Caracteristicas concatenadas", caracteristicas_concantenadas_norm.shape)
# Guardar caracteristicas
# ruta_final = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\arch_descomprimidos\visualizacion_pruebas\caracteristicas_concatenadas_reducidas_final.pkl"
# with open(ruta_final, "wb") as f:
#    pickle.dump(caracteristicas_concatenadas, f)
#    print("GUARDADO")


# ===============================================================================================================================

# Convertir pdf caracteristicas
im = mutual_info_classif(caracteristicas_concantenadas_norm, etiquetas, random_state=42)
df_im = pd.DataFrame(
    {
        "Característica": [
            f"Feature_{i}" for i in range(caracteristicas_concatenadas.shape[1])
        ],
        "IM": im,
    }
).sort_values(by="IM", ascending=False)
print(df_im)

# =================================
# 6. EVALUACION DE UMBRALES
# =================================
umbrales = [0.01, 0.02, 0.03, 0.05, 0.06, 0.1]
resultados = {}

for umbral in umbrales:
    print(f"\n=== Evaluando con umbral: {umbral} ===")

    # Seleccionar características con IM mayor al umbral
    seleccion_caracteristicas = df_im[df_im["IM"] > umbral]["Característica"]
    indices_seleccionados = [int(f.split("_")[1]) for f in seleccion_caracteristicas]
    caracteristicas_reducidas = caracteristicas_concantenadas_norm[
        :, indices_seleccionados
    ]

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
