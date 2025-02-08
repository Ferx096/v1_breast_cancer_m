# ============================
# IMPORTAR DATOS
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

# Importar df
ruta = r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Codigo\arch_descomprimidos\pdf_etiquetas.csv"
pdf_labels = pd.read_csv(ruta)
# importar datos
pgm_path = (
    r"C:\Users\grupo\OneDrive\Escritorio\MODELO_CANCER_MAMA\Dataset\MIAS\mias\MIASDBv1.21"
)



# ============================
# 1.TRABAJAR IMAGENES
# ============================
# Función para cargar imágenes en formato PGM
def cargar_pgm(pgm_file_path, img_size=(224, 224)):
    try:
        image = Image.open(pgm_file_path).convert("RGB")
        image = image.resize(img_size)
        return np.array(image) / 255.0
    except Exception as e:
        print(f"Error al cargar la imagen {pgm_file_path}: {e}")
        return None


# Cargar imágenes y asignar etiquetas (del df)BI-RADS
imagenes, clase_img = [], []
for _, row in pdf_labels.iterrows():
    img_id = row["img_id"]
    pgm_file_path = os.path.join(pgm_path, f"{img_id}.pgm")
    if os.path.exists(pgm_file_path):
        imagen = cargar_pgm(pgm_file_path)
        if imagen is not None:
            imagenes.append(imagen)
            clase_img.append(row["BI_RADS"])

# Convertir a arrays numpy
imagenes = np.array(imagenes)
clase_img = np.array(clase_img)
print("Imágenes cargadas iniciales:", imagenes.shape)
print("Etiquetas cargadas iniciales:", clase_img.shape)

# ======================================
# 2.BALANCEO DE DATOS + AUMENTO DE DATOS
# ======================================
# Data aumentation
inicial = pd.Series(clase_img).value_counts()
tamaño = 500
data_aumentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Funcion para balancear clases
def aumentar_data(clase_imagenes, tamaño, generator):
    aumentar_img = []
    aumentar_contador = tamaño - len(clase_imagenes)

    while aumentar_contador > 0:
        for img in clase_imagenes:
            if aumentar_contador <= 0:
                break
            img = np.expand_dims(img, 0)
            img_generada = generator.flow(img, batch_size=1)[0][0]
            aumentar_img.append(img_generada)
            aumentar_contador -= 1
    return np.array(aumentar_img)


# Balanceo de datos
img_balanceadas_aumentadas = []
clase_balanceadas_aumentadas = []
for clase in inicial.index:
    filtro = imagenes[np.array(clase_img) == clase]

    if len(filtro) < tamaño:
        imagenes_aumentadas = aumentar_data(filtro, tamaño, data_aumentation)
        filtro = np.vstack((filtro, imagenes_aumentadas))

    img_balanceadas_aumentadas.append(filtro)
    clase_balanceadas_aumentadas.extend([clase] * len(filtro))

# concatenar todas las clases balanceadas
clase_img_balanceada = np.array(clase_balanceadas_aumentadas)
# comparar
distribucion_balanceada = pd.Series(clase_img_balanceada).value_counts()
# df para graficar
df_distribucion_balanceada = distribucion_balanceada.reset_index()
df_distribucion_balanceada.columns = ["clase", "frecuencia"]


# Concatenar las imagenes aumentadas
img_balanceadas_aumentadas = np.vstack(img_balanceadas_aumentadas)
clase_balanceadas_aumentadas = np.array(clase_balanceadas_aumentadas)
print('imagenes', img_balanceadas_aumentadas.shape)
print('etq', clase_balanceadas_aumentadas.shape)

distribucion_final_balanceada = pd.Series(clase_balanceadas_aumentadas).value_counts()
# COMPARACION
print("\nDistribución inicial de clases:")
print(inicial)
print("\nDistribución  final balanceada de clases:")
print(distribucion_final_balanceada)



"""
# ============================
# 4. GRAFICO DE CLASES
# ============================

# GRAFICO1
# Distribucion inicial de clases
plt.figure(dpi=100, figsize=(6, 4))
ax = sns.countplot(data=pdf_labels, x="BI_RADS", hue="BI_RADS", palette="Blues_d")
# Añadir valores sobre las barras
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",  # Valor de la barra
        (p.get_x() + p.get_width() / 2.0, p.get_height()),  # Coordenadas
        ha="center",
        va="center",  # Alineación
        fontsize=10,
        color="black",
        xytext=(0, 5),  # Desplazamiento
        textcoords="offset points",
    )
ax.set_ylim(0, 220)
plt.title("Distribución Inicial de BI-RADS\n", fontsize=16)
plt.savefig("distribucion_inicial_birads.png", bbox_inches="tight")
plt.tight_layout()
plt.show()


# GRAFICO3
df_distribucion_aumentada = distribucion_final_balanceada.reset_index()
df_distribucion_aumentada.columns = ["clase", "frecuencia"]

plt.figure(dpi=100, figsize=(6, 4))
ax = sns.barplot(
    data=df_distribucion_aumentada,
    x="clase",
    y="frecuencia",
    hue="clase",
    palette="Blues_d",
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )
ax.set_ylim(0, 320)
plt.title("Distribución Aumentada de BI-RADS\n", fontsize=16)
plt.tight_layout()
plt.savefig("distribucion_aumentada_birads.png", bbox_inches="tight")
plt.show()
"""

# Ruta donde guardar el archivo
#ruta_guardado = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Codigo\imagenes_y_etiquetas_balanceadas4.pkl"

# Guardar las imágenes y etiquetas balanceadas en un archivo .pkl
#with open(ruta_guardado, 'wb') as f:
#    pickle.dump({'imagenes': img_balanceadas_aumentadas, 'etiquetas': clase_balanceadas_aumentadas}, f)
#print("Archivo guardado exitosamente en:", ruta_guardado)
