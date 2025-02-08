# Importar datos
import os
import zipfile

# Procesamiento, manipulacion y visualizacion de datos
import pandas as pd
import numpy as np
import PyPDF2
import fitz


# ============================
# 1. RUTAS DE ACCESO
# ============================
mias = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Dataset\MIAS\miasdbv1.21.zip"
mias_descomprimir = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Dataset\MIAS\mias"
# Descomprimir archivo zip
# with zipfile.ZipFile(mias, 'r') as zip_ref:
# zip_ref.extractall(mias_descomprimir)
# Ubicacion de imagenes y pdf
pgm_path = (
    r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Dataset\MIAS\mias\MIASDBv1.21"
)
pdf_path = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Dataset\MIAS\mias\MIASDBv1.21\00README.pdf"


# ============================
# 2.TRABAJAR PDF
# ============================
# Clases BI-RADS
mapeo_birads = {
    "NORM": "BI-RADS_1",
    "B2": "BI-RADS_2",
    "B3": "BI-RADS_3",
    "MISC": "BI-RADS_4",
    "CIRC": "BI-RADS_4",
    "ASYM": "BI-RADS_4",
    "CALC": "BI-RADS_4",
    "ARCH": "BI-RADS_5",
    "SPIC": "BI-RADS_5",
}


# Función para asignar BI-RADS
def asignar_birads(clase_anomalia, severidad):
    if clase_anomalia == "NORM":
        return mapeo_birads["NORM"]
    elif severidad == "B":
        if clase_anomalia in ["MISC", "ARCH", "ASYM", "SPIC"]:
            return mapeo_birads["B2"]
        elif clase_anomalia in ["CALC", "CIRC"]:
            return mapeo_birads["B3"]
    elif severidad == "M":
        if clase_anomalia in ["CALC", "CIRC", "MISC", "ASYM"]:
            return mapeo_birads["MISC"]
        elif clase_anomalia in ["SPIC", "ARCH"]:
            return mapeo_birads["ARCH"]
    return "Unknown"


# Función para verificar si un valor puede ser convertido a entero
def es_entero(valor):
    try:
        int(valor)
        return True
    except ValueError:
        return False


# Función para extraer texto del PDF
def extraer_texto_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# Función para extraer etiquetas y características del PDF
def labels_pdf(pdf_path):
    data = []
    text = extraer_texto_pdf(pdf_path)
    lines = text.splitlines()
    current_entry = None  # Para rastrear la entrada actual

    for linea in lines:
        linea = linea.lstrip().rstrip()

        # Limpiar texto innecesario
        for page in range(3, 13):
            pagina = f"MIAS database Page {page}"
            linea = linea.replace(pagina, "")
        if "mdb 273ll" in linea:
            linea = linea.replace("mdb 273ll", "mdb273ll")
        elif "mdb003l l" in linea:
            linea = linea.replace("mdb003l l", "mdb003ll")

        if not linea:  # Ignorar líneas vacías
            continue

        if linea.startswith("mdb"):  # Nueva entrada
            # Procesar entrada actual y reiniciar
            if current_entry:
                data.append(current_entry)
            current_entry = {
                "img_id": None,
                "tipo_tejido": None,
                "clase_anomalia": None,
                "severidad": None,
                "x_coord": None,
                "y_coord": None,
                "radio": None,
                "BI_RADS": None,
                "dimension": None,
                "lado": None,
                "clase_amomalia_2": None,
                "severidad_2": None,
                "x2": None,
                "y2": None,
                "radio_2": None,
                "clase_amomalia_3": None,
                "severidad_3": None,
                "x3": None,
                "y3": None,
                "radio_3": None,
            }

            # Procesar primera línea
            parts = linea.split()
            if len(parts) >= 1:
                current_entry["img_id"] = parts[0]
                current_entry["tipo_tejido"] = parts[1] if len(parts) > 1 else None
                current_entry["clase_anomalia"] = parts[2] if len(parts) > 2 else None
                current_entry["severidad"] = parts[3] if len(parts) > 3 else None
                current_entry["x_coord"] = (
                    float(parts[4]) if len(parts) > 4 and es_entero(parts[4]) else None
                )
                current_entry["y_coord"] = (
                    float(parts[5]) if len(parts) > 5 and es_entero(parts[5]) else None
                )
                current_entry["radio"] = (
                    float(parts[6]) if len(parts) > 6 and es_entero(parts[6]) else None
                )

                # Otras características
                current_entry["lado"] = (
                    "izquierda" if current_entry["img_id"][-2] == "l" else "derecho"
                )
                current_entry["dimension"] = current_entry["img_id"][-1]
                current_entry["BI_RADS"] = asignar_birads(
                    current_entry["clase_anomalia"], current_entry["severidad"]
                )

        elif current_entry:  # Línea adicional relacionada con la entrada actual
            parts = linea.split()
            if len(parts) >= 3:  # Asegurar que haya al menos 3 elementos
                idx = (
                    2
                    if current_entry["clase_amomalia_2"] is None
                    else 3 if current_entry["clase_amomalia_3"] is None else None
                )
                if idx:
                    current_entry[f"clase_amomalia_{idx}"] = (
                        parts[0] if parts[0] != 0 else "-"
                    )
                    current_entry[f"severidad_{idx}"] = (
                        parts[1] if parts[1] != 0 else "-"
                    )
                    current_entry[f"x{idx}"] = (
                        float(parts[2])
                        if len(parts) > 2 and es_entero(parts[2])
                        else None
                    )
                    current_entry[f"y{idx}"] = (
                        float(parts[3])
                        if len(parts) > 3 and es_entero(parts[3])
                        else None
                    )
                    current_entry[f"radio_{idx}"] = (
                        float(parts[4])
                        if len(parts) > 4 and es_entero(parts[4])
                        else None
                    )

    # Agregar última entrada
    if current_entry:
        data.append(current_entry)

    return pd.DataFrame(data)


# Extraer datos del PDF
pdf_labels_inicial = labels_pdf(pdf_path)
pdf_labels = pdf_labels_inicial.sort_values(by="BI_RADS", ascending=True).reset_index(
    drop=True
)


# ============================
# 3.LIMPIEZA DE DATOS
# ============================
# print(pdf_labels.isna().sum())
# Remmplazar nulos
pdf_labels.fillna(0, inplace=True)  # = x_coord, y_coord, radio
pdf_labels["severidad"] = pdf_labels["severidad"].replace(0, "-")
pdf_labels["clase_amomalia_2"] = pdf_labels["clase_amomalia_2"].replace(0, "-")
pdf_labels["clase_amomalia_3"] = pdf_labels["clase_amomalia_3"].replace(0, "-")
pdf_labels["severidad_2"] = pdf_labels["severidad_2"].replace(0, "-")
pdf_labels["severidad_3"] = pdf_labels["severidad_3"].replace(0, "-")

ruta = r"C:\Users\grupo\OneDrive\Escritorio\CANCER_MAMA\Codigo\pdf_etiquetas.csv"
pdf_labels.to_csv(ruta, index=False)
# print(pdf_labels.info())
# print(pdf_labels.head())
