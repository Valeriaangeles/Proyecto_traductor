import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuración ---
URL_ES = "https://object.pouta.csc.fi/OPUS-SciELO/v1/moses/es-pt.txt.zip"
# Nota: SciELO en OPUS suele venir en un solo archivo zip con ambos idiomas.
# Usaremos una fuente directa más sencilla para empezar: el corpus Tatoeba.
URL_TATOEBA = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/es-pt.txt.zip"

def descargar_corpus(url, ruta_destino):
    if os.path.exists(ruta_destino):
        print(f"El archivo {ruta_destino} ya existe. Saltando descarga.")
        return
    print(f"Descargando desde {url}...")
    response = requests.get(url)
    with open(ruta_destino, 'wb') as f:
        f.write(response.content)
    print("Descarga completada.")

def procesar_y_dividir():
    # 1. Descargar
    ruta_zip = "datos/raw/corpus_es_pt.zip"
    descargar_corpus(URL_TATOEBA, ruta_zip)
    
    # 2. Leer el zip
    print("Leyendo el corpus...")
    import zipfile
    with zipfile.ZipFile(ruta_zip, 'r') as z:
        # Los nombres de los archivos dentro del zip pueden variar.
        # OPUS suele nombrarlos como 'Tatoeba.es-pt.es' y 'Tatoeba.es-pt.pt'
        nombres = z.namelist()
        print(f"Archivos dentro del ZIP: {nombres}")
        
        archivo_es = [n for n in nombres if n.endswith('.es')][0]
        archivo_pt = [n for n in nombres if n.endswith('.pt')][0]
        
        with z.open(archivo_es) as f_es, z.open(archivo_pt) as f_pt:
            lineas_es = f_es.read().decode('utf-8').strip().split('\n')
            lineas_pt = f_pt.read().decode('utf-8').strip().split('\n')

    # 3. Limpieza básica
    print(f"Pares originales: {len(lineas_es)}")
    pares_limpios = []
    for es, pt in zip(lineas_es, lineas_pt):
        es, pt = es.strip(), pt.strip()
        if es and pt: # Evitar líneas vacías
            pares_limpios.append((es, pt))
    
    # Eliminar duplicados
    pares_limpios = list(set(pares_limpios))
    print(f"Pares después de limpieza: {len(pares_limpios)}")

    # 4. Dividir en 80/10/10
    # Primero separamos el 20% (test + val) del 80% (train)
    train, temp = train_test_split(pares_limpios, test_size=0.2, random_state=42)
    # Luego dividimos ese 20% a la mitad para val y test
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 5. Guardar en archivos .tsv
    def guardar(datos, nombre):
        df = pd.DataFrame(datos, columns=['es', 'pt'])
        df.to_csv(f"datos/final/{nombre}.tsv", sep='\t', index=False, header=False)
        print(f"Guardado datos/final/{nombre}.tsv")

    guardar(train, "train")
    guardar(val, "val")
    guardar(test, "test")

if __name__ == "__main__":
    procesar_y_dividir()