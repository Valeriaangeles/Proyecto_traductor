import torch
from transformers import MarianMTModel, MarianTokenizer
import time
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

# 1. Cargar el modelo preentrenado (Helsinki-NLP para español-portugués)
model_name = "AventIQ-AI/opus-mt-en-roa_multilanguageTranslation"
print(f"Cargando el modelo {model_name}...")
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Detectar si hay GPU (en Mac es MPS, en otros es CUDA)
device = "cpu"
model.to(device)

def traducir(textos):
    """Traduce una lista de textos en español a portugués."""
    inputs = tokenizer(textos, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Configuración base inicial (sin optimizar)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, num_beams=4, length_penalty=1.0, max_length=512)
    
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

def evaluar_modelo_base():
    print("Cargando conjunto de prueba...")
    # Leemos el archivo de test que creamos antes
    df_test = pd.read_csv("datos/final/test.tsv", sep='\t', header=None, names=['es', 'pt'])
    
    # Para hacer pruebas rápidas, usaremos solo las primeras 100 oraciones del test
    # (Para la entrega final, quitarás el [:100] y usarás todo el set)
    textos_es = df_test['es'].tolist()[:100]
    referencias = [df_test['pt'].tolist()[:100]]

    print("Generando traducciones del modelo base...")
    start_time = time.time()
    
    # Traducimos en lotes para no saturar la memoria
    traducciones = []
    batch_size = 16
    for i in range(0, len(textos_es), batch_size):
        batch = textos_es[i:i+batch_size]
        traducciones.extend(traducir(batch))
    
    end_time = time.time()
    tiempo_inferencia = end_time - start_time

    print(f"Tiempo de inferencia (100 oraciones): {tiempo_inferencia:.2f} segundos")
    
    # Calcular métricas
    bleu = BLEU()
    chrf = CHRF()
    
    score_bleu = bleu.corpus_score(traducciones, referencias).score
    score_chrf = chrf.corpus_score(traducciones, referencias).score

    print("-" * 30)
    print("RESULTADOS DEL MODELO BASE")
    print("-" * 30)
    print(f"BLEU: {score_bleu:.2f}")
    print(f"chrF: {score_chrf:.2f}")
    print(f"Tiempo: {tiempo_inferencia:.2f} s")

    # Guardar las traducciones para el Experimento 3
    print("Guardando traducciones base...")
    df_resultados = pd.DataFrame({'es': textos_es, 'pt_referencia': referencias[0], 'pt_base': traducciones})
    df_resultados.to_csv("resultados/exp1_traducciones_base.tsv", sep='\t', index=False)

if __name__ == "__main__":
    evaluar_modelo_base()