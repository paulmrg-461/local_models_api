import os
import random
import json
import torch
import time
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIGURACIÓN ---
PHOTO_DIR = "/home/devpaul/Documents/dev_projects/DevPaul/vision_cex_training/samples/photos"
JSON_OUTPUT_PATH = "reporte_local_vlm.json"

# Usaremos el modelo de 3 Billones de parámetros (Perfecto para 12GB VRAM)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

def main():
    print("🚀 Iniciando motor VLM Local...")
    
    # 1. Cargar el modelo directamente en la GPU usando bfloat16
    print(f"📦 Cargando {MODEL_ID} en la RTX 5070 (Puede tardar la primera vez)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("   ✅ Modelo cargado exitosamente en VRAM.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # 2. Buscar y seleccionar imágenes
    if not os.path.exists(PHOTO_DIR):
        print(f"❌ Error: La carpeta no existe en {PHOTO_DIR}")
        return
        
    todas_las_fotos = [f for f in os.listdir(PHOTO_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not todas_las_fotos:
        print("⚠️ No se encontraron imágenes en la carpeta.")
        return
        
    cantidad_a_elegir = min(5, len(todas_las_fotos))
    fotos_elegidas = random.sample(todas_las_fotos, cantidad_a_elegir)
    
    print(f"\n🎲 Se seleccionaron {cantidad_a_elegir} fotos al azar para inspección.")
    
    resultados = []

    # hora de inicio general
    inicio_total = time.time()

    # 3. Procesar cada foto
    for foto in fotos_elegidas:
        ruta_completa = os.path.join(PHOTO_DIR, foto)
        print(f"\n🔍 Analizando: {foto}...")
        
        try:
            foto_inicio = time.time()
            # Estructura de mensaje para Qwen-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ruta_completa},
                        {"type": "text", "text": "Eres un experto inspector de vehículos. Describe detalladamente qué daño, golpe, rayón o novedad ves en esta foto del bus. Sé conciso pero preciso. Responde en español."},
                    ],
                }
            ]
            
            # Preparar los tensores para la GPU
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda") # Enviamos los datos a la RTX 5070
            
            # Generar respuesta (Inferencia)
            with torch.no_grad(): # Desactivamos el cálculo de gradientes para ahorrar memoria
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
            # Limpiar el output para mostrar solo la respuesta nueva
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            descripcion = output_text[0].strip()
            duracion = time.time() - foto_inicio
            print(f"   🤖 VLM: {descripcion}")
            print(f"   ⏱️ Tiempo de análisis de esta imagen: {duracion:.2f} s")
            
            resultados.append({
                "archivo": foto,
                "descripcion": descripcion,
                "duracion_s": duracion
            })
            
        except Exception as e:
            print(f"   ⚠️ Error procesando la imagen {foto}: {e}")
            resultados.append({
                "archivo": foto,
                "error": str(e)
            })

    # fin de procesamiento
    total_duracion = time.time() - inicio_total
    print(f"\n✅ Procesamiento completo en {total_duracion:.2f} segundos\n")

    # 4. Guardar resultados
    try:
        with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Reporte guardado en: {JSON_OUTPUT_PATH}")
    except Exception as e:
        print(f"\n❌ Error al guardar el archivo JSON: {e}")

if __name__ == "__main__":
    main()