# Documentación de Nuevas Capacidades: Memoria e Inteligencia Local

Esta documentación detalla las mejoras implementadas en el sistema de procesamiento local para las gafas **OMIGlasses**, enfocándose en la eficiencia de la GPU, la calidad de la transcripción y la integración con la memoria a largo plazo (RAG).

---

## 1. Optimización de Hardware (NVIDIA RTX 5070 12GB)

Para maximizar el uso de los 12GB de VRAM y evitar el uso de CPU, se han implementado técnicas de cuantización avanzada:

- **Cuantización 4-bit (BitsAndBytes)**: Los modelos **Qwen2.5-VL** (Visión) y **Qwen2.5-3B** (Análisis de texto) se cargan ahora en modo 4-bit (`nf4`). Esto reduce el consumo de memoria de ~15GB a solo **~4-5GB por modelo**, permitiendo que ambos coexistan en la GPU.
- **Carga de Memoria Inteligente (`low_cpu_mem_usage`)**: Se evita la saturación de la memoria RAM del sistema durante el arranque, eliminando errores de "paging file".
- **Patrón Singleton**: Todos los modelos se cargan una sola vez al inicio del servidor y se mantienen residentes en la GPU para respuestas instantáneas.

---

## 2. Transcripción de Audio Avanzada (Whisper Large-v3)

Se ha migrado a la versión más potente de Whisper para garantizar una precisión profesional en español:

- **Modelo `large-v3`**: Configurado para usar `float16` nativo en los núcleos CUDA.
- **Filtro Anti-Alucinaciones**: Sistema de limpieza que elimina frases basura comunes como *"Suscríbete"*, *"Amara.org"* o *"Gracias por ver el video"*.
- **VAD Sensible**: Detección de actividad de voz optimizada para no perder palabras cortas y filtrar ruidos de fondo.
- **Beam Search (size=5)**: El modelo analiza múltiples posibilidades antes de escribir, mejorando la coherencia en oraciones largas.

---

## 3. Integración con el Servicio de Memorias (RAG)

El sistema ahora actúa como una extensión de tu memoria personal, enviando automáticamente cada conversación a una base de datos vectorial externa.

- **Flujo Automático**: Al terminar un stream de audio, el sistema:
    1. Transcribe con Whisper.
    2. Interpreta con Qwen (Resumen, Tareas, Soluciones).
    3. Envía un `POST` al servicio RAG (`RAG_SERVICE_URL`).
- **Endpoint de Destino**: `/user/memories/`
- **Formato de Envío**:
    ```json
    {
      "transcript_original": "...",
      "interpretation": {
        "summary": "Resumen narrativo del recuerdo...",
        "action_items": [{"title": "...", "description": "...", "steps": []}],
        "risks": ["Sugerencias y soluciones..."]
      }
    }
    ```

---

## 4. Gestión de Registros Locales

Independientemente del servicio RAG, el servidor mantiene un respaldo físico de todo lo procesado:

- **Carpeta `transcriptions/`**: Contiene archivos `.txt` organizados por fecha y hora.
- **Contenido del Registro**: Cada archivo incluye la transcripción original palabra por palabra y el análisis estructurado de la IA justo debajo.

---

## 5. Configuración del Entorno (.env)

Nuevas variables clave:
- `FWHISPER_MODEL_ID=large-v3`: Modelo de máxima precisión.
- `CONV_LLM_MAX_NEW_TOKENS=1024`: Evita cortes en resúmenes largos.
- `RAG_SERVICE_URL=http://192.168.0.9:8787`: Dirección del servicio de base de datos vectorial.
- `HF_TOKEN`: Token de seguridad para descarga de modelos (no se sube a Git).

---

## 6. Endpoints de Audio (REST & WebSocket)

Se han habilitado tres vías principales para procesar audio, optimizadas para diferentes casos de uso:

### **A. Tiempo Real (WebSocket)**
- **Ruta**: `ws://SERVER_IP:8989/ws/audio`
- **Uso**: Diseñado para streaming continuo desde la app Flutter.
- **Modelo**: Utiliza **Whisper Large-v3** en GPU.
- **Flujo**: Envío de chunks binarios -> `end_of_stream` -> Análisis completo + Sincronización RAG.
- **Guía Detallada**: Consulta [FLUTTER_WS_GUIDE.md](file:///c:/Users/paulm/Documents/dev-projects/IAAplicada/OMIGlasses/local_models_api/FLUTTER_WS_GUIDE.md).

### **B. Subida de Archivos (REST - Whisper)**
- **Ruta**: `POST /audio/whisper`
- **Uso**: Ideal para procesar grabaciones ya finalizadas (.wav, .mp3, .ogg).
- **Ventaja**: Utiliza el mismo motor **Large-v3** optimizado y filtros anti-alucinaciones.

### **C. Subida de Archivos (REST - SenseVoice)**
- **Ruta**: `POST /audio/sense-voice`
- **Uso**: Acceso al modelo **SenseVoiceSmall**.
- **Nota**: Este modelo es extremadamente rápido para eventos acústicos (aplausos, risas), pero su precisión en español es inferior a Whisper Large-v3.

---

**Nota**: Esta arquitectura convierte a las OMIGlasses en un dispositivo de computación de borde (Edge Computing) extremadamente capaz, delegando solo el almacenamiento vectorial al servicio externo.
