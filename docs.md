Perfecto, este enfoque (backend propio en Python con FastAPI sobre una RTX / Jetson) es muy buena idea para tu caso.

A continuación tienes una **guía completa** en formato markdown (puedes copiarla a un `.md` si quieres) para construir:

- Un backend FastAPI que:
  - **Reciba audio en tiempo real por WebSocket**.
  - **Reciba fotos periódicas por HTTP** (POST).
- Manteniendo **el mismo tipo de datos** que ya envías a GEMINI (audio crudo o codificado, imágenes en bytes/base64), de forma que a nivel de app sólo cambies la URL y, como mucho, el protocolo (HTTP → WebSocket para audio).

---

# Guía: Backend FastAPI para audio en tiempo real y fotos periódicas

## 1. Arquitectura general

### Objetivo

- Backend Python que corra en:
  - Un PC con **NVIDIA RTX 5070 12GB**, o
  - Una **Jetson Nano / Orin**.
- Usar **FastAPI** como framework web.
- Exponer:
  - **WebSocket** `/ws/audio` para recibir audio en streaming.
  - **HTTP POST** `/vision/frame` para recibir fotos cada cierto tiempo.
- Después tú conectarás modelos de IA (ASR, visión, multimodal) a estos endpoints.

### Flujo alto nivel

- **Audio:**
  - Tu app móvil abre un WebSocket a `ws://TU_HOST:PUERTO/ws/audio`.
  - Envía:
    - Un mensaje inicial de configuración (JSON).
    - Luego fragmentos de audio (bytes o base64) a medida que recibe el stream desde las gafas.
  - El backend:
    - Acumula los frames.
    - Opcionalmente hace inferencia incremental (streaming) o por bloques.
    - Devuelve por el mismo WebSocket mensajes JSON con:
      - Texto transcrito, o
      - Respuesta del modelo (según cómo lo orquestes).

- **Fotos:**
  - Tu app envía cada X segundos/minutos un `POST /vision/frame`.
  - Cuerpo:
    - `multipart/form-data` con el archivo, **o**
    - JSON con la imagen en base64 (similar a como se la mandas a GEMINI).
  - El backend:
    - Decodifica la imagen.
    - Llama al modelo de visión / multimodal.
    - Devuelve un JSON con descripción, etiquetas, etc.

---

## 2. Requisitos de entorno

### 2.1 Hardware

- PC con **NVIDIA RTX 5070 12GB**, o
- **Jetson Nano / Orin** (para producción embebida).
- Conectividad de red entre:
  - Backend (PC/Jetson).
  - Dispositivo móvil que envía audio/fotos.

### 2.2 Software base

En el servidor (Ubuntu / Debian recomendado):

```bash
# Python
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate
```

Instalar FastAPI y dependencias mínimas:

```bash
pip install "fastapi[standard]" uvicorn[standard]
```

Dependiendo del modelo que uses:

- Para **PyTorch + CUDA** en RTX:
  - Instalar drivers NVIDIA + CUDA adecuados.
  - Luego instalar PyTorch con CUDA (según instrucciones oficiales).
- Para **Jetson**:
  - Usar el stack que trae JetPack (CUDA, cuDNN) y versiones compatibles de PyTorch/ONNX Runtime/TensorRT.

---

## 3. Diseño de API y formato de datos

### 3.1 Mantener formato “similar a GEMINI”

Aunque GEMINI use su propio protocolo, tú puedes:

- Mantener **el mismo contenido de audio**:
  - Frecuencia (ej. 16 kHz), mono/stereo.
  - Codificación (PCM 16-bit, Opus, etc.).
- Mantener **el mismo formato de imagen**:
  - Bytes de JPEG/PNG.
  - O la imagen en base64 dentro de JSON (igual que haces para Gemini Vision).

Lo único que cambia:

- Audio: pasas de HTTP/REST a WebSocket (stream).
- Fotos: puedes seguir usando HTTP POST.

---

## 4. Endpoint WebSocket para audio en tiempo real

### 4.1 Protocolo propuesto

WebSocket `ws://host:port/ws/audio`

Mensajes **cliente → servidor**:

1. **Mensaje inicial (texto JSON):**

```json
{
  "type": "config",
  "session_id": "1234-5678",
  "sample_rate": 16000,
  "encoding": "pcm16", 
  "language": "es"
}
```

2. **Frames de audio:**

- Opción A (recomendable): mandar **mensajes binarios** con los bytes crudos (PCM/Opus).
- Opción B: mandar texto JSON con base64 (más overhead, pero más fácil de inspeccionar).

Ejemplo Opción B (texto):

```json
{
  "type": "audio_chunk",
  "chunk_id": 1,
  "data_b64": "BASE64_AUDIO_HERE"
}
```

Mensajes **servidor → cliente** (texto JSON):

- Parciales o finales:

```json
{
  "type": "partial_result",
  "text": "Esto es lo que llevo..."
}
```

```json
{
  "type": "final_result",
  "text": "Transcripción/Respuesta final"
}
```

### 4.2 Implementación básica en FastAPI

`app/main.py`:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import base64
from typing import Optional

app = FastAPI()


class AudioSession:
    def __init__(self, session_id: str, sample_rate: int, encoding: str, language: str):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.language = language
        self.audio_buffer = bytearray()  # almacenar audio crudo

    def add_chunk_bytes(self, data: bytes):
        self.audio_buffer.extend(data)

    def add_chunk_b64(self, data_b64: str):
        self.audio_buffer.extend(base64.b64decode(data_b64))

    def reset(self):
        self.audio_buffer = bytearray()


async def run_asr_on_buffer(buffer: bytes, language: str) -> str:
    # TODO: aquí conectas tu modelo (Whisper, etc.)
    # Por ahora devolvemos un texto de ejemplo
    return f"[ASR dummy] Longitud buffer: {len(buffer)} bytes, language={language}"


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    session: Optional[AudioSession] = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                # Mensaje de texto (JSON)
                import json
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    session = AudioSession(
                        session_id=data.get("session_id", "unknown"),
                        sample_rate=data.get("sample_rate", 16000),
                        encoding=data.get("encoding", "pcm16"),
                        language=data.get("language", "es"),
                    )
                    await websocket.send_json({
                        "type": "config_ack",
                        "message": "config received",
                    })

                elif msg_type == "audio_chunk":
                    if session is None:
                        continue
                    session.add_chunk_b64(data["data_b64"])

                elif msg_type == "end_of_stream":
                    if session is None:
                        continue
                    text = await run_asr_on_buffer(
                        bytes(session.audio_buffer),
                        session.language,
                    )
                    await websocket.send_json({
                        "type": "final_result",
                        "text": text,
                    })
                    session.reset()

            elif "bytes" in message:
                # Mensaje binario: asumimos audio crudo
                if session is None:
                    # podrías ignorarlo o cerrar el socket
                    continue
                raw_bytes = message["bytes"]
                session.add_chunk_bytes(raw_bytes)

    except WebSocketDisconnect:
        # El cliente cerró la conexión
        pass
```

### 4.3 Integrando el modelo de IA de audio

Dentro de `run_asr_on_buffer`:

- Conectas tu modelo (ej. Whisper, NeMo, etc.).
- Ejemplo con Whisper (conceptual):

```python
import torch
import whisper

model = whisper.load_model("small")  # con CUDA si está disponible

async def run_asr_on_buffer(buffer: bytes, language: str) -> str:
    import soundfile as sf
    import io

    # Suponiendo PCM16 mono 16k, podrías convertir a wav in-memory
    # o adaptar a lo que tu modelo espera.
    # Aquí solo mostramos la idea general.

    # TODO: convertir buffer -> array de audio -> pasar a modelo
    # audio_array = ...
    # result = model.transcribe(audio_array, language=language)
    # return result["text"]

    return f"[ASR dummy] Longitud buffer: {len(buffer)} bytes, language={language}"
```

---

## 5. Endpoint HTTP para fotos periódicas

### 5.1 Diseño de endpoint

URL: `POST /vision/frame`

Dos variantes:

1. **multipart/form-data** (recomendado si ya manejas archivos):

   - Campo `file`: imagen JPEG/PNG.
   - Campos opcionales: `session_id`, `timestamp`, etc.

2. **JSON + base64** (similar a GEMINI Vision):

   ```json
   {
     "session_id": "1234-5678",
     "image_b64": "BASE64_DE_LA_IMAGEN",
     "metadata": {
       "timestamp": 1234567890
     }
   }
   ```

### 5.2 Implementación en FastAPI (multipart/form-data)

```python
from fastapi import UploadFile, File, Form
from typing import Optional
from PIL import Image
import io

@app.post("/vision/frame")
async def receive_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO: llamar a tu modelo de visión/multimodal
    # result = run_vision_model(image)

    result = {
        "description": "[dummy] Imagen recibida",
        "size": image.size,
        "session_id": session_id,
    }

    return JSONResponse(result)
```

### 5.3 Implementación en FastAPI (JSON + base64)

```python
from pydantic import BaseModel
import base64

class VisionRequest(BaseModel):
    session_id: Optional[str] = None
    image_b64: str
    # metadata opcional

@app.post("/vision/frame_b64")
async def receive_frame_b64(payload: VisionRequest):
    image_bytes = base64.b64decode(payload.image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO: modelo de visión
    # result = run_vision_model(image)

    result = {
        "description": "[dummy] Imagen recibida (b64)",
        "size": image.size,
        "session_id": payload.session_id,
    }

    return JSONResponse(result)
```

---

## 6. Ejecutar el servidor

Desde tu entorno virtual:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Desde el móvil (en la misma red) apuntas a:
  - `ws://IP_DEL_SERVIDOR:8000/ws/audio`
  - `http://IP_DEL_SERVIDOR:8000/vision/frame`
  - `http://IP_DEL_SERVIDOR:8000/vision/frame_b64`

---

## 7. Limitaciones y decisiones de diseño

- **WebSocket vs HTTP para audio:**
  - WebSocket te permite streaming realmente en tiempo real.
  - Si quisieras mantener 100% HTTP, podrías mandar chunks de audio a un endpoint, pero es mucho menos natural que un WebSocket.

- **Formato de audio:**
  - Lo ideal es que mantengas exactamente lo que ya generas en tu app (por ejemplo, los mismos PCM/Opus que ahora mandas a GEMINI).
  - Solo cambia la “capa de transporte”: antes HTTP/SDK Gemini, ahora WebSocket propio.

- **Latencia:**
  - Define si quieres:
    - Inferencia por bloques (cada X ms de audio).
    - O solo al final (cuando recibes `end_of_stream`).

- **GPU/Jetson:**
  - La guía se centra en la **API**; el uso de GPU depende de cómo cargues el modelo:
    - En RTX: PyTorch + CUDA, ONNX Runtime con CUDA, TensorRT.
    - En Jetson: PyTorch Jetson, TensorRT, etc.

---

## 8. Siguientes pasos recomendados

1. Crear un pequeño repo para este backend (FastAPI).
2. Implementar exactamente las rutas:
   - `ws/audio` con el protocolo descrito.
   - `POST /vision/frame` o `/vision/frame_b64`.
3. Añadir logs para ver:
   - Longitud de los buffers de audio.
   - Frecuencia de llegada de fotos.
4. Implementar la integración con un modelo sencillo primero:
   - Audio: un modelo de ASR pequeño.
   - Visión: un modelo de clasificación simple.
5. Una vez que el protocolo y el flujo estén estables:
   - Sustituir GEMINI en tu app por llamadas a este backend, cambiando sólo:
     - URL.
     - Protocolo (HTTP→WebSocket para audio).
     - Formato de imagen (probablemente puedes mantener base64 si ya lo usas para Gemini).

Si quieres, en el siguiente paso podemos bajar esto a algo aún más específico: por ejemplo, diseñar el **payload exacto** que hoy le mandas a GEMINI (desde tu código Flutter) y hacer que este backend acepte exactamente ese formato para que el cambio sea prácticamente solo de URL.

## 9. Modelo de visión local recomendado (Qwen2.5-VL)

- Para la parte de visión, esta guía asume un VLM local:
  - `Qwen/Qwen2.5-VL-3B-Instruct` corriendo en una RTX 5070 12GB en `bfloat16` sobre CUDA.
- Este modelo se usa como motor de análisis para `/vision/frame` y `/vision/frame_b64`:
  - Describe lo que ocurre en la escena (personas, objetos, entorno, actividades).
  - Señala posibles situaciones de peligro o riesgo para quien toma la foto.
  - Propone sugerencias o recomendaciones contextuales cuando sea relevante.

Ejemplo de integración conceptual con el endpoint de visión (suponiendo que ya tienes un `PIL.Image`):

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
vl_processor = AutoProcessor.from_pretrained(MODEL_ID)


def run_vision_model(image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Actúa como asistente visual experto. "
                        "Describe con detalle qué se ve en la imagen: personas, objetos, "
                        "entorno y actividades relevantes. Indica si ves algo que pueda "
                        "ser peligroso o extraño y explica por qué. Si corresponde, "
                        "sugiere acciones o recomendaciones útiles para la persona que "
                        "toma la foto. Responde en español."
                    ),
                },
            ],
        }
    ]

    text = vl_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vl_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = vl_model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = vl_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()
```

Esta función se conecta directamente con el `# TODO: run_vision_model(image)` de los endpoints `/vision/frame` y `/vision/frame_b64`.

## 10. Modelo de audio y análisis de sesiones de conversación

Para el audio, el objetivo es:

- Recibir una sesión completa de audio por WebSocket.
- Transcribirla con buena calidad en español.
- Separar hablantes cuando sea posible.
- Generar un resumen de la conversación y una lista de pasos o tareas sugeridas.

### 10.1 Pipeline recomendado

- ASR:
  - `faster-whisper` con modelo `large-v3`, ejecutado con CTranslate2 en GPU.
  - Alternativamente modelos más pequeños (`medium`, `small`) si quieres menos consumo.
- Diarización:
  - Uso de un pipeline de diarización local (por ejemplo NVIDIA NeMo o WhisperX) para asignar hablantes a segmentos.
- LLM para análisis:
  - Un modelo de lenguaje instructivo de tamaño medio cuantizado a 4 bits:
    - `Qwen2.5-7B-Instruct` en formato GGUF.
    - O modelos similares (Mistral 7B, Llama 3.x 8B) siempre en 4-bit para encajar en 12GB.

El flujo completo por sesión es:

1. El cliente envía audio a `/ws/audio` con el protocolo definido:
   - Mensaje `config`.
   - Chunks de audio.
   - Mensaje `end_of_stream`.
2. El servidor acumula el audio en `AudioSession.audio_buffer`.
3. Al recibir `end_of_stream`:
   - Ejecuta ASR con `faster-whisper` sobre el buffer completo.
   - Opcionalmente aplica diarización para separar hablantes.
   - Construye un prompt estructurado y llama al LLM local para:
     - Resumir la conversación.
     - Extraer decisiones y dudas.
     - Generar una lista de tareas recomendadas.
4. Devuelve por WebSocket (o por un endpoint HTTP de consulta) un JSON con el análisis de la sesión.

### 10.2 Forma de respuesta sugerida

Un posible formato de respuesta para el análisis de sesión de audio:

```json
{
  "session_id": "1234-5678",
  "language": "es",
  "transcript": [
    {
      "speaker": "S1",
      "start": 0.0,
      "end": 12.3,
      "text": "Texto transcrito del primer segmento..."
    }
  ],
  "summary": "Resumen corto de la conversación en lenguaje natural.",
  "action_items": [
    {
      "title": "Implementar nueva funcionalidad X en la app",
      "description": "Detalle de lo acordado, riesgos y contexto.",
      "steps": [
        "Paso 1: analizar el módulo actual.",
        "Paso 2: diseñar la API.",
        "Paso 3: implementar y probar."
      ]
    }
  ],
  "risks": [
    "Dependencia de módulo externo sin tests.",
    "Tiempo estimado ajustado para la complejidad."
  ]
}
```

Este JSON es fácil de consumir desde tu app móvil o desde otros servicios y permite explotar al máximo el análisis posterior a cada sesión de audio.
