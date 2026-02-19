## Local Models API – FastAPI Backend for Audio and Vision

This service exposes a local AI backend running on your own hardware (RTX 5070 12GB or Jetson). It is designed to replace cloud multimodal APIs (such as Gemini) with:

- Real‑time audio ingestion over WebSocket.
- Periodic image uploads over HTTP.
- Local vision model (VLM) for scene understanding and safety awareness.
- Local ASR + LLM pipeline to analyze full audio sessions, summarize them, and propose action items.

The goal is to keep the client app almost unchanged: you keep sending raw audio and images, but change the URL and transport (HTTP → WebSocket for audio).

---

## Features

- FastAPI backend.
- WebSocket endpoint `/ws/audio` for streaming audio.
- HTTP endpoints `/vision/frame` and `/vision/frame_b64` for image analysis.
- Local VLM for visual understanding and risk detection:
  - `Qwen/Qwen2.5-VL-3B-Instruct` on GPU.
- Local audio understanding pipeline:
  - ASR with `faster-whisper` (`large-v3` recommended).
  - Optional speaker diarization.
  - Local LLM (e.g. `Qwen2.5-7B-Instruct` 4‑bit) to summarize sessions and suggest tasks.

---

## Architecture Overview

- **Hardware**
  - Primary: PC with NVIDIA RTX 5070 12GB.
  - Optional: Jetson Nano / Orin for embedded scenarios (use smaller models).

- **Server**
  - Python + FastAPI.
  - Uvicorn as ASGI server.

- **Endpoints**
  - `WebSocket /ws/audio`
    - Receives streaming audio from the mobile app.
    - Accumulates a session buffer.
    - After end of stream, runs ASR + analysis.
  - `POST /vision/frame`
    - Receives images via `multipart/form-data`.
  - `POST /vision/frame_b64`
    - Receives images encoded as base64 in JSON.

The client application:

- Opens a WebSocket to `/ws/audio` and sends config + audio chunks.
- Periodically sends photos to `/vision/frame` or `/vision/frame_b64`.

---

## Setup

### System Requirements

- Ubuntu/Debian recommended on the server.
- Python 3.10+ (or compatible).
- NVIDIA drivers and CUDA properly installed for GPU acceleration.

### Python Environment

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

python3 -m venv venv
source venv/bin/activate

pip install "fastapi[standard]" uvicorn[standard]
```

You will also need to install the libraries required by your chosen models, for example:

- PyTorch with CUDA (for Qwen VLM and some ASR variants).
- CTranslate2 and faster-whisper for efficient ASR.
- Transformers and the Qwen VL utilities for vision.

Refer to each model’s documentation for exact installation commands.

---

## API Design

### Audio WebSocket – `/ws/audio`

Client connects to:

```text
ws://SERVER_HOST:PORT/ws/audio
```

#### Client → Server Messages

1. **Config message (JSON text):**

```json
{
  "type": "config",
  "session_id": "1234-5678",
  "sample_rate": 16000,
  "encoding": "pcm16",
  "language": "es"
}
```

2. **Audio frames**

- Recommended: binary WebSocket messages with raw audio bytes (PCM/Opus).
- Alternative: JSON text messages with base64-encoded audio.

Example (base64 JSON):

```json
{
  "type": "audio_chunk",
  "chunk_id": 1,
  "data_b64": "BASE64_AUDIO_HERE"
}
```

3. **End of stream**

```json
{
  "type": "end_of_stream"
}
```

#### Server → Client Messages

The server can send partial or final results, for example:

```json
{
  "type": "partial_result",
  "text": "Current partial transcription or analysis..."
}
```

```json
{
  "type": "final_result",
  "text": "Final transcription or session summary"
}
```

Internally, after receiving `end_of_stream`, the backend runs the ASR + LLM pipeline described below.

---

### Vision HTTP – `/vision/frame` and `/vision/frame_b64`

#### `POST /vision/frame` (multipart/form-data)

- Request:
  - Field `file`: JPEG/PNG image.
  - Optional fields such as `session_id`.

- Server:
  - Reads the uploaded image.
  - Converts it to RGB (`PIL.Image`).
  - Passes it to the local VLM (`run_vision_model(image)`).
  - Returns a JSON response with a description and any detected risks or suggestions.

#### `POST /vision/frame_b64` (JSON + base64)

Example request:

```json
{
  "session_id": "1234-5678",
  "image_b64": "BASE64_IMAGE_HERE",
  "metadata": {
    "timestamp": 1234567890
  }
}
```

The backend decodes the image, runs the same VLM pipeline, and returns a JSON description.

---

## Vision Model – Qwen2.5-VL

For visual understanding, the backend uses a local VLM:

- `Qwen/Qwen2.5-VL-3B-Instruct`
  - Loaded on the RTX 5070 12GB using `bfloat16` on CUDA.

The model is used as the engine behind `/vision/frame` and `/vision/frame_b64` to:

- Describe what is happening in the scene (people, objects, environment, activities).
- Point out possible danger or risk for the person taking the photo.
- Provide contextual suggestions or recommendations when appropriate.

Conceptual example of the vision function:

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
                        "Act as an expert visual assistant. "
                        "Describe in detail what you see: people, objects, "
                        "environment, and relevant activities. Indicate if "
                        "anything looks dangerous or unusual and explain why. "
                        "If appropriate, suggest useful actions or recommendations "
                        "for the person taking the picture. Answer in Spanish."
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

You can wire this function into the `/vision/frame` and `/vision/frame_b64` endpoints.

---

## Audio Pipeline – Transcription and Session Analysis

For audio, the goal is to:

- Receive an entire audio session over WebSocket.
- Transcribe it with good Spanish quality.
- Optionally separate speakers.
- Generate a summary and a list of suggested tasks or next steps.

### Recommended Components

- **ASR**
  - `faster-whisper` with model `large-v3` running via CTranslate2 on GPU.
  - Smaller models (`medium`, `small`) are possible if you need to reduce memory or latency.

- **Diarization**
  - A local diarization pipeline (e.g. NVIDIA NeMo or WhisperX) to assign speakers to segments.

- **LLM for analysis**
  - A mid-size instruction-tuned model quantized to 4 bits, for example:
    - `Qwen2.5-7B-Instruct` in GGUF format.
    - Or alternatives like Mistral 7B, Llama 3.x 8B in 4‑bit.

### Session Flow

1. Client sends:
   - Config message on `/ws/audio`.
   - Binary or base64 audio chunks.
   - `end_of_stream` message.
2. Server accumulates audio in an `AudioSession` buffer.
3. On `end_of_stream`:
   - Run ASR over the full buffer.
   - Optionally run diarization to identify speakers.
   - Build a structured prompt and call the local LLM to:
     - Summarize the conversation.
     - Extract decisions and open questions.
     - Generate recommended tasks or action items.
4. Return a JSON object with the full analysis via WebSocket or a separate HTTP endpoint.

### Suggested Response JSON

Example of the JSON structure for an analyzed audio session:

```json
{
  "session_id": "1234-5678",
  "language": "es",
  "transcript": [
    {
      "speaker": "S1",
      "start": 0.0,
      "end": 12.3,
      "text": "Transcript of the first segment..."
    }
  ],
  "summary": "Short natural language summary of the session.",
  "action_items": [
    {
      "title": "Implement new feature X in the app",
      "description": "Details of what was agreed, risks, and context.",
      "steps": [
        "Step 1: analyze the current module.",
        "Step 2: design the API.",
        "Step 3: implement and test."
      ]
    }
  ],
  "risks": [
    "Dependency on an external module without tests.",
    "Timeline is tight for the described complexity."
  ]
}
```

This format is easy to consume from your mobile app or other services and lets you fully exploit post‑session analysis.

---

## Running the Server

From your virtual environment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On the mobile client (same network), you then point to:

- `ws://SERVER_IP:8000/ws/audio`
- `http://SERVER_IP:8000/vision/frame`
- `http://SERVER_IP:8000/vision/frame_b64`

---

## Next Steps

- Implement the FastAPI routes:
  - `/ws/audio` with the described protocol.
  - `POST /vision/frame` and/or `/vision/frame_b64`.
- Integrate:
  - Qwen2.5-VL for vision.
  - faster-whisper + diarization + local LLM for audio session analysis.
- Add logging and metrics to monitor:
  - Audio buffer size and processing times.
  - Frequency and latency of vision calls.
- Replace cloud multimodal calls in your client with this local backend, changing primarily:
  - URL.
  - Audio transport mechanism (HTTP → WebSocket).
  - Image format if needed (you can keep base64 if you already use it).

---

## Quick Install

- Create and activate a virtual environment:
  - Linux/macOS:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
  - Windows (PowerShell):
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
- Install project dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- GPU requirements:
  - Ensure NVIDIA drivers and CUDA are correctly installed.
  - Install a compatible PyTorch build with CUDA per official instructions.
  - Validate with:
    ```python
    import torch; print(torch.cuda.is_available())
    ```

---

## What’s Available

- Vision endpoints:
  - `POST /vision/frame` (multipart/form-data)
  - `POST /vision/frame_b64` (JSON + base64)
  - Backed by a local VLM: Qwen2.5‑VL‑3B on GPU.
- Audio WebSocket:
  - `ws://HOST:PORT/ws/audio`
  - Receives config + audio chunks, returns a final session analysis JSON.
  - Pipeline is structured to plug in faster‑whisper and a local LLM.

See detailed feature docs:
- Vision: [vision_docs.md](file:///c:/Users/paulm/Documents/dev-projects/IAAplicada/OMIGlasses/local_models_api/vision_docs.md)
- Audio: [audio_docs.md](file:///c:/Users/paulm/Documents/dev-projects/IAAplicada/OMIGlasses/local_models_api/audio_docs.md)

---

## Architecture

The project follows a clean architecture separation:

- Domain
  - Core models and interfaces (contracts).
- Application
  - Use cases orchestrating domain logic.
- Infrastructure
  - Concrete adapters to models and external systems.
- API
  - FastAPI routers and transport protocols (HTTP/WebSocket).

This keeps use cases independent from specific model implementations and makes it easy to swap gateways (e.g., dummy → faster‑whisper).

---

## Project Structure

- API
  - `app/api/vision_routes.py`
  - `app/api/audio_ws.py`
- Application (use cases)
  - `app/application/vision/use_cases.py`
  - `app/application/audio/use_cases.py`
- Domain (interfaces and data models)
  - `app/domain/vision/interfaces.py`
  - `app/domain/audio/interfaces.py`
- Infrastructure (concrete implementations)
  - `app/infrastructure/vision/qwen_service.py`
  - `app/infrastructure/audio/dummy_pipeline.py`
- Entry point
  - `app/main.py`
- Docs
  - [vision_docs.md](file:///c:/Users/paulm/Documents/dev-projects/IAAplicada/OMIGlasses/local_models_api/vision_docs.md)
  - [audio_docs.md](file:///c:/Users/paulm/Documents/dev-projects/IAAplicada/OMIGlasses/local_models_api/audio_docs.md)

---

## Development Workflow

- Create a venv and install dependencies (`requirements.txt`).
- Run the server with:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- Run tests:
  ```bash
  pytest
  ```
- Vision model setup:
  - Install PyTorch with CUDA, `transformers`, and Qwen VL utilities.
  - The vision gateway loads `Qwen/Qwen2.5-VL-3B-Instruct` to GPU.
- Audio pipeline:
  - Current: dummy ASR + dummy LLM analysis for functional API and tests.
  - Next: replace with faster‑whisper (ASR) and a local LLM gateway.

---

## Notes for New Contributors

- Keep changes within the appropriate layer (Domain/Application/Infrastructure/API).
- Add unit tests for use cases and integration tests for endpoints.
- For GPU inference, prefer batch‑free small requests initially and measure latency.
- Document new features in dedicated MD files and reference them from the README.

---

## Developed by:
- Paul Realpe
- Email: co.devpaul@gmail.com
- Phone: 3043162313
- <a href="https://devpaul.pro">https://devpaul.pro/</a>