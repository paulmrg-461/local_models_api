# Vision Feature – Local Qwen2.5-VL Integration

This document describes the vision feature of the Local Models API: what it does, how it is implemented using Qwen2.5-VL, and how to consume the available endpoints.

---

## Purpose

The vision feature provides a local, GPU-accelerated visual assistant that:

- Analyzes images sent from a client (for example, smart glasses or a mobile app).
- Describes in detail what is happening in the scene:
  - People, objects, environment, and relevant activities.
- Identifies possible danger or unusual situations for the person taking the picture.
- Suggests useful actions or recommendations when appropriate.

All processing happens locally on your hardware (RTX 5070 12GB or similar), without sending data to external cloud services.

---

## High-Level Architecture

The implementation follows a clean architecture approach with clear separation of concerns:

- **Domain layer**
  - Defines the abstraction for the vision model:
    - `VisionAnalysisResult`
    - `VisionModelGateway`
- **Application layer**
  - Implements the use case:
    - `AnalyzeImageUseCase`
  - This use case orchestrates the call to the vision model gateway.
- **Infrastructure layer**
  - Provides a concrete implementation of the vision model gateway using Qwen2.5-VL:
    - `QwenVisionModel`
- **API layer (FastAPI)**
  - Exposes HTTP endpoints to the client:
    - `POST /vision/frame` for multipart image uploads.
    - `POST /vision/frame_b64` for base64-encoded images in JSON.

---

## Implementation Details

### Domain Layer

**File:** `app/domain/vision/interfaces.py`

- `VisionAnalysisResult`
  - Simple data structure containing:
    - `description: str`
- `VisionModelGateway`
  - Abstract base class with a single method:
    - `analyze(image: Image) -> VisionAnalysisResult`
  - Any concrete vision model must implement this interface.

### Application Layer

**File:** `app/application/vision/use_cases.py`

- `AnalyzeImageUseCase`
  - Constructor receives a `VisionModelGateway` implementation.
  - Method `execute(image)`:
    - Validates that the image is not `None`.
    - Delegates to `model_gateway.analyze(image)`.
    - Returns a `VisionAnalysisResult`.

This layer contains no framework-specific code and does not depend on Qwen directly.

### Infrastructure Layer – Qwen2.5-VL

**File:** `app/infrastructure/vision/qwen_service.py`

- `QwenVisionModel(VisionModelGateway)`
  - Loads the Qwen2.5-VL model and processor:
    - `Qwen/Qwen2.5-VL-3B-Instruct`
    - Uses `torch.bfloat16` and `device_map="cuda"` to run on GPU.
  - Implements `analyze(image)`:
    - Builds a Qwen-style `messages` structure with:
      - The input image.
      - A prompt that:
        - Asks for a detailed scene description.
        - Asks the model to highlight dangerous or unusual elements.
        - Requests suggestions or recommendations.
        - Requests the answer in Spanish.
    - Uses `AutoProcessor` and `process_vision_info` to prepare tensors.
    - Calls `model.generate(max_new_tokens=256)` with `torch.no_grad()`.
    - Decodes the generated tokens and returns the final description as `VisionAnalysisResult`.

### API Layer – FastAPI

**File:** `app/api/vision_routes.py`

- Defines a FastAPI `APIRouter` with two endpoints:

1. `POST /vision/frame`
   - Accepts `multipart/form-data`:
     - Field `file`: image file (JPEG/PNG).
     - Optional field `session_id`.
   - Creates a `PIL.Image` from the uploaded bytes.
   - Calls `AnalyzeImageUseCase.execute(image)`.
   - Returns JSON with:
     - `description`
     - `session_id`
     - `width`
     - `height`

2. `POST /vision/frame_b64`
   - Accepts JSON body of type `VisionFrameB64Request`:

     ```json
     {
       "image_b64": "BASE64_IMAGE_HERE",
       "session_id": "optional-session-id"
     }
     ```

   - Decodes `image_b64` using `base64.b64decode` with validation.
   - If the base64 string is invalid, returns HTTP `400 Bad Request`.
   - Creates a `PIL.Image` from the decoded bytes.
   - Calls `AnalyzeImageUseCase.execute(image)`.
   - Returns JSON with:
     - `description`
     - `session_id`
     - `width`
     - `height`

**File:** `app/main.py`

- Creates the FastAPI application and includes the vision router.

---

## Endpoints – Request and Response

### 1. `POST /vision/frame`

Upload an image via `multipart/form-data`.

#### Request

- Method: `POST`
- Path: `/vision/frame`
- Headers:
  - `Content-Type: multipart/form-data`
- Body:
  - `file`: image file (JPEG/PNG).
  - Optional `session_id`: string.

Example with `curl`:

```bash
curl -X POST "http://localhost:8000/vision/frame" \
  -F "file=@/path/to/image.jpg" \
  -F "session_id=example-session"
```

#### Response

Status `200 OK`:

```json
{
  "description": "Detailed description in Spanish, including risks and suggestions.",
  "session_id": "example-session",
  "width": 1920,
  "height": 1080
}
```

If the image cannot be processed, FastAPI will return an appropriate error (for example, `422 Unprocessable Entity`).

---

### 2. `POST /vision/frame_b64`

Send a base64-encoded image via JSON.

#### Request

- Method: `POST`
- Path: `/vision/frame_b64`
- Headers:
  - `Content-Type: application/json`
- Body:

```json
{
  "image_b64": "BASE64_IMAGE_HERE",
  "session_id": "optional-session-id"
}
```

Example with `curl`:

```bash
IMAGE_B64=$(base64 -w0 /path/to/image.jpg)

curl -X POST "http://localhost:8000/vision/frame_b64" \
  -H "Content-Type: application/json" \
  -d "{
    \"image_b64\": \"${IMAGE_B64}\",
    \"session_id\": \"example-session\"
  }"
```

#### Response

Status `200 OK`:

```json
{
  "description": "Detailed description in Spanish, including risks and suggestions.",
  "session_id": "example-session",
  "width": 1920,
  "height": 1080
}
```

If `image_b64` is not a valid base64 string, the API returns:

```json
{
  "detail": "invalid image_b64"
}
```

with status `400 Bad Request`.

---

## Running and Testing

With your Python virtual environment activated and dependencies installed:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then:

- Call `/vision/frame` or `/vision/frame_b64` as shown in the examples above.
- Run tests:

```bash
pytest
```

The tests cover:

- The `AnalyzeImageUseCase` behavior with a fake gateway.
- The `/vision/frame` endpoint using a fake gateway.
- The `/vision/frame_b64` endpoint, including handling of invalid base64 input.

