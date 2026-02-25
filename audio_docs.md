# Audio Feature – WebSocket Session Analysis with ASR and LLM

This document describes the audio feature of the Local Models API: what it does, how it is structured using clean architecture, and how to consume the `/ws/audio` WebSocket endpoint.

---

## Purpose

The audio feature enables the backend to:

- Receive an entire audio session over a WebSocket connection from a client device.
- Transcribe the audio with good quality (Spanish-focused).
- Optionally support speaker separation (diarization) in future implementations.
- Generate:
  - A structured transcript.
  - A natural language summary of the conversation.
  - A list of action items or next steps.
  - A list of risks or concerns mentioned during the session.

The design prepares the pipeline to plug in:

- `faster-whisper` for ASR.
- A local LLM (for example, `Qwen2.5-7B-Instruct` 4-bit) for conversation analysis.

The current implementation uses a dummy pipeline for ASR and LLM so that the API is functional and testable. You can later replace the dummy implementations with real models.

---

## High-Level Architecture

The audio feature is implemented with the same clean architecture principles as the vision feature:

- **Domain layer**
  - Core data structures and interfaces for audio analysis.
- **Application layer**
  - Use case orchestrating ASR and LLM.
- **Infrastructure layer**
  - Concrete gateways (currently dummy) for ASR and conversation analysis.
- **API layer**
  - WebSocket endpoint `/ws/audio` handling the protocol and delegating to the use case.

---

## Implementation Details

### Domain Layer

**File:** `app/domain/audio/interfaces.py`

- `TranscriptSegment`
  - Represents a chunk of transcribed audio:
    - `speaker: str`
    - `start: float`
    - `end: float`
    - `text: str`

- `ActionItem`
  - Represents a suggested action:
    - `title: str`
    - `description: str`
    - `steps: list[str]`

- `AudioSessionAnalysis`
  - Represents the full analysis of an audio session:
    - `session_id: str`
    - `language: str`
    - `transcript: list[TranscriptSegment]`
    - `summary: str`
    - `action_items: list[ActionItem]`
    - `risks: list[str]`

- `ASRGateway`
  - Abstract interface for automatic speech recognition:
    - `transcribe(audio_bytes: bytes, language: str) -> list[TranscriptSegment]`

- `ConversationAnalysisGateway`
  - Abstract interface for the LLM-based conversation analysis:
    - `analyze(session_id: str, language: str, segments: list[TranscriptSegment]) -> AudioSessionAnalysis`

These abstractions isolate the core business logic from any specific ASR or LLM implementation.

### Application Layer

**File:** `app/application/audio/use_cases.py`

- `AnalyzeAudioSessionUseCase`
  - Constructor:
    - Receives an `ASRGateway` and a `ConversationAnalysisGateway`.
  - Method `execute(session_id, language, audio_bytes)`:
    - Calls `asr_gateway.transcribe(audio_bytes, language)` to obtain transcript segments.
    - Calls `conversation_gateway.analyze(session_id, language, segments)` to obtain the full `AudioSessionAnalysis`.
    - Returns the resulting analysis.

This use case encapsulates the orchestration logic and depends only on the domain interfaces.

### Infrastructure Layer – ASR and Conversation Pipelines

**File:** `app/infrastructure/audio/faster_whisper_gateway.py`

- `FasterWhisperASRGateway(ASRGateway)`
  - Provides a real ASR implementation based on `faster-whisper`.
  - Configuration:
    - Model ID:
      - `FWHISPER_MODEL_ID` environment variable (default: `large-v3`).
    - Device:
      - `FWHISPER_DEVICE` environment variable (default: `cuda`).
    - Compute type:
      - `FWHISPER_COMPUTE_TYPE` environment variable (default: `float16`).
  - Behavior:
    - Receives raw audio bytes for the whole session.
    - Writes bytes to a temporary `.wav` file.
    - Calls `WhisperModel.transcribe` on that file path with the requested language.
    - Maps returned segments to `TranscriptSegment` instances:
      - `speaker` is set to `"S1"` as a placeholder.
      - `start`, `end`, and `text` are taken from the model segments.

**File:** `app/infrastructure/audio/dummy_pipeline.py`

- `DummyASRGateway(ASRGateway)`
  - Simple ASR stub used for tests and as a fallback example.
- `DummyConversationAnalysisGateway(ConversationAnalysisGateway)`
  - Simple conversation-analysis stub that builds `AudioSessionAnalysis` with dummy values.

In the current wiring:

- The WebSocket endpoint uses `FasterWhisperASRGateway` for ASR.
- The conversation analysis still uses `DummyConversationAnalysisGateway` until a real LLM gateway is added.

### API Layer – WebSocket `/ws/audio`

**File:** `app/api/audio_ws.py`

- Defines a FastAPI `APIRouter` and a WebSocket endpoint:

#### `AudioSession` dataclass

- Holds session state per WebSocket connection:
  - `session_id: str`
  - `sample_rate: int`
  - `encoding: str`
  - `language: str`
  - `audio_buffer: bytearray`
- Methods:
  - `create(session_id, sample_rate, encoding, language)` to build a new session.
  - `add_bytes(data)` to append raw audio bytes.

#### Dependency – `get_analyze_audio_use_case()`

- Instantiates:
  - `FasterWhisperASRGateway` for ASR.
  - `DummyConversationAnalysisGateway` for conversation analysis (to be replaced by a real LLM).
  - `AnalyzeAudioSessionUseCase`.
- Registered as a dependency for the WebSocket handler so it can be overridden in tests or swapped for different gateway implementations.

#### WebSocket Endpoint – `/ws/audio`

- Path: `/ws/audio`
- Protocol:

Client connects to:

```text
ws://SERVER_HOST:PORT/ws/audio
```

1. Client sends a JSON config message:

```json
{
  "type": "config",
  "session_id": "1234-5678",
  "sample_rate": 16000,
  "encoding": "pcm16",
  "language": "es"
}
```

2. Client sends binary audio chunks:

- Binary WebSocket frames containing raw audio bytes (e.g. PCM16).

3. Client sends an end-of-stream message:

```json
{
  "type": "end_of_stream"
}
```

- On `config`, the server initializes an `AudioSession` instance.
- On binary messages, it appends data to `audio_buffer`.
- On `end_of_stream`, the server:
  - Calls `AnalyzeAudioSessionUseCase.execute(session_id, language, audio_bytes)`.
  - Sends a final JSON message:

```json
{
  "type": "final_result",
  "analysis": {
    "session_id": "1234-5678",
    "language": "es",
    "transcript": [
      {
        "speaker": "S1",
        "start": 0.0,
        "end": 1.0,
        "text": "dummy transcript"
      }
    ],
    "summary": "dummy summary",
    "action_items": [
      {
        "title": "dummy action",
        "description": "dummy description",
        "steps": [
          "dummy step"
        ]
      }
    ],
    "risks": [
      "dummy risk"
    ]
  }
}
```

In a production setup, the content of `analysis` will be generated by real ASR and LLM implementations, but the structure will remain the same.

---

## How to Consume `/ws/audio`

Example client flow (pseudo-steps):

1. Open a WebSocket connection to `ws://SERVER_HOST:PORT/ws/audio`.
2. Send a JSON `config` message with session metadata and audio format.
3. Stream audio frames as binary messages.
4. When the session finishes, send `{"type": "end_of_stream"}` as JSON.
5. Wait for a `final_result` message and process the `analysis` payload.

This pattern works well for mobile or wearable devices streaming microphone audio in real time.

### Flutter integration notes

If you are writing a Flutter frontend, follow this sequence carefully to avoid
errors such as "No final_result from local audio backend":

1. Establish the socket with the **WS scheme** (`ws://` or `wss://`), not
   `http://`.

```dart
final channel = WebSocketChannel.connect(
    Uri.parse('ws://192.168.0.6:8989/ws/audio'));
```

2. **Listen for the server greeting**. The backend sends a
   `{'type': 'ready'}` JSON message immediately after accepting the
   connection; wait for it before transmitting anything else.

3. Send configuration as JSON once ready:

```dart
channel.sink.add(jsonEncode({
  'type': 'config',
  'session_id': sessionId,
  'sample_rate': 16000,
  'encoding': 'pcm16',
  'language': 'es',
}));
```

4. **Transmit audio as binary frames**. Each frame should be a
   `Uint8List` containing at least two bytes (one 16‑bit PCM sample). For
   example, inside a microphone callback:

```dart
void onAudioData(List<int> pcmBytes) {
  // pcmBytes must contain real audio data, not a single byte.
  channel.sink.add(Uint8List.fromList(pcmBytes));
}
```

5. After recording, notify the server that the stream has ended:

```dart
channel.sink.add(jsonEncode({'type': 'end_of_stream'}));
```

6. **Do not close the socket** until you receive the
   `final_result` message; closing early yields the "No final_result" error.

7. Handle server responses:

```dart
channel.stream.listen((message) {
  final data = jsonDecode(message);
  switch (data['type']) {
    case 'ready':
      // now safe to send config/audio
      break;
    case 'final_result':
      handleAnalysis(data['analysis']);
      break;
  }
});
```

By following these steps you ensure audio bytes reach the backend and a proper
analysis is returned. The server logs (`received X bytes...` lines) will
confirm when data has arrived.
---

## Testing

**File:** `tests/test_audio_use_case.py`

- Uses fake implementations of `ASRGateway` and `ConversationAnalysisGateway`.
- Verifies that:
  - The use case returns an `AudioSessionAnalysis` with the expected values.
  - The transcript, summary, action items, and risks are passed through correctly.

**File:** `tests/test_audio_websocket.py`

- Overrides the audio use case dependency in FastAPI with fake gateways.
- Connects to `/ws/audio` using `TestClient`:
  - Sends a config message.
  - Sends one binary chunk.
  - Sends `end_of_stream`.
  - Receives a `final_result` message and validates its structure and content.

You can run all tests with:

```bash
pytest
```

---

## Next Steps for Conversation Analysis

The current setup uses real ASR with `FasterWhisperASRGateway` and a configurable LLM-based conversation analysis gateway.

### Conversation Analysis Gateway

**File:** `app/infrastructure/audio/llm_gateway.py`

- `TransformersLLMConversationGateway(ConversationAnalysisGateway)`
  - Loads a local HF model via `transformers`:
    - Model ID: `CONV_LLM_MODEL_ID` (default: `Qwen/Qwen2.5-3B-Instruct`).
    - Device: `CONV_LLM_DEVICE` (default: `cuda`).
    - Torch dtype: `CONV_LLM_TORCH_DTYPE` (default: `bfloat16`).
    - Max tokens: `CONV_LLM_MAX_NEW_TOKENS` (default: `512`).
  - Builds a language-aware prompt from transcript segments, requesting strictly valid JSON with:
    - `summary`, `action_items` (array of `{title, description, steps}`), `risks` (array of strings).
  - Parses the model output; if not valid JSON, falls back to using the raw text as `summary`.

### Wiring

- In `app/api/audio_ws.py`, the dependency resolver:
  - Always uses `FasterWhisperASRGateway` for ASR.
  - Chooses conversation analysis based on `CONV_LLM_ENABLED`:
    - `true` → `TransformersLLMConversationGateway`.
    - Otherwise → `DummyConversationAnalysisGateway`.

### Environment Variables

- `CONV_LLM_ENABLED` = `true` to enable the real LLM gateway.
- `CONV_LLM_BACKEND` is reserved for future backends (e.g., llama.cpp); current implementation uses `TRANSFORMERS`.
- `CONV_LLM_MODEL_ID` = HF model id, e.g. `Qwen/Qwen2.5-3B-Instruct`.
- `CONV_LLM_DEVICE` = `cuda` or `cpu`.
- `CONV_LLM_TORCH_DTYPE` = `bfloat16` | `float16` | `float32`.
- `CONV_LLM_MAX_NEW_TOKENS` = integer limit for generated tokens.

Adjust these variables in your environment or load them from `example.env` before starting the server.
