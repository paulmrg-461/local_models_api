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

### Infrastructure Layer – Dummy Pipeline

**File:** `app/infrastructure/audio/dummy_pipeline.py`

This module provides simple, fully local implementations of the domain gateways to keep the system functional while you integrate real models later.

- `DummyASRGateway(ASRGateway)`
  - Returns a fixed list with one `TranscriptSegment`:
    - Speaker `S1`, time range `[0.0, 1.0]`, text `"dummy transcript"`.

- `DummyConversationAnalysisGateway(ConversationAnalysisGateway)`
  - Builds a fixed `AudioSessionAnalysis` using the provided `session_id`, `language`, and transcript segments:
    - `summary`: `"dummy summary"`
    - `action_items`: one dummy item with a single step.
    - `risks`: one dummy risk string.

In production, you would replace these dummy classes with:

- An ASR implementation using `faster-whisper` and possibly VAD.
- A conversation analysis implementation using a local LLM (Qwen, Mistral, Llama, etc.).

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
  - `DummyASRGateway`
  - `DummyConversationAnalysisGateway`
  - `AnalyzeAudioSessionUseCase`
- Registered as a dependency for the WebSocket handler so it can be overridden in tests or replaced in production with real gateways.

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

## Next Steps for Integrating Real Models

To move from the dummy pipeline to a real faster-whisper + LLM pipeline:

1. Implement an `ASRGateway` using `faster-whisper`:
   - Convert raw PCM bytes to the format expected by faster-whisper.
   - Use the GPU for inference.
   - Map segments to `TranscriptSegment`.
2. Implement a `ConversationAnalysisGateway` using a local LLM:
   - Build a prompt from the transcript.
   - Ask the model for summary, action items, and risks.
   - Map the response to `AudioSessionAnalysis`.
3. Update `get_analyze_audio_use_case()` to instantiate the real gateways instead of the dummy ones.
4. Optionally add configuration (environment variables) to choose between dummy and real pipelines at runtime.

