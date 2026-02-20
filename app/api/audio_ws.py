from dataclasses import asdict, dataclass
from typing import Optional
import io
import wave

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import AudioSessionAnalysis
import os
from app.infrastructure.audio.dummy_pipeline import DummyConversationAnalysisGateway
from app.infrastructure.audio.faster_whisper_gateway import FasterWhisperASRGateway
from app.infrastructure.audio.llm_gateway import TransformersLLMConversationGateway


router = APIRouter()


@dataclass
class AudioSession:
    session_id: str
    sample_rate: int
    encoding: str
    language: str
    audio_buffer: bytearray

    @classmethod
    def create(cls, session_id: str, sample_rate: int, encoding: str, language: str):
        return cls(
            session_id=session_id,
            sample_rate=sample_rate,
            encoding=encoding,
            language=language,
            audio_buffer=bytearray(),
        )

    def add_bytes(self, data: bytes) -> None:
        self.audio_buffer.extend(data)


def get_analyze_audio_use_case() -> AnalyzeAudioSessionUseCase:
    asr_gateway = FasterWhisperASRGateway()

    use_real_llm = os.getenv("CONV_LLM_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    if use_real_llm:
        conversation_gateway = TransformersLLMConversationGateway()
    else:
        conversation_gateway = DummyConversationAnalysisGateway()

    return AnalyzeAudioSessionUseCase(asr_gateway, conversation_gateway)


@router.websocket("/ws/audio")
async def websocket_audio(
    websocket: WebSocket,
    use_case: AnalyzeAudioSessionUseCase = Depends(get_analyze_audio_use_case),
) -> None:
    await websocket.accept()
    session: Optional[AudioSession] = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                import json

                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    session = AudioSession.create(
                        session_id=data.get("session_id", "unknown"),
                        sample_rate=data.get("sample_rate", 16000),
                        encoding=data.get("encoding", "pcm16"),
                        language=data.get("language", "es"),
                    )

                if msg_type == "end_of_stream" and session is not None:
                    payload = bytes(session.audio_buffer)
                    if session.encoding.lower() == "pcm16":
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(session.sample_rate)
                            wf.writeframes(payload)
                        payload = buf.getvalue()

                    analysis: AudioSessionAnalysis = use_case.execute(
                        session_id=session.session_id,
                        language=session.language,
                        audio_bytes=payload,
                    )
                    await websocket.send_json(
                        {
                            "type": "final_result",
                            "analysis": asdict(analysis),
                        }
                    )
                    session.audio_buffer = bytearray()

            if "bytes" in message and session is not None:
                raw_bytes = message["bytes"]
                session.add_bytes(raw_bytes)

    except WebSocketDisconnect:
        return
