from dataclasses import asdict, dataclass
from typing import Optional
import io
import wave
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import AudioSessionAnalysis
import os
from app.infrastructure.audio.dummy_pipeline import DummyConversationAnalysisGateway
from app.infrastructure.audio.faster_whisper_gateway import FasterWhisperASRGateway
from app.infrastructure.audio.llm_gateway import TransformersLLMConversationGateway

logger = logging.getLogger(__name__)


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
    logger.info("audio websocket accepted connection")
    # let the client know we're ready to receive audio
    try:
        await websocket.send_json({"type": "ready"})
    except Exception:
        logger.warning("unable to send ready message")

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("websocket_disconnect exception raised")
                break

            logger.debug("received ws message: %s", message)
            # log any unexpected message fields for diagnostics
            if not ("text" in message or "bytes" in message):
                logger.warning("unhandled message type: %s", message)

            # proper handling of disconnect events avoids the runtime error
            # seen in tests when the client closes immediately after sending.
            if message.get("type") == "websocket.disconnect":
                logger.info("received explicit disconnect message")
                break

            if "text" in message:
                import json

                data = json.loads(message["text"])
                msg_type = data.get("type")
                logger.debug("parsed text message type: %s", msg_type)

                if msg_type == "config":
                    session = AudioSession.create(
                        session_id=data.get("session_id", "unknown"),
                        sample_rate=data.get("sample_rate", 16000),
                        encoding=data.get("encoding", "pcm16"),
                        language=data.get("language", "es"),
                    )
                    logger.info("created new audio session %s", session.session_id)

                if msg_type == "end_of_stream":
                    if session is None:
                        logger.warning("end_of_stream received but no session configured")
                    else:
                        logger.info("end_of_stream received, processing audio")
                        payload = bytes(session.audio_buffer)
                        if not payload:
                            logger.warning("end_of_stream but audio buffer is empty")
                        if len(payload) < 2:
                            logger.warning("audio buffer too small (%d bytes), skipping ASR", len(payload))
                        if session.encoding.lower() == "pcm16" and len(payload) >= 2:
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
                    logger.info("audio analysis complete for %s", session.session_id)
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
                logger.info("received %d bytes of audio, buffer now %d bytes",
                            len(raw_bytes), len(session.audio_buffer))

    except Exception as exc:  # catch anything unexpected and log
        logger.exception("unexpected error in audio websocket: %s", exc)
    finally:
        logger.info("audio websocket closing")
