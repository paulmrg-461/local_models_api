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
    # Forzamos print para que el usuario vea output inmediato en la consola
    print(">>> AUDIO WEBSOCKET: Conexión aceptada")
    logger.info("audio websocket accepted connection")
    
    # send ready message
    try:
        print(">>> AUDIO WEBSOCKET: Enviando mensaje 'ready'...")
        await websocket.send_json({"type": "ready"})
        print(">>> AUDIO WEBSOCKET: Mensaje 'ready' enviado")
    except Exception as e:
        print(f">>> AUDIO WEBSOCKET: Error al enviar 'ready': {e}")
        logger.warning("unable to send ready message, client might have disconnected")
        return

    try:
        while True:
            try:
                # Use receive() to get the raw message event, allowing us to inspect the type
                message = await websocket.receive()
            except WebSocketDisconnect:
                print(">>> AUDIO WEBSOCKET: Cliente desconectado (WebSocketDisconnect)")
                logger.info("client disconnected (WebSocketDisconnect)")
                break
            except Exception as e:
                print(f">>> AUDIO WEBSOCKET: Error recibiendo mensaje: {e}")
                logger.warning("error receiving message: %s", e)
                break

            msg_type_event = message.get("type")

            if msg_type_event == "websocket.disconnect":
                print(">>> AUDIO WEBSOCKET: Recibido evento de desconexión")
                logger.info("received websocket.disconnect event")
                break
            
            if "text" in message:
                try:
                    import json
                    data = json.loads(message["text"])
                    client_msg_type = data.get("type")
                    print(f">>> AUDIO WEBSOCKET: Recibido mensaje de texto tipo '{client_msg_type}'")
                    
                    if client_msg_type == "config":
                        session = AudioSession.create(
                            session_id=data.get("session_id", "unknown"),
                            sample_rate=data.get("sample_rate", 16000),
                            encoding=data.get("encoding", "pcm16"),
                            language=data.get("language", "es"),
                        )
                        print(f">>> AUDIO WEBSOCKET: Sesión configurada: {session.session_id}")
                        logger.info("configured session: %s", session.session_id)

                    elif client_msg_type == "end_of_stream":
                        if session:
                            print(f">>> AUDIO WEBSOCKET: Fin de stream recibido. Buffer: {len(session.audio_buffer)} bytes")
                            logger.info("end_of_stream received. buffer size: %d", len(session.audio_buffer))
                            
                            # Prepare audio bytes (PCM16 -> WAV if needed)
                            payload = bytes(session.audio_buffer)
                            
                            if session.encoding.lower() == "pcm16" and len(payload) > 0:
                                buf = io.BytesIO()
                                with wave.open(buf, "wb") as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(session.sample_rate)
                                    wf.writeframes(payload)
                                payload = buf.getvalue()
                            
                            # Execute analysis
                            print(">>> AUDIO WEBSOCKET: Iniciando análisis de audio...")
                            analysis = use_case.execute(
                                session_id=session.session_id,
                                language=session.language,
                                audio_bytes=payload,
                                filename=f"stream_{session.session_id}.wav" # Default to wav for streaming
                            )
                            print(">>> AUDIO WEBSOCKET: Análisis completado. Enviando resultado.")
                            
                            # Send result
                            await websocket.send_json({
                                "type": "final_result",
                                "analysis": asdict(analysis),
                            })
                            
                            # Clear buffer
                            session.audio_buffer = bytearray()
                        else:
                            print(">>> AUDIO WEBSOCKET: Fin de stream pero sin sesión configurada")
                            logger.warning("end_of_stream received but no session configured")
                
                except json.JSONDecodeError:
                    print(">>> AUDIO WEBSOCKET: Error decodificando JSON")
                    logger.warning("received invalid json text")
                except Exception as e:
                    print(f">>> AUDIO WEBSOCKET: Error procesando texto: {e}")
                    logger.exception("error processing text message: %s", e)

            elif "bytes" in message:
                if session:
                    chunk_size = len(message["bytes"])
                    # Solo imprimimos cada 10 chunks o si es grande para no saturar, pero el usuario quiere ver ALGO
                    # Mejor imprimimos siempre un puntito o un mensaje corto
                    # print(f">>> AUDIO: Recibidos {chunk_size} bytes") 
                    session.add_bytes(message["bytes"])
                else:
                    print(">>> AUDIO WEBSOCKET: Recibidos bytes sin sesión configurada (ignorado)")
                    # Client sending bytes before config? ignore or log warning
                    pass
            
    except Exception as exc:
        print(f">>> AUDIO WEBSOCKET: Error inesperado en bucle principal: {exc}")
        logger.exception("unexpected error in audio websocket loop: %s", exc)
    finally:
        print(">>> AUDIO WEBSOCKET: Cerrando conexión")
        logger.info("audio websocket closing")
