from dataclasses import asdict, dataclass
from typing import Optional
import io
import wave
import logging
import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import AudioSessionAnalysis, ConversationAnalysisGateway
import os
from app.infrastructure.audio.dummy_pipeline import DummyConversationAnalysisGateway
from app.infrastructure.audio.faster_whisper_gateway import FasterWhisperASRGateway
from app.infrastructure.audio.llm_gateway import TransformersLLMConversationGateway


# Configuration constants that guard against unbounded buffering. Values
# can be overridden through environment variables so deployments can tune
# limits based on available RAM, GPU memory, or expected session lengths.
#
# - ``AUDIO_WS_MAX_PRECONFIG_BYTES``: bytes to buffer before client sends a
#   configuration message (default 5 MiB).
# - ``AUDIO_WS_MAX_SESSION_BYTES``: bytes to buffer after configuration,
#   i.e. the maximum length of a single audio session (default 50 MiB).
#
# These limits exist to prevent a misbehaving client from filling up RAM and
# crashing the process; when exceeded the connection is terminated with an
# error code.
MAX_PRECONFIG_BUFFER = int(os.getenv("AUDIO_WS_MAX_PRECONFIG_BYTES", 5 * 1024 * 1024))
MAX_SESSION_BUFFER = int(os.getenv("AUDIO_WS_MAX_SESSION_BYTES", 50 * 1024 * 1024))

logger = logging.getLogger(__name__)


router = APIRouter()

# To avoid blowing up the machine when multiple clients submit large audio
# sessions we only allow a single analysis to execute at a time.  Any
# additional requests will automatically queue by awaiting on the semaphore.
# The intent is to serialize the expensive call to ``use_case.execute`` so
# that the GPU/CPU memory footprint stays bounded.
_analysis_semaphore = asyncio.Semaphore(1)


async def _serialize_analysis(
    use_case: AnalyzeAudioSessionUseCase, session_id: str, language: str, audio_bytes: bytes, filename: Optional[str] = None
) -> AudioSessionAnalysis:
    """Run ``use_case.execute`` while holding the global semaphore.

    The *use case* itself is synchronous, so we offload it to a thread
    pool with ``run_in_executor``.  The semaphore ensures only one thread
    at a time is performing the expensive analysis, thus queuing any
    additional requests that arrive while another is running.
    """
    async with _analysis_semaphore:
        loop = asyncio.get_running_loop()
        # run in default executor (thread pool) to avoid blocking the
        # event loop during long model inference.
        return await loop.run_in_executor(
            None, use_case.execute, session_id, language, audio_bytes, filename
        )


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
        # guard against a single conversation growing without bounds. if the
        # limit is hit we raise; the outer websocket handler will convert that
        # into a clean shutdown so the whole process doesn't crash due to an
        # errant client.
        if len(self.audio_buffer) + len(data) > MAX_SESSION_BUFFER:
            raise MemoryError(
                f"session audio buffer would exceed {MAX_SESSION_BUFFER} bytes"
            )
        self.audio_buffer.extend(data)


# Global singleton instances to avoid reloading models on every connection (and OOM errors)
_asr_gateway: Optional[FasterWhisperASRGateway] = None
_conversation_gateway: Optional[ConversationAnalysisGateway] = None
_analyze_audio_use_case: Optional[AnalyzeAudioSessionUseCase] = None

def get_analyze_audio_use_case() -> AnalyzeAudioSessionUseCase:
    global _asr_gateway, _conversation_gateway, _analyze_audio_use_case
    
    if _analyze_audio_use_case is None:
        if _asr_gateway is None:
            _asr_gateway = FasterWhisperASRGateway()

        if _conversation_gateway is None:
            use_real_llm = os.getenv("CONV_LLM_ENABLED", "false").lower() in ("1", "true", "yes", "on")
            if use_real_llm:
                _conversation_gateway = TransformersLLMConversationGateway()
            else:
                _conversation_gateway = DummyConversationAnalysisGateway()
        
        _analyze_audio_use_case = AnalyzeAudioSessionUseCase(_asr_gateway, _conversation_gateway)

    return _analyze_audio_use_case


@router.websocket("/ws/audio")
async def websocket_audio(
    websocket: WebSocket,
    use_case: AnalyzeAudioSessionUseCase = Depends(get_analyze_audio_use_case),
) -> None:
    await websocket.accept()
    session: Optional[AudioSession] = None
    # buffer bytes that may arrive before a config message
    pre_config_buffer = bytearray()


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
                # a genuine SocketDisconnect event triggers the same cleanup we
                # perform for an explicit ``websocket.disconnect`` message, so
                # reuse the helper below.
                async def _handle_implicit_end():
                    if session or pre_config_buffer:
                        print(">>> AUDIO WEBSOCKET: Desconexión con datos de audio pendientes, ejecutando análisis local...")
                        logger.info("disconnect triggered implicit end_of_stream")
                        temp = session
                        if not temp and pre_config_buffer:
                            temp = AudioSession.create(
                                session_id="auto",
                                sample_rate=16000,
                                encoding="pcm16",
                                language="es",
                            )
                            temp.add_bytes(bytes(pre_config_buffer))
                            pre_config_buffer.clear()
                        if temp:
                            payload = bytes(temp.audio_buffer)
                            if temp.encoding.lower() == "pcm16" and len(payload) > 0:
                                buf = io.BytesIO()
                                with wave.open(buf, "wb") as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(temp.sample_rate)
                                    wf.writeframes(payload)
                                payload = buf.getvalue()
                            
                            audio_seconds = len(payload) / (temp.sample_rate * 2)
                            print(f">>> AUDIO WEBSOCKET: Desconexión, analizando {len(payload)} bytes ({audio_seconds:.2f}s)...")
                            analysis = await _serialize_analysis(
                                use_case, 
                                temp.session_id, 
                                temp.language, 
                                payload,
                                filename=f"disconnect_{temp.session_id}.wav"
                            )
                            transcription_text = " ".join([s.text for s in analysis.transcript]) if analysis.transcript else "VACÍA"
                            print(f">>> AUDIO WEBSOCKET: Análisis de desconexión completado. Transcripción: '{transcription_text}'")
                            logger.info("analysis on disconnect: %s", asdict(analysis))
                await _handle_implicit_end()
                break
            except Exception as e:
                print(f">>> AUDIO WEBSOCKET: Error recibiendo mensaje: {e}")
                logger.warning("error receiving message: %s", e)
                break

            msg_type_event = message.get("type")

            if msg_type_event == "websocket.disconnect":
                print(">>> AUDIO WEBSOCKET: Recibido evento de desconexión")
                logger.info("received websocket.disconnect event")
                # run the same implicit end‑of‑stream cleanup as in the
                # WebSocketDisconnect exception handler
                if session or pre_config_buffer:
                    print(">>> AUDIO WEBSOCKET: Desconexión con datos de audio pendientes, ejecutando análisis local...")
                    logger.info("disconnect triggered implicit end_of_stream")
                    temp = session
                    if not temp and pre_config_buffer:
                        temp = AudioSession.create(
                            session_id="auto",
                            sample_rate=16000,
                            encoding="pcm16",
                            language="es",
                        )
                        temp.add_bytes(bytes(pre_config_buffer))
                        pre_config_buffer.clear()
                    if temp:
                        payload = bytes(temp.audio_buffer)
                        if temp.encoding.lower() == "pcm16" and len(payload) > 0:
                            buf = io.BytesIO()
                            with wave.open(buf, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(temp.sample_rate)
                                wf.writeframes(payload)
                            payload = buf.getvalue()
                        
                        audio_seconds = len(payload) / (temp.sample_rate * 2)
                        print(f">>> AUDIO WEBSOCKET: Evento disconnect, analizando {len(payload)} bytes ({audio_seconds:.2f}s)...")
                        analysis = await _serialize_analysis(
                            use_case, 
                            temp.session_id, 
                            temp.language, 
                            payload,
                            filename=f"disconnect_event_{temp.session_id}.wav"
                        )
                        transcription_text = " ".join([s.text for s in analysis.transcript]) if analysis.transcript else "VACÍA"
                        print(f">>> AUDIO WEBSOCKET: Análisis de evento disconnect completado. Transcripción: '{transcription_text}'")
                        logger.info("analysis on disconnect: %s", asdict(analysis))
                break
            
            if "text" in message:
                try:
                    import json
                    data = json.loads(message["text"])
                    client_msg_type = data.get("type")
                    print(f">>> AUDIO WEBSOCKET: Recibido mensaje de texto tipo '{client_msg_type}'")
                    
                    if client_msg_type == "config":
                        # initialize session; if we already buffered bytes, move them in
                        session = AudioSession.create(
                            session_id=data.get("session_id", "unknown"),
                            sample_rate=data.get("sample_rate", 16000),
                            encoding=data.get("encoding", "pcm16"),
                            language=data.get("language", "es"),
                        )
                        if pre_config_buffer:
                            session.add_bytes(bytes(pre_config_buffer))
                            print(f">>> AUDIO WEBSOCKET: Migrados {len(pre_config_buffer)} bytes preconfigurados al buffer de sesión")
                            logger.info("migrated %d pre-config bytes into session", len(pre_config_buffer))
                            pre_config_buffer.clear()
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
                            
                            # Check minimum duration (e.g., 0.5s) to avoid hallucinations
                            audio_seconds = len(payload) / (session.sample_rate * 2) # assuming pcm16
                            if audio_seconds < 0.5:
                                print(f">>> AUDIO WEBSOCKET: Audio demasiado corto ({audio_seconds:.2f}s), omitiendo análisis.")
                                await websocket.send_json({
                                    "type": "final_result",
                                    "analysis": asdict(AudioSessionAnalysis(
                                        session_id=session.session_id,
                                        language=session.language,
                                        transcript=[],
                                        summary="Audio demasiado corto para analizar.",
                                        action_items=[],
                                        risks=[]
                                    )),
                                })
                                session.audio_buffer = bytearray()
                                continue

                            # Execute analysis (possibly waiting for other sessions)
                            print(f">>> AUDIO WEBSOCKET: Iniciando análisis de audio ({len(payload)} bytes, {audio_seconds:.2f}s)...")
                            analysis = await _serialize_analysis(
                                use_case, 
                                session.session_id, 
                                session.language, 
                                payload,
                                filename=f"stream_{session.session_id}.wav"
                            )
                            transcription_text = " ".join([s.text for s in analysis.transcript]) if analysis.transcript else "VACÍA"
                            print(f">>> AUDIO WEBSOCKET: Análisis completado. Transcripción: '{transcription_text}'")
                            
                            # Send result
                            await websocket.send_json({
                                "type": "final_result",
                                "analysis": asdict(analysis),
                            })
                            
                            # Clear buffer
                            session.audio_buffer = bytearray()
                        elif pre_config_buffer:
                            # no explicit config was ever received; auto‑create a default session
                            print(">>> AUDIO WEBSOCKET: end_of_stream sin sesión, pero hay bytes preconfigurados, creando sesión temporal")
                            logger.info("end_of_stream with buffer but no session; auto-configuring")
                            session = AudioSession.create(
                                session_id="auto",
                                sample_rate=16000,
                                encoding="pcm16",
                                language="es",
                            )
                            session.add_bytes(bytes(pre_config_buffer))
                            pre_config_buffer.clear()
                            # and re‑enter same logic by continuing loop iteration
                            # (we could duplicate, but easier to call recursively)
                            continue
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
                    try:
                        session.add_bytes(message["bytes"])
                    except MemoryError as me:
                        # buffer grew too big; we cannot continue safely.
                        logger.error("session buffer overflow: %s", me)
                        await websocket.send_json({
                            "type": "error",
                            "message": "audio too large for this session",
                        })
                        await websocket.close(code=1009)  # 1009 = message too big
                        break
                else:
                    # buffer bytes until we know session config, but enforce a cap
                    incoming = len(message["bytes"])
                    if len(pre_config_buffer) + incoming > MAX_PRECONFIG_BUFFER:
                        logger.warning(
                            "pre-config buffer would exceed %d bytes, discarding %d bytes",
                            MAX_PRECONFIG_BUFFER,
                            incoming,
                        )
                        # optionally inform client once
                        try:
                            await websocket.send_json({
                                "type": "warning",
                                "message": "data received before config discarded",
                            })
                        except Exception:
                            pass
                    else:
                        pre_config_buffer.extend(message["bytes"])
                        print(f">>> AUDIO WEBSOCKET: Recibidos {incoming} bytes antes de configurar sesión (almacenados)")
                        logger.warning("received %d bytes before session config, buffering", incoming)
                    # do not discard; wait for config
                    pass
            
    except Exception as exc:
        print(f">>> AUDIO WEBSOCKET: Error inesperado en bucle principal: {exc}")
        logger.exception("unexpected error in audio websocket loop: %s", exc)
    finally:
        print(">>> AUDIO WEBSOCKET: Cerrando conexión")
        logger.info("audio websocket closing")
