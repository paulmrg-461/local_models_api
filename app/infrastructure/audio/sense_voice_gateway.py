import os
import logging
import tempfile
from typing import List, Optional
from funasr import AutoModel
from app.domain.audio.interfaces import ASRGateway, TranscriptSegment

logger = logging.getLogger(__name__)

DEFAULT_SENSEVOICE_MODEL = "iic/SenseVoiceSmall"
DEFAULT_VAD_MODEL = "fsmn-vad"
DEFAULT_DEVICE = "cuda:0"

class SenseVoiceASRGateway(ASRGateway):
    def __init__(self) -> None:
        model_dir = os.getenv("SENSEVOICE_MODEL_ID", DEFAULT_SENSEVOICE_MODEL)
        vad_model = os.getenv("SENSEVOICE_VAD_MODEL", DEFAULT_VAD_MODEL)
        device = os.getenv("SENSEVOICE_DEVICE", DEFAULT_DEVICE)
        
        logger.info("Loading SenseVoice model: %s on %s", model_dir, device)
        
        try:
            self._model = AutoModel(
                model=model_dir,
                vad_model=vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
                device=device
            )
            logger.info("SenseVoice model loaded successfully")
        except Exception as e:
            logger.error("Failed to load SenseVoice model: %s", e)
            raise

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str,
        filename: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        if not audio_bytes:
            return []

        # Determine suffix from original filename
        suffix = ".wav"
        if filename:
            _, ext = os.path.splitext(filename)
            if ext:
                suffix = ext

        # Save to temporary file. Now that FFmpeg is installed, 
        # SenseVoice (funasr) will be able to decode .ogg, .mp3, etc.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            target_lang = language if language != "auto" else "auto"
            
            logger.info("Processing file with SenseVoiceSmall: %s (lang=%s)", tmp_path, target_lang)
            
            # We use the file path directly. funasr will use ffmpeg to load it.
            res = self._model.generate(
                input=tmp_path,
                cache={},
                language=target_lang,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            if not res or len(res) == 0:
                logger.warning("SenseVoice returned no results")
                return []

            full_text = res[0].get("text", "").strip()
            
            if not full_text:
                return []

            return [
                TranscriptSegment(
                    speaker="Speaker",
                    start=0.0,
                    end=0.0,
                    text=full_text
                )
            ]

        except Exception as e:
            logger.exception("Error during SenseVoice transcription: %s", e)
            return []
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
