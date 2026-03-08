import os
import logging
import tempfile
import io
import re
from typing import List, Optional
import librosa
import numpy as np
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
                device=device,
                disable_update=True # Avoid slow checks on startup
            )
            logger.info("SenseVoice model loaded successfully")
        except Exception as e:
            logger.error("Failed to load SenseVoice model: %s", e)
            raise

    def _clean_text(self, text: str) -> str:
        """Removes acoustic and language tags like <|en|>, <|HAPPY|>, etc."""
        return re.sub(r"<\|.*?\|>", "", text).strip()

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str,
        filename: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        if not audio_bytes:
            return []

        try:
            # Load and normalize audio to avoid hallucinations due to low volume or noise
            logger.info("Loading and normalizing audio with librosa...")
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=16000)
            
            if len(y) == 0:
                return []
                
            # Normalize amplitude to -1.0 to 1.0 range
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val
            
            duration = len(y) / sr
            logger.info("Audio processed: %.2fs", duration)

            # Map "es" to the specific code SenseVoice expects if needed, 
            # but usually "es" or "auto" works if the model is loaded correctly.
            target_lang = language if language != "auto" else "auto"
            
            logger.info("Inference with SenseVoiceSmall (lang=%s)...", target_lang)
            res = self._model.generate(
                input=y,
                cache={},
                language=target_lang,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            if not res or len(res) == 0:
                return []

            raw_text = res[0].get("text", "")
            clean_text = self._clean_text(raw_text)
            
            logger.info("Raw output: %s", raw_text)
            logger.info("Clean output: %s", clean_text)
            
            if not clean_text:
                return []

            return [
                TranscriptSegment(
                    speaker="Speaker",
                    start=0.0,
                    end=duration,
                    text=clean_text
                )
            ]

        except Exception as e:
            logger.exception("Error during SenseVoice transcription: %s", e)
            return []
