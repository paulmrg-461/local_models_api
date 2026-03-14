import os
import tempfile
import io
import re
from typing import List, Optional

from faster_whisper import WhisperModel
from pydub import AudioSegment

from app.domain.audio.interfaces import ASRGateway, TranscriptSegment


# choose a model that will comfortably fit in 12 GB of VRAM by default;
# users with more memory can override via FWHISPER_MODEL_ID.
DEFAULT_MODEL_ID = "small"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# Known Whisper hallucinations to filter out
WHISPER_HALLUCINATIONS = [
    r"subtítulos por la comunidad de amara\.org",
    r"¡suscríbete!",
    r"suscríbete",
    r"gracias por ver el video",
    r"thanks for watching",
    r"amara\.org",
    r"community subtitles",
    r"subtítulos por la comunidad",
]

class FasterWhisperASRGateway(ASRGateway):
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        model: Optional[WhisperModel] = None,
    ):
        if model is not None:
            self._model = model
            return

        resolved_model_id = model_id or os.getenv("FWHISPER_MODEL_ID")
        if not resolved_model_id:
            print(f"⚠️  FWHISPER_MODEL_ID no encontrada en env, usando default: {DEFAULT_MODEL_ID}")
            resolved_model_id = DEFAULT_MODEL_ID
            
        resolved_device = device or os.getenv("FWHISPER_DEVICE", DEFAULT_DEVICE)
        resolved_compute_type = compute_type or os.getenv(
            "FWHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE
        )

        # log our resolved configuration so users can verify which device is
        # being used. this runs even if the model later falls back to CPU.
        print(
            f"FasterWhisperASRGateway initialising model={resolved_model_id} "
            f"device={resolved_device} compute_type={resolved_compute_type}"
        )
        try:
            self._model = WhisperModel(
                resolved_model_id,
                device=resolved_device,
                compute_type=resolved_compute_type,
            )
        except RuntimeError as exc:
            # common case: CUDA OOM during model load. by default we fall back
            # to CPU so the server remains usable, but the user may explicitly
            # request "GPU only" and prefer a hard failure instead.
            msg = str(exc).lower()
            if ("out of memory" in msg or "cuda" in msg) and not os.getenv(
                "FWHISPER_REQUIRE_CUDA", "false"
            ).lower() in ("1", "true", "yes", "on"):
                # fallback path
                fallback_device = "cpu"
                fallback_compute = "float32"
                print(
                    "⚠️  whisper init OOM on CUDA, retrying on CPU (this may be slow)"
                )
                self._model = WhisperModel(
                    resolved_model_id,
                    device=fallback_device,
                    compute_type=fallback_compute,
                )
            else:
                # either it wasn’t a CUDA OOM, or the user demanded GPU-only.
                raise

    def _is_hallucination(self, text: str) -> bool:
        """Checks if a given text is a known Whisper hallucination."""
        clean_text = text.strip().lower()
        if not clean_text:
            return True
            
        # If it's just punctuation, it's noise
        if re.match(r'^[.,!?;:\s]+$', clean_text):
            return True
            
        for pattern in WHISPER_HALLUCINATIONS:
            if re.search(pattern, clean_text):
                return True
        return False

    def _prepare_audio(self, audio_bytes: bytes) -> str:
        """Prepares audio for Whisper: 16kHz, Mono, Normalized."""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Normalize and boost slightly
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio = audio.normalize()
            audio = audio + 3 # +3dB boost
            
            # Add a bit of silence at the ends to prevent truncation
            silence = AudioSegment.silent(duration=300, frame_rate=16000)
            audio = silence + audio + silence
            
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            audio.export(tmp_path, format="wav")
            return tmp_path
        except Exception as e:
            # Fallback to direct write
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, 'wb') as f:
                f.write(audio_bytes)
            return tmp_path

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str,
        filename: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        # if we have fewer than 2 bytes there is no valid PCM16 frame
        if not audio_bytes or len(audio_bytes) < 2:
            return []

        tmp_path = self._prepare_audio(audio_bytes)

        try:
            # Transcribe with VAD filtering to reduce hallucinations
            segments, _ = self._model.transcribe(
                tmp_path,
                language=language or None,
                vad_filter=True,
                vad_parameters=dict(min_speech_duration_ms=500),
                initial_prompt="Conversación en español, clara y directa.",
            )
            results: List[TranscriptSegment] = []
            for segment in segments:
                txt = str(segment.text).strip()
                
                # Filter out hallucinations
                if self._is_hallucination(txt):
                    continue
                    
                results.append(
                    TranscriptSegment(
                        speaker="S1",
                        start=float(segment.start),
                        end=float(segment.end),
                        text=txt,
                    )
                )
            # log what we heard
            try:
                import logging
                _logger = logging.getLogger(__name__)
                _logger.info("asr transcribed %d segments", len(results))
                for r in results:
                    _logger.info("asr segment: [%0.2f-%0.2f] %s", r.start, r.end, r.text)
            except Exception:
                pass
            return results
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

