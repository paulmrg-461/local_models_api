import os
import tempfile
from typing import List, Optional

from faster_whisper import WhisperModel

from app.domain.audio.interfaces import ASRGateway, TranscriptSegment


# choose a model that will comfortably fit in 12 GB of VRAM by default;
# users with more memory can override via FWHISPER_MODEL_ID.
DEFAULT_MODEL_ID = "small"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"


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

        resolved_model_id = model_id or os.getenv("FWHISPER_MODEL_ID", DEFAULT_MODEL_ID)
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

    def transcribe(self, audio_bytes: bytes, language: str) -> List[TranscriptSegment]:
        # if we have fewer than 2 bytes there is no valid PCM16 frame
        if not audio_bytes or len(audio_bytes) < 2:
            # avoid calling whisper/ffmpeg on a nonsense file; the library
            # would otherwise print an "Invalid PCM packet" warning and
            # potentially raise.
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, _ = self._model.transcribe(
                tmp_path,
                language=language or None,
            )
            results: List[TranscriptSegment] = []
            for segment in segments:
                txt = str(segment.text)
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

