import os
import tempfile
from typing import List, Optional

from faster_whisper import WhisperModel

from app.domain.audio.interfaces import ASRGateway, TranscriptSegment


DEFAULT_MODEL_ID = "large-v3"
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

        self._model = WhisperModel(
            resolved_model_id,
            device=resolved_device,
            compute_type=resolved_compute_type,
        )

    def transcribe(self, audio_bytes: bytes, language: str) -> List[TranscriptSegment]:
        if not audio_bytes:
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
                results.append(
                    TranscriptSegment(
                        speaker="S1",
                        start=float(segment.start),
                        end=float(segment.end),
                        text=str(segment.text),
                    )
                )
            return results
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

