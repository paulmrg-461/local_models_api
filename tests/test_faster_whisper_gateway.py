from typing import List

from app.domain.audio.interfaces import TranscriptSegment
from app.infrastructure.audio.faster_whisper_gateway import FasterWhisperASRGateway


class FakeSegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class FakeModel:
    def transcribe(self, audio_path: str, language: str):
        segments: List[FakeSegment] = [FakeSegment(0.0, 1.0, "hello")]
        return segments, None


def test_faster_whisper_gateway_maps_segments_to_domain_model():
    fake_model = FakeModel()
    gateway = FasterWhisperASRGateway(model=fake_model)  # type: ignore[arg-type]

    segments = gateway.transcribe(b"\x00\x01", language="es")

    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment, TranscriptSegment)
    assert segment.speaker == "S1"
    assert segment.start == 0.0
    assert segment.end == 1.0
    assert segment.text == "hello"

