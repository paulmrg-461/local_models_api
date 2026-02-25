import pytest
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


def test_gateway_cpu_fallback_on_cuda_oom(monkeypatch, capsys):
    """Simulate a CUDA OOM during model initialization and ensure we retry on CPU."""

    class OOMModelFactory:
        def __init__(self, model_id, device=None, compute_type=None):
            if device == "cuda":
                raise RuntimeError("CUDA failed with error out of memory")
            # otherwise create a simple dummy object
            self.args = (model_id, device, compute_type)

    monkeypatch.setattr(
        "app.infrastructure.audio.faster_whisper_gateway.WhisperModel",
        OOMModelFactory,
    )

    gw = FasterWhisperASRGateway()
    # after construction the _model should be an instance of our fake type
    assert isinstance(gw._model, OOMModelFactory)
    # and it should have been created with device='cpu'
    assert gw._model.args[1] == "cpu"
    # we should have printed a warning message plus our initialization log
    captured = capsys.readouterr()
    assert "OOM on CUDA" in captured.out or "retrying on CPU" in captured.out
    assert "initialising model=" in captured.out


def test_require_cuda_raises_on_oom(monkeypatch):
    """If FWHISPER_REQUIRE_CUDA is truthy we should propagate the OOM instead
    of falling back to CPU.
    """
    class OOMModelFactory:
        def __init__(self, model_id, device=None, compute_type=None):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            self.args = (model_id, device, compute_type)

    monkeypatch.setattr(
        "app.infrastructure.audio.faster_whisper_gateway.WhisperModel",
        OOMModelFactory,
    )

    monkeypatch.setenv("FWHISPER_REQUIRE_CUDA", "true")
    with pytest.raises(RuntimeError):
        FasterWhisperASRGateway()


def test_gateway_ignores_tiny_audio():
    gw = FasterWhisperASRGateway(model=FakeModel())  # type: ignore[arg-type]
    segments = gw.transcribe(b"\x00", language="es")
    assert segments == []

