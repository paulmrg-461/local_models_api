from fastapi.testclient import TestClient

from app.api.audio_ws import get_analyze_audio_use_case
from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import (
    ActionItem,
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
    TranscriptSegment,
)
from app.main import app


class FakeASR(ASRGateway):
    def transcribe(self, audio_bytes: bytes, language: str) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                speaker="S1",
                start=0.0,
                end=1.0,
                text="hello",
            )
        ]


class FakeConversationAnalyzer(ConversationAnalysisGateway):
    def analyze(
        self,
        session_id: str,
        language: str,
        segments: list[TranscriptSegment],
    ) -> AudioSessionAnalysis:
        return AudioSessionAnalysis(
            session_id=session_id,
            language=language,
            transcript=segments,
            summary="summary",
            action_items=[
                ActionItem(
                    title="title",
                    description="description",
                    steps=["one"],
                )
            ],
            risks=["risk"],
        )


def override_use_case() -> AnalyzeAudioSessionUseCase:
    asr = FakeASR()
    analyzer = FakeConversationAnalyzer()
    return AnalyzeAudioSessionUseCase(asr, analyzer)


app.dependency_overrides[get_analyze_audio_use_case] = override_use_case


def test_audio_websocket_final_result_message():
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as websocket:
        websocket.send_json(
            {
                "type": "config",
                "session_id": "abc",
                "sample_rate": 16000,
                "encoding": "pcm16",
                "language": "es",
            }
        )
        websocket.send_bytes(b"\x00\x01")
        websocket.send_json({"type": "end_of_stream"})

        message = websocket.receive_json()

        assert message["type"] == "final_result"
        analysis = message["analysis"]
        assert analysis["session_id"] == "abc"
        assert analysis["language"] == "es"
        assert analysis["summary"] == "summary"
        assert len(analysis["transcript"]) == 1
        assert analysis["transcript"][0]["text"] == "hello"
        assert len(analysis["action_items"]) == 1
        assert len(analysis["risks"]) == 1

