from app.application.audio.use_cases import AnalyzeAudioSessionUseCase
from app.domain.audio.interfaces import (
    ActionItem,
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
    TranscriptSegment,
)


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


def test_analyze_audio_session_returns_expected_analysis():
    asr = FakeASR()
    analyzer = FakeConversationAnalyzer()
    use_case = AnalyzeAudioSessionUseCase(asr, analyzer)

    analysis = use_case.execute(
        session_id="abc",
        language="es",
        audio_bytes=b"\x00\x01",
    )

    assert analysis.session_id == "abc"
    assert analysis.language == "es"
    assert analysis.summary == "summary"
    assert len(analysis.transcript) == 1
    assert analysis.transcript[0].text == "hello"
    assert len(analysis.action_items) == 1
    assert analysis.action_items[0].title == "title"
    assert len(analysis.risks) == 1

