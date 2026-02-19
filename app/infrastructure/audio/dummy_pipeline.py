from typing import List

from app.domain.audio.interfaces import (
    ASRGateway,
    ActionItem,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
    TranscriptSegment,
)


class DummyASRGateway(ASRGateway):
    def transcribe(self, audio_bytes: bytes, language: str) -> List[TranscriptSegment]:
        return [
            TranscriptSegment(
                speaker="S1",
                start=0.0,
                end=1.0,
                text="dummy transcript",
            )
        ]


class DummyConversationAnalysisGateway(ConversationAnalysisGateway):
    def analyze(
        self,
        session_id: str,
        language: str,
        segments: List[TranscriptSegment],
    ) -> AudioSessionAnalysis:
        summary = "dummy summary"
        action_items = [
            ActionItem(
                title="dummy action",
                description="dummy description",
                steps=["dummy step"],
            )
        ]
        risks = ["dummy risk"]
        return AudioSessionAnalysis(
            session_id=session_id,
            language=language,
            transcript=segments,
            summary=summary,
            action_items=action_items,
            risks=risks,
        )

