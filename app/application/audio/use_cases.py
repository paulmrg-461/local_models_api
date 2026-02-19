from app.domain.audio.interfaces import (
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
)


class AnalyzeAudioSessionUseCase:
    def __init__(
        self,
        asr_gateway: ASRGateway,
        conversation_gateway: ConversationAnalysisGateway,
    ):
        self.asr_gateway = asr_gateway
        self.conversation_gateway = conversation_gateway

    def execute(
        self,
        session_id: str,
        language: str,
        audio_bytes: bytes,
    ) -> AudioSessionAnalysis:
        segments = self.asr_gateway.transcribe(audio_bytes, language)
        analysis = self.conversation_gateway.analyze(session_id, language, segments)
        return analysis

