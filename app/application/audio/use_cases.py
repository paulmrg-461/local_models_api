from app.domain.audio.interfaces import (
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
)


import logging

logger = logging.getLogger(__name__)

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
        logger.info("use case execute: transcribing %d bytes", len(audio_bytes))
        segments = self.asr_gateway.transcribe(audio_bytes, language)
        logger.info("transcription yielded %d segments", len(segments))
        for seg in segments:
            logger.info("segment: [%0.2f-%0.2f] %s", seg.start, seg.end, seg.text)
        analysis = self.conversation_gateway.analyze(session_id, language, segments)
        return analysis

