from typing import Optional
from app.domain.audio.interfaces import (
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
)


import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AnalyzeAudioSessionUseCase:
    def __init__(
        self,
        asr_gateway: ASRGateway,
        conversation_gateway: ConversationAnalysisGateway,
    ):
        self.asr_gateway = asr_gateway
        self.conversation_gateway = conversation_gateway
        # Create transcriptions directory if it doesn't exist
        self.transcriptions_dir = "transcriptions"
        if not os.path.exists(self.transcriptions_dir):
            os.makedirs(self.transcriptions_dir)

    def execute(
        self,
        session_id: str,
        language: str,
        audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> AudioSessionAnalysis:
        logger.info("use case execute: transcribing %d bytes", len(audio_bytes))
        segments = self.asr_gateway.transcribe(audio_bytes, language, filename=filename)
        
        logger.info("transcription yielded %d segments", len(segments))
        for seg in segments:
            logger.info("segment: [%0.2f-%0.2f] %s", seg.start, seg.end, seg.text)
        
        # 1. Analizar con LLM
        analysis = self.conversation_gateway.analyze(session_id, language, segments)

        # 2. Guardar transcripción Y análisis en el archivo de texto
        if segments:
            full_text = " ".join([s.text for s in segments])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = filename.replace(" ", "_") if filename else "session"
            transcript_file = os.path.join(self.transcriptions_dir, f"transcript_{timestamp}_{safe_filename}.txt")
            
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(f"ID DE SESIÓN: {session_id}\n")
                    f.write(f"FECHA: {timestamp}\n")
                    f.write(f"IDIOMA: {language}\n")
                    f.write(f"ARCHIVO ORIGINAL: {filename}\n")
                    f.write("\n" + "="*30 + "\n")
                    f.write("TRANSCRIPCIÓN ORIGINAL:\n")
                    f.write("="*30 + "\n")
                    f.write(full_text + "\n\n")
                    
                    f.write("="*30 + "\n")
                    f.write("RESUMEN E INTERPRETACIÓN (IA):\n")
                    f.write("="*30 + "\n")
                    f.write(f"{analysis.summary}\n\n")
                    
                    if analysis.action_items:
                        f.write("TAREAS PENDIENTES IDENTIFICADAS:\n")
                        for item in analysis.action_items:
                            f.write(f"- {item.title}: {item.description}\n")
                            for step in item.steps:
                                f.write(f"  * {step}\n")
                        f.write("\n")
                    
                    if analysis.risks:
                        f.write("SOLUCIONES Y SUGERENCIAS:\n")
                        for risk in analysis.risks:
                            f.write(f"- {risk}\n")
                        f.write("\n")

                logger.info("Saved transcription and analysis to %s", transcript_file)
            except Exception as e:
                logger.error("Failed to save transcription to file: %s", e)

        return analysis

