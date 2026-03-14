from typing import Optional
from app.domain.audio.interfaces import (
    ASRGateway,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
)


import os
import logging
from datetime import datetime
import httpx
from dataclasses import asdict

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
        
        # RAG Service URL
        self.rag_service_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8787")

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

        full_text = " ".join([s.text for s in segments]) if segments else ""

        # 2. Guardar en DB Vectorial (RAG Service)
        if full_text:
            try:
                # Prepare payload according to rag_docs.md
                payload = {
                    "transcript_original": full_text,
                    "interpretation": {
                        "summary": analysis.summary,
                        "action_items": [asdict(item) for item in analysis.action_items],
                        "risks": analysis.risks
                    }
                }
                
                logger.info("Sending memory to RAG Service: %s", self.rag_service_url)
                # We do a fire-and-forget or simple sync call here. 
                # Since this is a sync use case called in a threadpool, a sync post is fine.
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(f"{self.rag_service_url}/user/memories/", json=payload)
                    response.raise_for_status()
                    logger.info("Successfully stored memory in RAG Service")
            except Exception as e:
                logger.error("Failed to store memory in RAG Service: %s", e)

        # 3. Guardar transcripción Y análisis en el archivo de texto local (Backup)
        if segments:
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
                    # Check if analysis.summary is raw JSON or actual summary
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

