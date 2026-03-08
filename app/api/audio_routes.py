import logging
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.infrastructure.audio.sense_voice_gateway import SenseVoiceASRGateway
from app.domain.audio.interfaces import TranscriptSegment

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple singleton pattern for the gateway
_sensevoice_gateway: Optional[SenseVoiceASRGateway] = None

def get_sensevoice_gateway() -> SenseVoiceASRGateway:
    global _sensevoice_gateway
    if _sensevoice_gateway is None:
        _sensevoice_gateway = SenseVoiceASRGateway()
    return _sensevoice_gateway

@router.post("/audio/sense-voice")
async def transcribe_with_sense_voice(
    file: UploadFile = File(...),
    language: str = Form("auto"),
) -> dict:
    """
    Endpoint for uploading an audio file and transcribing it using SenseVoiceSmall.
    Supports detecting acoustic events like <|applause|>, <|laughter|>, etc.
    """
    logger.info("Received audio file for SenseVoice transcription: %s", file.filename)
    
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Could not read uploaded audio file")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file provided")

    gateway = get_sensevoice_gateway()
    
    try:
        segments: List[TranscriptSegment] = gateway.transcribe(
            audio_bytes, 
            language, 
            filename=file.filename
        )
        
        if not segments:
            return {
                "transcription": "",
                "segments": [],
                "filename": file.filename,
                "message": "No speech detected or transcription failed"
            }

        # SenseVoiceSmall consolidation
        full_transcription = " ".join([s.text for s in segments])
        
        return {
            "transcription": full_transcription,
            "segments": [
                {
                    "speaker": s.speaker,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text
                } for s in segments
            ],
            "filename": file.filename
        }

    except Exception as e:
        logger.exception("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
