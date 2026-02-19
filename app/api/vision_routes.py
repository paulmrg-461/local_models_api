import base64
import io
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

from app.application.vision.use_cases import AnalyzeImageUseCase
from app.domain.vision.interfaces import VisionAnalysisResult
from app.infrastructure.vision.qwen_service import QwenVisionModel


router = APIRouter()


class VisionFrameB64Request(BaseModel):
    image_b64: str
    session_id: Optional[str] = None


def get_analyze_image_use_case() -> AnalyzeImageUseCase:
    gateway = QwenVisionModel()
    return AnalyzeImageUseCase(gateway)


@router.post("/vision/frame")
async def receive_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    use_case: AnalyzeImageUseCase = Depends(get_analyze_image_use_case),
) -> dict:
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result: VisionAnalysisResult = use_case.execute(image)

    return {
        "description": result.description,
        "session_id": session_id,
        "width": image.width,
        "height": image.height,
    }


@router.post("/vision/frame_b64")
async def receive_frame_b64(
    payload: VisionFrameB64Request,
    use_case: AnalyzeImageUseCase = Depends(get_analyze_image_use_case),
) -> dict:
    try:
        image_bytes = base64.b64decode(payload.image_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image_b64")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result: VisionAnalysisResult = use_case.execute(image)

    return {
        "description": result.description,
        "session_id": payload.session_id,
        "width": image.width,
        "height": image.height,
    }
