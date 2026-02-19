import io
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from PIL import Image

from app.application.vision.use_cases import AnalyzeImageUseCase
from app.domain.vision.interfaces import VisionAnalysisResult
from app.infrastructure.vision.qwen_service import QwenVisionModel


router = APIRouter()


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

