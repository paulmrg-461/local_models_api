from io import BytesIO
import base64

from fastapi.testclient import TestClient
from PIL import Image

from app.api.vision_routes import get_analyze_image_use_case, router
from app.application.vision.use_cases import AnalyzeImageUseCase
from app.domain.vision.interfaces import VisionAnalysisResult, VisionModelGateway
from app.main import app


class FakeGateway(VisionModelGateway):
    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        return VisionAnalysisResult(description="fake response")


def override_use_case() -> AnalyzeImageUseCase:
    gateway = FakeGateway()
    return AnalyzeImageUseCase(gateway)


app.dependency_overrides[get_analyze_image_use_case] = override_use_case
app.include_router(router)


def test_vision_frame_endpoint_returns_description():
    client = TestClient(app)
    image = Image.new("RGB", (8, 8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"file": ("test.png", buffer.getvalue(), "image/png")}
    response = client.post("/vision/frame", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "fake response"


def test_vision_frame_b64_endpoint_returns_description():
    client = TestClient(app)
    image = Image.new("RGB", (8, 8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    payload = {"image_b64": image_b64}
    response = client.post("/vision/frame_b64", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["description"] == "fake response"


def test_vision_frame_b64_endpoint_rejects_invalid_base64():
    client = TestClient(app)
    payload = {"image_b64": "not-base64"}

    response = client.post("/vision/frame_b64", json=payload)

    assert response.status_code == 400
