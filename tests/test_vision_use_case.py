from PIL import Image

from app.application.vision.use_cases import AnalyzeImageUseCase
from app.domain.vision.interfaces import VisionAnalysisResult, VisionModelGateway


class FakeVisionModel(VisionModelGateway):
    def __init__(self, description: str):
        self._description = description

    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        return VisionAnalysisResult(description=self._description)


def test_analyze_image_returns_description():
    gateway = FakeVisionModel("ok")
    use_case = AnalyzeImageUseCase(gateway)
    image = Image.new("RGB", (10, 10))

    result = use_case.execute(image)

    assert result.description == "ok"


def test_analyze_image_raises_when_image_is_none():
    gateway = FakeVisionModel("unused")
    use_case = AnalyzeImageUseCase(gateway)

    try:
        use_case.execute(None)
        assert False
    except ValueError:
        assert True

