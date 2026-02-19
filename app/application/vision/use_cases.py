from PIL import Image

from app.domain.vision.interfaces import VisionAnalysisResult, VisionModelGatewayProtocol


class AnalyzeImageUseCase:
    def __init__(self, model_gateway: VisionModelGatewayProtocol):
        self.model_gateway = model_gateway

    def execute(self, image: Image.Image) -> VisionAnalysisResult:
        if image is None:
            raise ValueError("image must not be None")
        return self.model_gateway.analyze(image)

