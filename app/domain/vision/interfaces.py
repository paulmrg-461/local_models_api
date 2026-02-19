from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass
class VisionAnalysisResult:
    description: str


class VisionModelGateway(ABC):
    @abstractmethod
    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        raise NotImplementedError


class VisionModelGatewayProtocol(Protocol):
    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        ...

