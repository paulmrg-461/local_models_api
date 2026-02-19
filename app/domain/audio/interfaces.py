from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class ActionItem:
    title: str
    description: str
    steps: List[str]


@dataclass
class AudioSessionAnalysis:
    session_id: str
    language: str
    transcript: List[TranscriptSegment]
    summary: str
    action_items: List[ActionItem]
    risks: List[str]


class ASRGateway(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str) -> List[TranscriptSegment]:
        raise NotImplementedError


class ConversationAnalysisGateway(ABC):
    @abstractmethod
    def analyze(
        self,
        session_id: str,
        language: str,
        segments: List[TranscriptSegment],
    ) -> AudioSessionAnalysis:
        raise NotImplementedError

