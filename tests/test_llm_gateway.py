from typing import List

from app.domain.audio.interfaces import TranscriptSegment
from app.infrastructure.audio.llm_gateway import TransformersLLMConversationGateway


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"dummy": 0}

    def __call__(self, inputs, return_tensors="pt"):
        import torch
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": ids, "to": lambda device: {"input_ids": ids, "attention_mask": ids}}

    def batch_decode(self, tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ['{"summary":"ok","action_items":[{"title":"t","description":"d","steps":["s1"]}],"risks":["r1"]}']


class FakeModel:
    def generate(self, **kwargs):
        import torch
        return torch.tensor([[1, 2, 3, 4, 5, 6]])


def test_llm_gateway_maps_json_to_audio_session_analysis(monkeypatch):
    monkeypatch.setattr("app.infrastructure.audio.llm_gateway.AutoTokenizer", type("T", (), {"from_pretrained": lambda *args, **kwargs: FakeTokenizer()}) )
    monkeypatch.setattr("app.infrastructure.audio.llm_gateway.AutoModelForCausalLM", type("M", (), {"from_pretrained": lambda *args, **kwargs: FakeModel()}) )

    gateway = TransformersLLMConversationGateway(model_id="any", device="cpu", torch_dtype_str="float32", max_new_tokens=32)

    segments: List[TranscriptSegment] = [
        TranscriptSegment(speaker="S1", start=0.0, end=1.0, text="hola mundo")
    ]

    analysis = gateway.analyze(session_id="sess", language="es", segments=segments)

    assert analysis.session_id == "sess"
    assert analysis.language == "es"
    assert analysis.transcript == segments
    assert analysis.summary == "ok"
    assert len(analysis.action_items) == 1
    assert analysis.action_items[0].title == "t"
    assert analysis.risks == ["r1"]

