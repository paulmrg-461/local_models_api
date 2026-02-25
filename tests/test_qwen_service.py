from app.infrastructure.vision.qwen_service import QwenVisionModel




def test_analyze_returns_refusal_text(monkeypatch):
    # create a fake model that always responds with refusal text; the service
    # should return whatever the model gives us rather than raising an error.
    class FakeModel:
        def generate(self, **kwargs):
            import torch
            return torch.tensor([[1, 2, 3]])

    class DummyInputs(dict):
        # mimic transformers BatchEncoding/to behavior
        def to(self, device):
            return self
        # allow attribute access (e.g. inputs.input_ids)
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    class FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return ""

        def __call__(self, *args, **kwargs):
            import torch
            return DummyInputs({
                "input_ids": torch.tensor([[1]]),
                "images": torch.zeros((1,3,64,64)),
                "videos": torch.zeros((1,1,1)),
                "attention_mask": torch.tensor([[1]]),
            })

        def batch_decode(self, generated_ids_trimmed, **kwargs):
            return ["Lo siento, pero no puedo ayudarte con esta solicitud."]

    # the environment flag no longer changes behaviour, but set it anyway to
    # prove it doesn't crash or alter the result.
    monkeypatch.setenv("QWEN_VL_ALLOW_REFUSAL", "false")

    # avoid real model/proc initialization and qwen_vl_utils imports
    monkeypatch.setattr(QwenVisionModel, "__init__", lambda self: None)

    import types, sys
    fake_utils = types.SimpleNamespace(process_vision_info=lambda msgs: (None, None))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_utils)

    gadget = QwenVisionModel()
    gadget._device = "cpu"
    gadget._model = FakeModel()
    gadget._processor = FakeProcessor()

    image = __import__("PIL").Image.new("RGB", (1,1))
    result = gadget.analyze(image)
    assert result.description.startswith("Lo siento")

    # flipping the flag has no effect
    monkeypatch.setenv("QWEN_VL_ALLOW_REFUSAL", "true")
    result2 = gadget.analyze(image)
    assert result2.description.startswith("Lo siento")


def test_retry_on_apology(monkeypatch):
    # first model response is an apology, second is a real description
    class FakeModel:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            import torch
            self.calls += 1
            if self.calls == 1:
                return torch.tensor([[1, 2, 3]])
            else:
                return torch.tensor([[4, 5, 6]])

    class DummyInputs(dict):
        def to(self, device):
            return self
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    class FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return ""

        def __call__(self, *args, **kwargs):
            import torch
            return DummyInputs({
                "input_ids": torch.tensor([[1]]),
                "images": torch.zeros((1,3,64,64)),
                "videos": torch.zeros((1,1,1)),
                "attention_mask": torch.tensor([[1]]),
            })

        def batch_decode(self, generated_ids_trimmed, **kwargs):
            # decide based on first token value to simulate different outputs
            first_val = generated_ids_trimmed[0][0].item()
            if first_val == 1:
                return ["Lo siento, pero no puedo ayudarte con esta solicitud."]
            else:
                return ["Hay una mancha de aceite en el motor del bus."]

    monkeypatch.setattr(QwenVisionModel, "__init__", lambda self: None)
    import types, sys
    fake_utils = types.SimpleNamespace(process_vision_info=lambda msgs: (None, None))
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_utils)

    gadget = QwenVisionModel()
    gadget._device = "cpu"
    gadget._model = FakeModel()
    gadget._processor = FakeProcessor()

    image = __import__("PIL").Image.new("RGB", (1,1))
    result = gadget.analyze(image)
    assert "mancha de aceite" in result.description
