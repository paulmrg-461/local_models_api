import json
import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.domain.audio.interfaces import (
    ActionItem,
    AudioSessionAnalysis,
    ConversationAnalysisGateway,
    TranscriptSegment,
)


DEFAULT_BACKEND = "TRANSFORMERS"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 512


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


class TransformersLLMConversationGateway(ConversationAnalysisGateway):
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype_str: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ):
        resolved_model_id = model_id or os.getenv("CONV_LLM_MODEL_ID", DEFAULT_MODEL_ID)
        resolved_device = device or os.getenv("CONV_LLM_DEVICE", DEFAULT_DEVICE)
        resolved_dtype_str = torch_dtype_str or os.getenv(
            "CONV_LLM_TORCH_DTYPE", DEFAULT_DTYPE
        )
        resolved_max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else int(os.getenv("CONV_LLM_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
        )

        if resolved_dtype_str.lower() == "bfloat16":
            dtype = torch.bfloat16
        elif resolved_dtype_str.lower() == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id,
            torch_dtype=dtype,
            device_map=resolved_device,
        )
        self._max_new_tokens = resolved_max_new_tokens
        self._device = resolved_device

    def analyze(
        self,
        session_id: str,
        language: str,
        segments: List[TranscriptSegment],
    ) -> AudioSessionAnalysis:
        prompt = self._build_prompt(language, segments)

        inputs = self._tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )

        output_text = self._tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        try:
            data = json.loads(output_text)
        except Exception:
            data = {
                "summary": output_text,
                "action_items": [],
                "risks": [],
            }

        summary = str(data.get("summary", "")).strip()
        action_items_raw = data.get("action_items", []) or []
        risks_raw = data.get("risks", []) or []

        action_items: List[ActionItem] = []
        for item in action_items_raw:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            steps = [str(s).strip() for s in item.get("steps", []) or []]
            action_items.append(ActionItem(title=title, description=description, steps=steps))

        risks: List[str] = [str(r).strip() for r in risks_raw]

        return AudioSessionAnalysis(
            session_id=session_id,
            language=language,
            transcript=segments,
            summary=summary,
            action_items=action_items,
            risks=risks,
        )

    def _build_prompt(self, language: str, segments: List[TranscriptSegment]) -> str:
        lines = []
        for seg in segments:
            lines.append(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.speaker}: {seg.text}")
        transcript_text = "\n".join(lines)

        instruction = (
            "You are an expert meeting assistant. Analyze the conversation transcript below and "
            "produce a JSON with keys: summary (string), action_items (array of {title, description, steps}), "
            "and risks (array of strings). Keep the output strictly valid JSON."
        )

        if language.lower().startswith("es"):
            instruction = (
                "Eres un asistente experto de reuniones. Analiza la transcripción y devuelve un JSON "
                "con: summary (string), action_items (array de {title, description, steps}), y risks "
                "(array de strings). El JSON debe ser estrictamente válido."
            )

        return f"{instruction}\n\nTranscript:\n{transcript_text}\n\nJSON:"

