from typing import List
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from app.domain.vision.interfaces import VisionAnalysisResult, VisionModelGateway


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"


class QwenVisionModel(VisionModelGateway):
    def __init__(self) -> None:
        model_id = os.getenv("QWEN_VL_MODEL_ID", DEFAULT_MODEL_ID)
        device = os.getenv("QWEN_VL_DEVICE", DEFAULT_DEVICE)
        dtype_str = os.getenv("QWEN_VL_TORCH_DTYPE", DEFAULT_DTYPE)
        if dtype_str.lower() == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str.lower() == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )
        self._processor = AutoProcessor.from_pretrained(model_id)

    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Act as an expert visual assistant. "
                            "Describe in detail what you see: people, objects, "
                            "environment, and relevant activities. Indicate if "
                            "anything looks dangerous or unusual and explain why. "
                            "If appropriate, suggest useful actions or recommendations "
                            "for the person taking the picture. Answer in Spanish."
                        ),
                    },
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed: List[torch.Tensor] = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return VisionAnalysisResult(description=output_text[0].strip())
