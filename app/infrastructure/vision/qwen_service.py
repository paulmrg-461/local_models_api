from typing import List
import os
import logging

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

from app.domain.vision.interfaces import VisionAnalysisResult, VisionModelGateway

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"


class QwenVisionModel(VisionModelGateway):
    def __init__(self) -> None:
        model_id = os.getenv("QWEN_VL_MODEL_ID", DEFAULT_MODEL_ID)
        requested_device = os.getenv("QWEN_VL_DEVICE", DEFAULT_DEVICE)
        dtype_str = os.getenv("QWEN_VL_TORCH_DTYPE", DEFAULT_DTYPE)
        if dtype_str.lower() == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str.lower() == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if requested_device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        if device == "cuda" and dtype is torch.bfloat16:
            dtype = torch.float16

        self._device = device
        
        # Optimize for VRAM: load in 4-bit quantization if possible
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Loading Vision Model {model_id} in 4-bit mode for VRAM optimization")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload",
            )
        except Exception as e:
            logger.warning(f"Could not load vision in 4-bit ({e}), using {dtype} on {device}")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)

        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def analyze(self, image: Image.Image) -> VisionAnalysisResult:
        from qwen_vl_utils import process_vision_info

        # build the standard chat message with an explicit "no apologies"
        # clause.  in practice the model still sometimes responds with a
        # boilerplate "Lo siento..." string, so we detect that case and
        # perform a second pass using an even stronger prompt.  this allows the
        # API to recover in many real‑world situations where the default safety
        # filter would otherwise short‑circuit the description.
        base_prompt = (
            "Act as an expert visual assistant. "
            "Describe in detail what you see: people, objects, "
            "environment, and relevant activities. Indicate if "
            "anything looks dangerous or unusual and explain why. "
            "If appropriate, suggest useful actions or recommendations "
            "for the person taking the picture. Answer in Spanish. "
            "Do not say 'Lo siento' or 'No puedo'; always return a description."
        )

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": base_prompt}]}
        ]

        description = self._run_model(messages)

        # if the model still refused, try once more with an even more forceful
        # instruction.  we avoid an infinite loop by only retrying a single time.
        if description.lower().startswith("lo siento") or "no puedo" in description.lower():
            alt_prompt = base_prompt + " Por favor, describe la imagen sin ningún tipo de disculpa."
            messages[0]["content"][1]["text"] = alt_prompt
            description = self._run_model(messages)

        return VisionAnalysisResult(description=description)

    def _run_model(self, messages: list) -> str:
        """Execute the processor/model pipeline and return the decoded text."""
        from qwen_vl_utils import process_vision_info

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
        ).to(self._device)

        try:
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=256)
        except RuntimeError as e:
            if "no kernel image is available for execution on the device" not in str(e):
                raise
            cpu_device = "cpu"
            self._device = cpu_device
            self._model = self._model.to(cpu_device)
            inputs = {k: v.to(cpu_device) for k, v in inputs.items()}
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

        return output_text[0].strip()

