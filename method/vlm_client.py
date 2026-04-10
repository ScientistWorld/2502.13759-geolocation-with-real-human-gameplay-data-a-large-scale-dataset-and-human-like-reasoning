"""
VLM Client for running GeoCoT with various vision-language models.

Supports:
- OpenAI GPT-4o (via API)
- Local VLMs via transformers (Qwen2.5-VL, LLaVA, etc.)
"""

import base64
import os
import json
from pathlib import Path
from typing import Optional, List, Tuple

# Lazy imports - torch and PIL are imported inside methods to ensure
# PYTHONPATH is set correctly before importing


def encode_image_base64(image_path: str) -> str:
    """Encode an image file as base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMClient:
    """Base class for VLM clients."""

    def predict(self, image_path: str, prompt: str) -> str:
        """
        Predict location from image using the given prompt.

        Args:
            image_path: Path to the image file
            prompt: The prompt to send to the model

        Returns:
            The model's text response
        """
        raise NotImplementedError


class GPT4oClient(VLMClient):
    """GPT-4o via OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def predict(self, image_path: str, prompt: str) -> str:
        b64_image = encode_image_base64(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            max_tokens=512
        )
        return response.choices[0].message.content


class QwenVLClient(VLMClient):
    """Qwen2.5-VL via transformers (local)."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        # Add package paths before importing torch/transformers
        import sys as _sys
        for _pylib in ['/pylib', '/dev/shm/pylib']:
            if os.path.isdir(_pylib) and _pylib not in _sys.path:
                _sys.path.insert(0, _pylib)

        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen-vl-utils and transformers with Qwen2.5-VL support required. "
                f"Install: pip install qwen-vl-utils transformers. Error: {e}"
            )

        # Default to shared models directory
        if model_path is None:
            # Check common locations
            for candidate in [
                "/home/user/shared/models/Qwen2.5-VL-7B-Instruct",
                "/home/user/Qwen/Qwen2.5-VL-7B-Instruct",
                os.path.join(os.path.dirname(__file__), "..", "Qwen2.5-VL-7B-Instruct"),
            ]:
                if os.path.isdir(candidate):
                    model_path = candidate
                    break
            if model_path is None:
                model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # fallback: try HF download

        self.model_path = model_path
        import torch as _torch
        self.device = device if _torch.cuda.is_available() else "cpu"

        print(f"Loading Qwen2.5-VL from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print("Model loaded successfully.")

    def predict(self, image_path: str, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


class LLaVAClient(VLMClient):
    """LLaVA via transformers (local), using llava repo LlavaLlamaForCausalLM."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        import sys as _sys
        import os as _os

        # Set HF cache before imports
        _os.environ.setdefault("HF_HOME", "/home/user/shared/models/hf")
        _os.environ.setdefault("TRANSFORMERS_CACHE", "/home/user/shared/models/hf")
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # Add package paths (try multiple locations)
        # 1. System Python (container with packages installed system-wide)
        # 2. /pylib (alternate package location)
        # 3. /dev/shm/pylib (pre-installed on some nodes)
        for _pylib in ['/pylib', '/dev/shm/pylib']:
            if os.path.isdir(_pylib) and _pylib not in _sys.path:
                _sys.path.insert(0, _pylib)
        _sys.path.insert(0, '/home/user/shared/models/llava_repo')

        import torch

        # Default to shared models directory
        if model_path is None:
            for candidate in [
                "/dev/shm/llava-v1.5-7b-config",
                "/home/user/shared/models/llava-v1.5-7b",
                "/home/user/shared/models/llava-hf/llava-1.5-7b-hf",
                os.path.join(os.path.dirname(__file__), "..", "llava-v1.5-7b"),
            ]:
                if os.path.isdir(candidate):
                    model_path = candidate
                    break
            if model_path is None:
                model_path = "llava-hf/llava-1.5-7b-hf"

        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"

        print(f"Loading LLaVA from {self.model_path}...")

        # Monkeypatch CLIPVisionConfig.from_pretrained and CLIPModel.from_pretrained
        # to use local CLIP path (required for offline loading)
        from transformers import CLIPVisionConfig, CLIPModel

        _original_clipconfig = CLIPVisionConfig.from_pretrained
        def _patched_clipconfig(cls, pretrained_model_name_or_path, **kwargs):
            name_map = {
                "openai/clip-vit-large-patch14-336": "/home/user/shared/models/clip-vit-large-patch14-336",
            }
            if pretrained_model_name_or_path in name_map:
                pretrained_model_name_or_path = name_map[pretrained_model_name_or_path]
            kwargs['local_files_only'] = True
            return _original_clipconfig(pretrained_model_name_or_path, **kwargs)
        CLIPVisionConfig.from_pretrained = classmethod(_patched_clipconfig.__get__(None, CLIPVisionConfig))

        _original_clipmodel = CLIPModel.from_pretrained
        def _patched_clipmodel(cls, pretrained_model_name_or_path, **kwargs):
            name_map = {
                "openai/clip-vit-large-patch14-336": "/home/user/shared/models/clip-vit-large-patch14-336",
            }
            if pretrained_model_name_or_path in name_map:
                pretrained_model_name_or_path = name_map[pretrained_model_name_or_path]
            kwargs['local_files_only'] = True
            return _original_clipmodel(pretrained_model_name_or_path, **kwargs)
        CLIPModel.from_pretrained = classmethod(_patched_clipmodel.__get__(None, CLIPModel))

        # Register LlavaLlamaForCausalLM with AutoModelForCausalLM
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

        # Create config from model directory
        import json
        with open(os.path.join(model_path, "config.json")) as f:
            config_dict = json.load(f)
        # Override model_type to match the registered llava_llama type
        config_dict["model_type"] = "llava_llama"
        config = LlavaConfig.from_dict(config_dict)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

        # Load processor from transformers
        from transformers.models.llava.processing_llava import LlavaProcessor
        self.processor = LlavaProcessor.from_pretrained(model_path)

        self._LlavaLlamaForCausalLM = LlavaLlamaForCausalLM
        print("Model loaded successfully.")

    def predict(self, image_path: str, prompt: str) -> str:
        import torch
        from PIL import Image
        from llava.conversation import default_conversation
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        image = Image.open(image_path).convert("RGB")

        # Insert image tokens at the start of the user's message
        # LLaVA-1.5 expects <im_start><image><im_end> to be present in the prompt
        img_prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        qs = img_prefix + prompt

        conv = default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # Process inputs
        input_dict = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        input_ids = input_dict['input_ids'].to(self.model.device)
        pixel_values = input_dict['pixel_values'].to(self.model.device)

        with torch.no_grad():
            # Call generate with inputs (positional) and images (kwarg)
            # LlavaLlamaForCausalLM.generate signature: (inputs, images, image_sizes, **kwargs)
            output_ids = self.model.generate(
                input_ids,
                images=pixel_values,
                max_new_tokens=256,
                do_sample=False,
            )

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response


def get_vlm_client(client_type: str = "auto", **kwargs) -> VLMClient:
    """
    Factory function to get a VLM client.

    Args:
        client_type: One of "gpt4o", "qwen_vl", "llava", "auto"
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        A VLMClient instance
    """
    if client_type == "gpt4o":
        return GPT4oClient(**kwargs)
    elif client_type == "qwen_vl":
        return QwenVLClient(**kwargs)
    elif client_type == "llava":
        return LLaVAClient(**kwargs)
    elif client_type == "auto":
        # Try to use GPT-4o if API key is available, otherwise fall back to local
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return GPT4oClient(api_key=api_key)
        # Try local VLMs
        try:
            return QwenVLClient(**kwargs)
        except Exception as e:
            print(f"QwenVLClient failed: {e}")
            try:
                return LLaVAClient(**kwargs)
            except Exception as e2:
                raise RuntimeError(
                    f"No VLM available. Set OPENAI_API_KEY for GPT-4o, "
                    f"or ensure Qwen2.5-VL/LLaVA models are downloaded. Errors: {e}, {e2}"
                )
    else:
        raise ValueError(f"Unknown client type: {client_type}")
