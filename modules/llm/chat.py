import os
import json
import base64
import requests
import torch
from io import BytesIO
from enum import Enum
from torch import Tensor
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any
from PIL import Image

from ..utils import ensure_package, tensor2pil, pil2base64, pil2tensor


def image_urls_to_tensor(image_urls: List[str], timeout: int = 60) -> Optional[Tensor]:
    tensors: List[Tensor] = []

    for url in image_urls:
        try:
            if isinstance(url, str) and url.startswith("data:"):
                comma_index = url.find(",")
                if comma_index == -1:
                    continue
                header = url[:comma_index]
                data_part = url[comma_index + 1 :]

                if ";base64" in header:
                    raw_bytes = base64.b64decode(data_part)
                else:
                    raw_bytes = data_part.encode("utf-8")

                image = Image.open(BytesIO(raw_bytes)).convert("RGB")
            else:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content)).convert("RGB")

            tensor = pil2tensor(image)
            tensors.append(tensor)
        except Exception:
            # Silently skip any image that fails to load/parse
            continue

    if tensors:
        return torch.cat(tensors, dim=0)

    return None


gpt_models = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat-latest",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "o3",
    "o3-mini",
    "o3-pro",
    "o4-mini",
]


claude_models = [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

gemini_models = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

aws_regions = [
    "us-east-1",
    "us-west-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-northeast-1",
    "eu-central-1",
    "eu-west-3",
    "eu-west-1",
    "ap-south-3",
]

bedrock_anthropic_versions = ["bedrock-2023-05-31"]

bedrock_claude_models = [
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
]

bedrock_mistral_models = [
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

all_models = gpt_models + claude_models + gemini_models + bedrock_claude_models + bedrock_mistral_models

default_system_prompt = "You are a useful AI agent."


class LLMConfig(BaseModel):
    model: str
    max_token: int
    temperature: float
    modalities: str = "image+text"
    aspect_ratio: str = "auto"


class LLMMessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class LLMMessage(BaseModel):
    role: LLMMessageRole = LLMMessageRole.user
    text: str
    images: Optional[List[str]] = None  # list of base64 encoded images

    def to_openai_message(self):
        content = [{"type": "text", "text": self.text}]

        if self.images:
            for img in self.images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        return {
            "role": self.role,
            "content": content,
        }

    def to_claude_message(self):
        content = [{"type": "text", "text": self.text}]

        if self.images:
            for img in reversed(self.images):
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img},
                    }
                )

        return {
            "role": self.role,
            "content": content,
        }

    def to_gemini_message(self):
        parts = [{"text": self.text}]

        if self.images:
            for img in self.images:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img,
                        }
                    }
                )

        # Gemini uses "model" and "user" roles instead of "assistant" and "user"
        role = "model" if self.role == "assistant" else "user"

        return {
            "role": role,
            "parts": parts,
        }


class OpenAIApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://api.openai.com/v1"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model in all_models and config.model not in gpt_models:
            raise Exception(f"Must provide an OpenAI model, got {config.model}")

        formated_messages = [m.to_openai_message() for m in messages]

        url = f"{self.endpoint}/chat/completions"
        data = {
            "messages": formated_messages,
            "model": config.model,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            return (f"OpenAI API error: {data.get('error').get('message')}", None)

        text = data["choices"][0]["message"]["content"]
        return (text, None)

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]

        return self.chat(messages, config, seed)


class OpenRouterApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://openrouter.ai/api/v1"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        formated_messages = [m.to_openai_message() for m in messages]

        url = f"{self.endpoint}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = {
            "messages": formated_messages,
            "model": config.model,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
        }

        if "gemini-2.5-flash-image" in config.model:
            data["model"] = "google/gemini-2.5-flash-image"

            modalities = (config.modalities or "image+text").split("+")
            modalities = [modality.strip().lower() for modality in modalities]
            data["modalities"] = modalities

            if config.aspect_ratio and config.aspect_ratio != "auto":
                data["image_config"] = {
                    "aspect_ratio": config.aspect_ratio,
                }

        response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            return (f"OpenRouter API error: {data.get('error').get('message')}", None)

        message = data["choices"][0]["message"]
        text = message["content"]
        images = None

        if message.get("images", None) is not None:
            urls: List[str] = []
            for m in message["images"]:
                if isinstance(m, dict):
                    if "image_url" in m and isinstance(m["image_url"], dict) and "url" in m["image_url"]:
                        urls.append(m["image_url"]["url"])

            images = image_urls_to_tensor(urls, timeout=self.timeout)

        return (text, images)

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]

        return self.chat(messages, config, seed)


class ClaudeApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://api.anthropic.com/v1"
    version: Optional[str] = "2023-06-01"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model in all_models and config.model not in claude_models:
            raise Exception(f"Must provide a Claude model, got {config.model}")

        system_message = [m for m in messages if m.role == "system"]
        user_messages = [m for m in messages if m.role != "system"]
        formated_messages = [m.to_claude_message() for m in user_messages]

        url = f"{self.endpoint}/messages"
        data = {
            "messages": formated_messages,
            "model": config.model,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
            "system": system_message[0].text if len(system_message) > 0 else None,
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}

        response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            return (data.get("error").get("message"), None)

        text = data["content"][0]["text"]
        return (text, None)

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]
        return self.chat(messages, config, seed)


class GeminiApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://generativelanguage.googleapis.com/v1beta"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model in all_models and config.model not in gemini_models:
            raise Exception(f"Must provide a Gemini model, got {config.model}")

        system_message = [m for m in messages if m.role == "system"]
        user_messages = [m for m in messages if m.role != "system"]
        formated_messages = [m.to_gemini_message() for m in user_messages]

        url = f"{self.endpoint}/models/{config.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}

        modalities = (config.modalities or "text+image").split("+")
        responseModalities = [modality.strip().capitalize() for modality in modalities]
        aspectRatio = config.aspect_ratio if (config.aspect_ratio and config.aspect_ratio != "auto") else None

        data = {
            "contents": formated_messages,
            "generationConfig": {
                "maxOutputTokens": config.max_token,
                "temperature": config.temperature,
                "responseModalities": responseModalities,
                "imageConfig": (
                    {
                        "aspectRatio": aspectRatio,
                    }
                    if aspectRatio
                    else None
                ),
            },
        }

        # Add system instruction if provided
        if len(system_message) > 0:
            data["systemInstruction"] = {"parts": [{"text": system_message[0].text}]}

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            error_message = data.get("error").get("message", "Unknown error")
            return (f"Gemini API error: {error_message}", None)

        # Extract text and images from response
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]

                # Collect text parts
                text_parts = []
                image_urls = []

                for part in parts:
                    # Handle text
                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])

                    # Handle inline images
                    elif "inlineData" in part and part["inlineData"]:
                        image_urls.append(f"data:{part['inlineData']['mimeType']};base64,{part['inlineData']['data']}")

                text = "".join(text_parts) if text_parts else "No text response from Gemini API"
                images = image_urls_to_tensor(image_urls, timeout=self.timeout)

                return (text, images)

        return (f"No response from Gemini API", None)

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]
        return self.chat(messages, config, seed)


class AwsBedrockMistralApi(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str] = None
    region: Optional[str] = aws_regions[0]
    timeout: Optional[int] = 60
    bedrock_runtime: Any = None

    def __init__(self, **data):
        super().__init__(**data)

        ensure_package("boto3", required_version=">=1.34.101")
        import boto3

        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region_name=self.region,
        )

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        raise Exception("Mistral doesn't support chat API")

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        if config.model not in bedrock_mistral_models:
            raise Exception(f"Must provide a Mistral model, got {config.model}")

        prompt = f"<s>[INST]{prompt}[/INST]"
        data = {
            "prompt": prompt,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
        }

        response = self.bedrock_runtime.invoke_model(body=json.dumps(data), modelId=config.model)
        data: Dict = json.loads(response.get("body").read())

        if data.get("error", None) is not None:
            return (f"Mistral API error: {data.get('error').get('message')}", None)

        text = data["outputs"][0]["text"]
        return (text, None)


class AwsBedrockClaudeApi(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str] = None
    region: Optional[str] = aws_regions[0]
    version: Optional[str] = bedrock_anthropic_versions[0]
    timeout: Optional[int] = 60
    bedrock_runtime: Any = None

    def __init__(self, **data):
        super().__init__(**data)

        ensure_package("boto3", required_version=">=1.34.101")
        import boto3

        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region_name=self.region,
        )

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model not in bedrock_claude_models:
            raise Exception(f"Must provide a Claude v3 model, got {config.model}")

        system_message = [m for m in messages if m.role == "system"]
        user_messages = [m for m in messages if m.role != "system"]
        formated_messages = [m.to_claude_message() for m in user_messages]

        data = {
            "anthropic_version": self.version,
            "messages": formated_messages,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
            "system": system_message[0].text if len(system_message) > 0 else None,
        }

        response = self.bedrock_runtime.invoke_model(body=json.dumps(data), modelId=config.model)
        data: Dict = json.loads(response.get("body").read())

        if data.get("error", None) is not None:
            return (f"Claude API error: {data.get('error').get('message')}", None)

        text = data["content"][0]["text"]
        return (text, None)

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]
        return self.chat(messages, config, seed)


LLMApi = Union[OpenAIApi, OpenRouterApi, ClaudeApi, GeminiApi]


class OpenAIApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, openai_api_key, endpoint):
        if not openai_api_key or openai_api_key == "":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OpenAI API key is required.")

        return (OpenAIApi(api_key=openai_api_key, endpoint=endpoint),)


class OpenRouterApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openrouter_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://openrouter.ai/api/v1"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, openrouter_api_key, endpoint):
        if not openrouter_api_key or openrouter_api_key == "":
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise Exception("OpenRouter API key is required.")

        return (OpenRouterApi(api_key=openrouter_api_key, endpoint=endpoint),)


class ClaudeApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "claude_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.anthropic.com/v1"}),
                "version": (["2023-06-01"], {"default": "2023-06-01"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, claude_api_key, endpoint, version):
        if not claude_api_key or claude_api_key == "":
            claude_api_key = os.environ.get("ANTHROPIC_API_KEY", os.environ.get("CLAUDE_API_KEY"))
        if not claude_api_key:
            raise Exception("Anthropic API key is required.")

        return (ClaudeApi(api_key=claude_api_key, endpoint=endpoint, version=version),)


class GeminiApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {"multiline": False}),
                "endpoint": (
                    "STRING",
                    {"multiline": False, "default": "https://generativelanguage.googleapis.com/v1beta"},
                ),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, gemini_api_key, endpoint):
        if not gemini_api_key or gemini_api_key == "":
            gemini_api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        if not gemini_api_key:
            raise Exception("Gemini API key is required.")

        return (GeminiApi(api_key=gemini_api_key, endpoint=endpoint),)


class AwsBedrockMistralApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aws_access_key_id": ("STRING", {"multiline": False}),
                "aws_secret_access_key": ("STRING", {"multiline": False}),
                "aws_session_token": ("STRING", {"multiline": False}),
                "region": (aws_regions, {"default": aws_regions[0]}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, aws_access_key_id, aws_secret_access_key, aws_session_token, region):
        if not aws_access_key_id or aws_access_key_id == "":
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        if not aws_secret_access_key or aws_secret_access_key == "":
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        if not aws_session_token or aws_session_token == "":
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

        if not aws_access_key_id or not aws_secret_access_key:
            raise Exception("AWS credentials is required.")

        return (
            AwsBedrockMistralApi(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region=region,
            ),
        )


class AwsBedrockClaudeApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aws_access_key_id": ("STRING", {"multiline": False}),
                "aws_secret_access_key": ("STRING", {"multiline": False}),
                "aws_session_token": ("STRING", {"multiline": False}),
                "region": (aws_regions, {"default": aws_regions[0]}),
                "version": (bedrock_anthropic_versions, {"default": bedrock_anthropic_versions[0]}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, aws_access_key_id, aws_secret_access_key, aws_session_token, region, version):
        if not aws_access_key_id or aws_access_key_id == "":
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
        if not aws_secret_access_key or aws_secret_access_key == "":
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        if not aws_session_token or aws_session_token == "":
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

        if not aws_access_key_id or not aws_secret_access_key:
            raise Exception("AWS credentials is required.")

        return (
            AwsBedrockClaudeApi(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region=region,
                version=version,
            ),
        )


class LLMApiConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    gpt_models + claude_models + gemini_models + bedrock_claude_models + bedrock_mistral_models,
                    {"default": gpt_models[0]},
                ),
                "max_token": ("INT", {"default": 1024, "min": 1, "max": 102400}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.001}),
            },
            "optional": {"custom_model": ("STRING", {"multiline": False, "default": ""})},
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "make_config"
    CATEGORY = "ArtVenture/LLM"

    def make_config(self, model, max_token, temperature, custom_model=""):
        # Use custom_model if provided, otherwise use the selected model from dropdown
        final_model = custom_model.strip() if custom_model and custom_model.strip() else model
        return (LLMConfig(model=final_model, max_token=max_token, temperature=temperature),)


class NanoBananaApiConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.001}),
                "modalities": (["image+text", "image", "text"], {"default": "image+text"}),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    {"default": "auto"},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "make_config"
    CATEGORY = "ArtVenture/LLM"

    def make_config(self, temperature, modalities="image+text", aspect_ratio="auto"):
        return (
            LLMConfig(
                model="gemini-2.5-flash-image",
                max_token=8192,
                temperature=temperature,
                modalities=modalities,
                aspect_ratio=aspect_ratio,
            ),
        )


class LLMMessageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["system", "user", "assistant"],),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "messages": ("LLM_MESSAGE",),
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LLM_MESSAGE",)
    RETURN_NAMES = ("messages",)
    FUNCTION = "make_message"
    CATEGORY = "ArtVenture/LLM"

    def make_message(
        self,
        role,
        text,
        messages: Optional[List[LLMMessage]] = None,
        image: Optional[Tensor] = None,
        image_2: Optional[Tensor] = None,
        image_3: Optional[Tensor] = None,
        image_4: Optional[Tensor] = None,
    ):
        messages = [] if messages is None else messages.copy()

        if role == "system":
            system_message = [m for m in messages if m.role == "system"]
            if len(system_message) > 0:
                raise Exception("Only one system prompt is allowed.")

            if any(isinstance(img, Tensor) for img in [image, image_2, image_3, image_4]):
                raise Exception("System prompt does not support image.")

        all_images = []
        for img_tensor in [image, image_2, image_3, image_4]:
            if isinstance(img_tensor, Tensor):
                if len(img_tensor.shape) == 4:  # Batch of images
                    for i in range(img_tensor.shape[0]):
                        pil = tensor2pil(img_tensor[i])
                        content = pil2base64(pil)
                        all_images.append(content)
                else:  # Single image
                    pil = tensor2pil(img_tensor)
                    content = pil2base64(pil)
                    all_images.append(content)

        if all_images:
            messages.append(LLMMessage(role=role, text=text, images=all_images))
        else:
            messages.append(LLMMessage(role=role, text=text))

        return (messages,)


class LLMChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LLM_MESSAGE",),
                "api": ("LLM_API",),
                "config": ("LLM_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x1FFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "(optional) images")
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, messages: List[LLMMessage], api: LLMApi, config: LLMConfig, seed):
        text, images = api.chat(messages, config, seed)
        return (text, images)


class LLMCompletionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "api": ("LLM_API",),
                "config": ("LLM_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x1FFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "(optional) images")
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, prompt: str, api: LLMApi, config: LLMConfig, seed):
        text, images = api.complete(prompt, config, seed)
        return (text, images)


NODE_CLASS_MAPPINGS = {
    "AV_OpenAIApi": OpenAIApiNode,
    "AV_OpenRouterApi": OpenRouterApiNode,
    "AV_ClaudeApi": ClaudeApiNode,
    "AV_GeminiApi": GeminiApiNode,
    "AV_AwsBedrockClaudeApi": AwsBedrockClaudeApiNode,
    "AV_AwsBedrockMistralApi": AwsBedrockMistralApiNode,
    "AV_LLMApiConfig": LLMApiConfigNode,
    "AV_NanoBananaApiConfig": NanoBananaApiConfigNode,
    "AV_LLMMessage": LLMMessageNode,
    "AV_LLMChat": LLMChatNode,
    "AV_LLMCompletion": LLMCompletionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_OpenAIApi": "OpenAI API",
    "AV_OpenRouterApi": "OpenRouter API",
    "AV_ClaudeApi": "Claude API",
    "AV_GeminiApi": "Gemini API",
    "AV_AwsBedrockClaudeApi": "AWS Bedrock Claude API",
    "AV_AwsBedrockMistralApi": "AWS Bedrock Mistral API",
    "AV_LLMApiConfig": "LLM API Config",
    "AV_NanoBananaApiConfig": "NanoBanana API Config",
    "AV_LLMMessage": "LLM Message",
    "AV_LLMChat": "LLM Chat",
    "AV_LLMCompletion": "LLM Completion",
}
