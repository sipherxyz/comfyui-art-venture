import os
from enum import Enum
from torch import Tensor
from pydantic import BaseModel
from typing import List, Union, Optional

from ..utils import tensor2pil, pil2base64

gpt_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-4",
]

gpt_vision_models = ["gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4-vision-preview"]

claude_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

default_system_prompt = "You are a useful AI agent."


class LLMConfig(BaseModel):
    model: str
    max_token: int
    temperature: float


class LLMMessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class LLMMessage(BaseModel):
    role: LLMMessageRole = LLMMessageRole.user
    text: str
    image: Optional[str] = None  # base64 enoded image

    def to_openai_message(self):
        content = [{"type": "text", "text": self.text}]

        if self.image:
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.text}"}})

        return {
            "role": self.role,
            "content": content,
        }

    def to_claude_message(self):
        content = [{"type": "text", "text": self.text}]

        if self.image:
            content.insert(
                0,
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": self.text},
                },
            )

        return {
            "role": self.role,
            "content": content,
        }


class OpenAIApi:
    def __init__(self, openai_api_key: str, endpoint: str) -> None:
        from openai import OpenAI

        self._type = "openai"
        self.client = OpenAI(api_key=openai_api_key, base_url=endpoint)

    def completion(self, messages: List[LLMMessage], config: LLMConfig):
        formated_messages = [m.to_openai_message() for m in messages]

        response = self.client.chat.completions.create(
            messages=formated_messages, model=config.model, max_tokens=config.max_token, temperature=config.temperature
        )
        content = response.choices[0].message.content

        return content


class ClaudeApi:
    def __init__(self, claude_api_key: str, endpoint: str) -> None:
        from anthropic import Anthropic

        self._type = "claude"
        self.client = Anthropic(api_key=claude_api_key, base_url=endpoint)

    def completion(self, messages: List[LLMMessage], config: LLMConfig):
        formated_messages = [m.to_claude_message() for m in messages]

        response = self.client.messages.create(
            messages=formated_messages,
            model=config.model,
            max_tokens=config.max_token,
            temperature=config.temperature,
        )
        content = response.content[0].text

        return content


LLMApi = Union[OpenAIApi, ClaudeApi]


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

        return (OpenAIApi(openai_api_key, endpoint),)


class ClaudeApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "claude_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.anthropic.com"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, claude_api_key, endpoint):
        if not claude_api_key or claude_api_key == "":
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            raise Exception("Claude API key is required.")

        return (ClaudeApi(claude_api_key, endpoint),)


class LLMApiConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (gpt_models + claude_models, {"default": gpt_vision_models[0]}),
                "max_token": ("INT", {"default": 1024}),
                "temperature": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "make_config"
    CATEGORY = "ArtVenture/LLM"

    def make_config(self, max_token, model, temperature):
        return (LLMConfig(model=model, max_token=max_token, temperature=temperature),)


class LLMMessageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["system", "user", "assistant"],),
                "text": ("STRING", {"multiline": True}),
            },
            "optional": {"image": ("IMAGE",), "messages": ("LLM_MESSAGE",)},
        }

    RETURN_TYPES = ("LLM_MESSAGE",)
    RETURN_NAMES = ("messages",)
    FUNCTION = "make_message"
    CATEGORY = "ArtVenture/LLM"

    def make_message(self, role, text, image=None, messages=None):
        if messages is None:
            messages = []

        if isinstance(image, Tensor):
            pil = tensor2pil(image)
            content = pil2base64(pil)
            messages.append(LLMMessage(role=role, text=text, image=content))
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, messages: List[LLMMessage], api: LLMApi, config: LLMConfig):
        try:
            response = api.completion(messages, config)
            return (response,)
        except Exception as e:
            return (f"Error: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "AV_OpenAIApi": OpenAIApiNode,
    "AV_ClaudeApi": ClaudeApiNode,
    "AV_LLMApiConfig": LLMApiConfigNode,
    "AV_LLMMessage": LLMMessageNode,
    "AV_LLMChat": LLMChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_OpenAIApi": "OpenAI Api",
    "AV_ClaudeApi": "Claude Api",
    "AV_LLMApiConfig": "LLM Api Config",
    "AV_LLMMessage": "LLM Message",
    "AV_LLMChat": "LLM Chat",
}
