import os
import json
import requests
from enum import Enum
from torch import Tensor
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Any

from ..utils import ensure_package, tensor2pil, pil2base64

gpt_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-vision-preview",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-0613",
    "gpt-4",
]

gpt_vision_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4-vision-preview"]

claude3_models = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]
claude2_models = ["claude-2.1"]

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

bedrock_claude3_models = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0",
]

bedrock_claude2_models = [
    "anthropic.claude-v2",
    "anthropic.claude-v2.1",
]

bedrock_mistral_models = [
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-large-2402-v1:0",
]

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
                    "source": {"type": "base64", "media_type": "image/png", "data": self.image},
                },
            )

        return {
            "role": self.role,
            "content": content,
        }


class OpenAIApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://api.openai.com/v1"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model not in gpt_models:
            raise Exception(f"Must provide an OpenAI model, got {config.model}")

        formated_messages = [m.to_openai_message() for m in messages]

        url = f"{self.endpoint}/chat/completions"
        data = {
            "messages": formated_messages,
            "model": config.model,
            "max_tokens": config.max_token,
            "temperature": config.temperature,
            # "seed": seed,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            raise Exception(data.get("error").get("message"))

        return data["choices"][0]["message"]["content"]

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        messages = [LLMMessage(role=LLMMessageRole.user, text=prompt)]

        return self.chat(messages, config, seed)


class ClaudeApi(BaseModel):
    api_key: str
    endpoint: Optional[str] = "https://api.anthropic.com/v1"
    version: Optional[str] = "2023-06-01"
    timeout: Optional[int] = 60

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model not in claude3_models:
            raise Exception(f"Must provide a Claude v3 model, got {config.model}")

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
            raise Exception(data.get("error").get("message"))

        return data["content"][0]["text"]

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        if config.model not in claude2_models:
            raise Exception(f"Must provide a Claude v2 model, got {config.model}")

        prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        url = f"{self.endpoint}/complete"
        data = {
            "prompt": prompt,
            "max_tokens_to_sample": config.max_token,
            "temperature": config.temperature,
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": self.version}

        response = requests.post(url, json=data, headers=headers, timeout=self.timeout)
        data: Dict = response.json()

        if data.get("error", None) is not None:
            raise Exception(data.get("error").get("message"))

        return data["completion"]


class AwsBedrockMistralApi(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str] = None
    region: Optional[str] = aws_regions[0]
    timeout: Optional[int] = 60
    bedrock_runtime: Any = None

    def __init__(self, **data):
        super().__init__(**data)

        ensure_package("boto3", version="1.34.101")
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
            raise Exception(data.get("error").get("message"))

        return data["outputs"][0]["text"]


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

        ensure_package("boto3", version="1.34.101")
        import boto3

        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region_name=self.region,
        )

    def chat(self, messages: List[LLMMessage], config: LLMConfig, seed=None):
        if config.model not in bedrock_claude3_models:
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
            raise Exception(data.get("error").get("message"))

        return data["content"][0]["text"]

    def complete(self, prompt: str, config: LLMConfig, seed=None):
        if config.model not in bedrock_claude2_models:
            raise Exception(f"Must provide a Claude v2 model, got {config.model}")

        prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        data = {
            "prompt": prompt,
            "max_tokens_to_sample": config.max_token,
            "temperature": config.temperature,
        }

        response = self.bedrock_runtime.invoke_model(body=json.dumps(data), modelId=config.model)
        data: Dict = json.loads(response.get("body").read())

        if data.get("error", None) is not None:
            raise Exception(data.get("error").get("message"))

        return data["completion"]


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

        return (OpenAIApi(api_key=openai_api_key, endpoint=endpoint),)


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
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            raise Exception("Claude API key is required.")

        return (ClaudeApi(api_key=claude_api_key, endpoint=endpoint, version=version),)


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
                    gpt_models
                    + claude3_models
                    + claude2_models
                    + bedrock_claude3_models
                    + bedrock_claude2_models
                    + bedrock_mistral_models,
                    {"default": gpt_vision_models[0]},
                ),
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

    def make_message(self, role, text, image: Optional[Tensor] = None, messages: Optional[List[LLMMessage]] = None):
        messages = [] if messages is None else messages.copy()

        if role == "system":
            if isinstance(image, Tensor):
                raise Exception("System prompt does not support image.")

            system_message = [m for m in messages if m.role == "system"]
            if len(system_message) > 0:
                raise Exception("Only one system prompt is allowed.")

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x1FFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, messages: List[LLMMessage], api: LLMApi, config: LLMConfig, seed):
        response = api.chat(messages, config, seed)
        return (response,)


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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, prompt: str, api: LLMApi, config: LLMConfig, seed):
        response = api.complete(prompt, config, seed)
        return (response,)


NODE_CLASS_MAPPINGS = {
    "AV_OpenAIApi": OpenAIApiNode,
    "AV_ClaudeApi": ClaudeApiNode,
    "AV_AwsBedrockClaudeApi": AwsBedrockClaudeApiNode,
    "AV_AwsBedrockMistralApi": AwsBedrockMistralApiNode,
    "AV_LLMApiConfig": LLMApiConfigNode,
    "AV_LLMMessage": LLMMessageNode,
    "AV_LLMChat": LLMChatNode,
    "AV_LLMCompletion": LLMCompletionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_OpenAIApi": "OpenAI API",
    "AV_ClaudeApi": "Claude API",
    "AV_AwsBedrockClaudeApi": "AWS Bedrock Claude API",
    "AV_AwsBedrockMistralApi": "AWS Bedrock Mistral API",
    "AV_LLMApiConfig": "LLM API Config",
    "AV_LLMMessage": "LLM Message",
    "AV_LLMChat": "LLM Chat",
    "AV_LLMCompletion": "LLM Completion",
}
