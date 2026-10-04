"""Adapter tests: request translation and SDK interaction, with every SDK mocked."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import httpx2
import ollama
import openai
import pytest
import respx
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from tests.fakes import VALID_JSON
from tradesense.config import LLMSettings, ProviderConfig
from tradesense.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMProvider,
    LLMQuotaError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransientError,
    LLMUnavailableError,
)
from tradesense.llm.factory import provider_types
from tradesense.llm.messages import ImagePart, LLMRequest, Message
from tradesense.llm.providers.anthropic_provider import AnthropicProvider
from tradesense.llm.providers.gemini_provider import GeminiProvider
from tradesense.llm.providers.ollama_provider import OllamaProvider
from tradesense.llm.providers.openai_compatible import OpenAICompatibleProvider
from tradesense.llm.providers.openai_provider import OpenAIProvider
from tradesense.llm.providers.openrouter_decisions import OpenRouterDecisionsProvider
from tradesense.llm.schemas import DECISION_JSON_SCHEMA, parse_decision_payload

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
SETTINGS = LLMSettings(temperature=0.1, max_output_tokens=777)
ENV = {"KEY": "secret-key-value"}


def make(cls: type[LLMProvider], **config: Any) -> Any:
    config.setdefault("api_key_env", "KEY")
    config.setdefault("capabilities", ["text", "vision"])
    return cls("p", ProviderConfig(type=cls.type_name, **config), SETTINGS, ENV)


def text_request(**kwargs: Any) -> LLMRequest:
    return LLMRequest(system="be precise", messages=[Message.user("hello")], **kwargs)


def image_request() -> LLMRequest:
    return LLMRequest(
        system="be precise",
        messages=[Message.user("what is this", ImagePart(data=PNG, media_type="image/png"))],
        json_schema=DECISION_JSON_SCHEMA,
    )


class Recorder:
    """Async callable recording kwargs; returns or raises scripted items."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, BaseException):
            raise item
        return item


def sdk_error(module: Any, cls_name: str, status: int, message: str = "error") -> Exception:
    request = httpx2.Request("POST", "https://api.example.com/v1")
    response = httpx2.Response(status, request=request)
    cls = getattr(module, cls_name)
    return cls(message, response=response, body=None)  # type: ignore[no-any-return]


def test_all_types_registered() -> None:
    assert {
        "anthropic",
        "openai",
        "openai_compatible",
        "gemini",
        "ollama",
        "openrouter_decisions",
    } <= set(provider_types())


# --------------------------------------------------------------------------- anthropic


def anthropic_reply(text: str = VALID_JSON) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=120, output_tokens=60),
    )


def test_anthropic_translation_text_and_image() -> None:
    provider = make(AnthropicProvider, model="claude-sonnet-5-5")
    params = provider.build_params(text_request())
    assert params["system"] == "be precise"  # top-level, not a message
    assert params["model"] == "claude-sonnet-5-5"
    assert params["max_tokens"] == 777
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert "temperature" not in params  # not part of the current Messages API

    params = provider.build_params(image_request())
    blocks = params["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(PNG).decode(),
    }
    assert params["output_config"]["format"]["type"] == "json_schema"


async def test_anthropic_generate_and_usage() -> None:
    provider = make(
        AnthropicProvider,
        model="claude-sonnet-5-5",
        pricing={"input_per_mtok": 3.0, "output_per_mtok": 15.0},
    )
    create = Recorder(anthropic_reply())
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    response = await provider.generate(text_request())
    assert response.text == VALID_JSON
    assert (response.input_tokens, response.output_tokens) == (120, 60)
    assert response.cost_usd == pytest.approx((120 * 3 + 60 * 15) / 1e6)
    assert response.provider == "p"


async def test_anthropic_structured_output_downgrade() -> None:
    provider = make(AnthropicProvider, model="m")
    rejected = sdk_error(anthropic, "BadRequestError", 400, "output_config.format not supported")
    create = Recorder(rejected, anthropic_reply())
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    await provider.generate(image_request())
    assert "output_config" in create.calls[0]
    assert "output_config" not in create.calls[1]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (sdk_error(anthropic, "RateLimitError", 429), LLMRateLimitError),
        (sdk_error(anthropic, "AuthenticationError", 401), LLMAuthError),
        (sdk_error(anthropic, "PermissionDeniedError", 403), LLMAuthError),
        (sdk_error(anthropic, "InternalServerError", 500), LLMTransientError),
        (sdk_error(anthropic, "APIStatusError", 529, "overloaded"), LLMTransientError),
        (sdk_error(anthropic, "BadRequestError", 400, "bad"), LLMBadRequestError),
        (anthropic.APITimeoutError(httpx2.Request("POST", "https://x")), LLMTransientError),
    ],
)
async def test_anthropic_error_mapping(error: Exception, expected: type[Exception]) -> None:
    provider = make(AnthropicProvider, model="m")
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=Recorder(error)))
    with pytest.raises(expected):
        await provider.generate(text_request())


async def test_anthropic_empty_response() -> None:
    provider = make(AnthropicProvider, model="m")
    provider._client = SimpleNamespace(
        messages=SimpleNamespace(create=Recorder(anthropic_reply("  ")))
    )
    with pytest.raises(LLMResponseError, match="empty"):
        await provider.generate(text_request())


async def test_missing_key_is_unavailable_without_calling_sdk() -> None:
    provider = AnthropicProvider(
        "p", ProviderConfig(type="anthropic", model="m", api_key_env="ABSENT"), SETTINGS, {}
    )
    assert not provider.configured
    assert "ABSENT" in (provider.missing_config_reason() or "")
    with pytest.raises(LLMUnavailableError):
        await provider.generate(text_request())


# --------------------------------------------------------------------------- openai


def openai_reply(text: str = VALID_JSON) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=90, completion_tokens=45),
    )


def test_openai_translation_text_and_image() -> None:
    provider = make(OpenAIProvider, model="gpt-4o")
    params = provider.build_params(text_request())
    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["messages"][1] == {"role": "user", "content": "hello"}
    assert params["response_format"] == {"type": "json_object"}
    assert params["max_completion_tokens"] == 777
    assert params["temperature"] == 0.1

    params = provider.build_params(image_request())
    parts = params["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()


def test_openai_options() -> None:
    provider = make(
        OpenAIProvider,
        model="o-model",
        json_mode=False,
        options={"token_param": "max_tokens", "send_temperature": False, "extra_body": {"x": 1}},
    )
    params = provider.build_params(text_request())
    assert "response_format" not in params
    assert "temperature" not in params
    assert params["max_tokens"] == 777
    assert params["extra_body"] == {"x": 1}


async def test_openai_generate_and_json_downgrade() -> None:
    provider = make(OpenAIProvider, model="gpt-4o")
    rejected = sdk_error(openai, "BadRequestError", 400, "'response_format' is not supported")
    create = Recorder(rejected, openai_reply())
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    response = await provider.generate(text_request())
    assert response.text == VALID_JSON
    assert (response.input_tokens, response.output_tokens) == (90, 45)
    assert "response_format" in create.calls[0]
    assert "response_format" not in create.calls[1]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (sdk_error(openai, "RateLimitError", 429), LLMRateLimitError),
        (sdk_error(openai, "AuthenticationError", 401), LLMAuthError),
        (sdk_error(openai, "InternalServerError", 503), LLMTransientError),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), LLMTransientError),
    ],
)
async def test_openai_error_mapping(error: Exception, expected: type[Exception]) -> None:
    provider = make(OpenAIProvider, model="gpt-4o")
    create = Recorder(error)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(expected):
        await provider.generate(text_request())


def test_azure_uses_deployment_and_azure_client() -> None:
    provider = make(
        OpenAIProvider,
        azure=True,
        endpoint="https://res.openai.azure.com",
        deployment="my-deploy",
        api_version="2024-10-21",
    )
    assert provider.build_params(text_request())["model"] == "my-deploy"
    assert isinstance(provider.make_client(), openai.AsyncAzureOpenAI)


def test_openai_compatible_local_needs_no_key() -> None:
    local = OpenAICompatibleProvider(
        "vllm",
        ProviderConfig(
            type="openai_compatible",
            model="m",
            base_url="http://localhost:8001/v1",
            api_key_env="VLLM_API_KEY",
        ),
        SETTINGS,
        {},
    )
    assert local.configured
    params = local.build_params(text_request())
    assert params["max_tokens"] == 777  # widest-compatible token parameter
    client = local.make_client()
    assert str(client.base_url).startswith("http://localhost:8001/v1")

    remote = OpenAICompatibleProvider(
        "groq",
        ProviderConfig(
            type="openai_compatible",
            model="m",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
        ),
        SETTINGS,
        {},
    )
    assert not remote.configured


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider(
            "x", ProviderConfig(type="openai_compatible", model="m"), SETTINGS, {}
        )


# --------------------------------------------------------------------------- gemini


def test_gemini_translation_text_and_image() -> None:
    provider = make(GeminiProvider, model="gemini-2.5-flash", options={"thinking_budget": 0})
    contents, config = provider.build_request(image_request())
    assert len(contents) == 1
    assert contents[0].role == "user"
    parts = contents[0].parts or []
    assert parts[0].text == "what is this"
    assert parts[1].inline_data is not None
    assert parts[1].inline_data.data == PNG
    assert parts[1].inline_data.mime_type == "image/png"
    assert config.system_instruction == "be precise"
    assert config.response_mime_type == "application/json"
    assert config.max_output_tokens == 777
    assert config.temperature == 0.1
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 0

    followup = text_request().with_followup("bad", "fix it")
    contents, _ = provider.build_request(followup)
    assert [c.role for c in contents] == ["user", "model", "user"]


async def test_gemini_generate_and_errors() -> None:
    provider = make(GeminiProvider, model="gemini-2.5-flash")
    reply = SimpleNamespace(
        text=VALID_JSON,
        usage_metadata=SimpleNamespace(
            prompt_token_count=80, candidates_token_count=30, thoughts_token_count=5
        ),
    )
    generate = Recorder(reply)
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    response = await provider.generate(text_request())
    assert (response.input_tokens, response.output_tokens) == (80, 35)
    assert generate.calls[0]["model"] == "gemini-2.5-flash"
    assert isinstance(generate.calls[0]["config"], genai_types.GenerateContentConfig)

    for code, expected in ((429, LLMRateLimitError), (403, LLMAuthError), (500, LLMTransientError)):
        error = genai_errors.APIError(code, {"error": {"message": "x", "status": "S"}})
        provider._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=Recorder(error)))
        )
        with pytest.raises(expected):
            await provider.generate(text_request())


# --------------------------------------------------------------------------- ollama


def test_ollama_translation_text_and_image() -> None:
    provider = make(
        OllamaProvider,
        model="qwen2.5vl:7b",
        api_key_env=None,
        options={"think": False, "num_ctx": 8192, "num_gpu": 0, "num_thread": True},
    )
    assert provider.configured  # no key needed
    params = provider.build_params(image_request())
    assert params["model"] == "qwen2.5vl:7b"
    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    user = params["messages"][1]
    assert user["content"] == "what is this"
    assert user["images"] == [base64.b64encode(PNG).decode()]
    assert params["format"] == "json"
    assert params["stream"] is False
    assert params["think"] is False
    # num_gpu=0 (CPU only) must survive the int check; a bool is not a thread count
    assert params["options"] == {
        "temperature": 0.1,
        "num_predict": 777,
        "num_ctx": 8192,
        "num_gpu": 0,
    }
    assert "images" not in provider.build_params(text_request())["messages"][1]


async def test_ollama_generate_and_errors() -> None:
    provider = make(OllamaProvider, model="llama3.1:8b", api_key_env=None)
    reply = SimpleNamespace(
        message=SimpleNamespace(content=VALID_JSON),
        prompt_eval_count=50,
        eval_count=20,
        done_reason="stop",
    )
    provider._client = SimpleNamespace(chat=Recorder(reply))
    response = await provider.generate(text_request())
    assert (response.input_tokens, response.output_tokens) == (50, 20)

    provider._client = SimpleNamespace(chat=Recorder(ollama.ResponseError("not found", 404)))
    with pytest.raises(LLMBadRequestError):
        await provider.generate(text_request())
    provider._client = SimpleNamespace(chat=Recorder(httpx.ConnectError("refused")))
    with pytest.raises(LLMTransientError, match="cannot reach Ollama"):
        await provider.generate(text_request())


async def test_ollama_health_check_missing_model() -> None:
    provider = make(OllamaProvider, model="absent:1b", api_key_env=None)
    provider._client = SimpleNamespace(show=Recorder(ollama.ResponseError("not found", 404)))
    assert await provider.health_check() is False


# --------------------------------------------------------------------------- openrouter decisions

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


def decisions_provider() -> OpenRouterDecisionsProvider:
    return make(OpenRouterDecisionsProvider, model="typesafe/jev-1.13", capabilities=["decision"])


def decision_request(**kwargs: Any) -> LLMRequest:
    return text_request(task="decision", **kwargs)


def test_decisions_provider_serves_only_text_decisions() -> None:
    provider = decisions_provider()
    assert provider.supports(decision_request())
    assert not provider.supports(text_request())  # extraction / summaries
    assert not provider.supports(image_request().model_copy(update={"task": "decision"}))
    assert not provider.generates_text
    body = provider.build_body(decision_request().with_followup("bad", "fix it"))
    assert body["model"] == "typesafe/jev-1.13"
    assert body["state"] == "hello"  # correction turns are not part of the state
    question = body["questions"]["signal"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"BUY", "SELL", "SIDEWAYS"}


@respx.mock
async def test_decisions_provider_answer_becomes_a_decision() -> None:
    route = respx.post(DECISIONS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "signal": {
                        "type": "choice",
                        "choice": "BUY",
                        "probabilities": {"SELL": 0.01, "SIDEWAYS": 0.09, "BUY": 0.9},
                        "confidence": 0.86,
                    }
                },
                "usage": {"input_tokens": 439, "output_tokens": 39, "cost": 1.8e-05},
            },
        )
    )
    provider = decisions_provider()
    response = await provider.generate(decision_request())
    assert route.calls.last.request.headers["Authorization"] == "Bearer secret-key-value"
    assert (response.input_tokens, response.output_tokens) == (439, 39)
    decision = parse_decision_payload(response.text)
    assert decision.signal == "BUY"
    assert decision.probability.as_dict() == {"buy": 0.9, "sell": 0.01, "sideways": 0.09}
    assert decision.confidence == 0.86
    assert decision.key_levels.support is None
    assert "BUY 90%" in decision.observations[0]

    with pytest.raises(LLMBadRequestError, match="only answer"):
        await provider.generate(text_request())
    await provider.aclose()


@respx.mock
async def test_decisions_provider_errors_and_health() -> None:
    provider = decisions_provider()
    respx.post(DECISIONS_URL).mock(
        side_effect=[
            httpx.Response(401, json={"error": {"code": 401, "message": "No auth"}}),
            httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}}),
            httpx.Response(529, json={"error": {"code": 529, "message": "Overloaded"}}),
            httpx.Response(200, json={"answers": {}}),
        ]
    )
    with pytest.raises(LLMAuthError, match="No auth"):
        await provider.generate(decision_request())
    with pytest.raises(LLMQuotaError):
        await provider.generate(decision_request())
    with pytest.raises(LLMTransientError):
        await provider.generate(decision_request())
    with pytest.raises(LLMResponseError, match="unexpected response"):
        await provider.generate(decision_request())

    respx.get("https://openrouter.ai/api/v1/key").mock(
        side_effect=[httpx.Response(200, json={"data": {}}), httpx.Response(401, json={})]
    )
    assert await provider.health_check() is True
    with pytest.raises(LLMAuthError):
        await provider.health_check()
    await provider.aclose()
