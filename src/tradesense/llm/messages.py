"""Provider-neutral request / response format.

Adapters translate :class:`LLMRequest` into each SDK's native shape, so prompts are
written once and work with every provider.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64")

    type: Literal["image"] = "image"
    data: bytes
    media_type: ImageMediaType

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.b64}"


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentPart]

    @classmethod
    def user(cls, *parts: str | ImagePart) -> Message:
        return cls(
            role="user",
            content=[TextPart(text=p) if isinstance(p, str) else p for p in parts],
        )

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextPart(text=text)])

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.content if isinstance(p, ImagePart)]


class LLMRequest(BaseModel):
    """What the application asks for, independent of any provider."""

    system: str = ""
    messages: list[Message]
    temperature: float | None = None  # None -> llm.yml settings.temperature
    max_tokens: int | None = None  # None -> llm.yml settings.max_output_tokens
    json_mode: bool = True
    #: Optional JSON schema; providers with native structured output may enforce it.
    json_schema: dict[str, Any] | None = None
    #: What the request is for. Decision-only providers serve only ``"decision"``.
    task: Literal["decision"] | None = None
    #: Skip local providers (Ollama, local servers): for bulk work they are too slow.
    remote_only: bool = False

    @property
    def has_images(self) -> bool:
        return any(m.images for m in self.messages)

    def with_followup(self, assistant_text: str, user_text: str) -> LLMRequest:
        """Append an assistant turn and a user turn (used for JSON error correction)."""
        return self.model_copy(
            update={
                "messages": [
                    *self.messages,
                    Message.assistant(assistant_text),
                    Message.user(user_text),
                ]
            }
        )


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
