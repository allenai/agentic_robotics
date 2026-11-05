"""Define a class providing a client to call models (i.e., LLMs/VLMs) using the OpenAI API."""

import time
from dataclasses import dataclass
from datetime import datetime
from os import getenv
from typing import Any, Literal, cast

import httpx
from mcp_types import Tool as MCPTool
from openai import AsyncOpenAI, omit
from openai.types.responses import Response, ResponseInputItemParam, ResponseOutputItem

from agentic_robotics.core.foundation_models.model_client import History, ModelClient
from agentic_robotics.core.foundation_models.usage_tracking import TokenUsageSnapshot
from agentic_robotics.core.mcp_conversion import mcp_tool_to_openai_tool


def _openai_output_to_input_item(item: ResponseOutputItem) -> ResponseInputItemParam:
    """Convert an OpenAI model output into the equivalent Responses API input item."""
    return cast("ResponseInputItemParam", item.model_dump(exclude_unset=True, mode="json"))


ConversationRole = Literal["user", "assistant", "developer", "system"]


@dataclass(frozen=True)
class OpenAIModelConfig:
    """Configuration for a model called using the OpenAI API."""

    model_id: str
    base_url: str = "https://openrouter.ai/api/v1"

    max_output_tokens: int = 5000
    """Maximum output tokens permitted per response (default: 5000)."""

    caching: bool = True
    """If True, the model client will attempt to cache the conversation as it continues."""

    allowed_providers: tuple[str, ...] | None = None
    """Optional OpenRouter-specific setting to specify allowed providers (needed for caching)."""


OPENAI_API_MODEL_CONFIGS: dict[str, OpenAIModelConfig] = {
    "haiku-4.5": OpenAIModelConfig(
        model_id="anthropic/claude-haiku-4.5", allowed_providers=("anthropic",)
    ),
    "sonnet-5.5": OpenAIModelConfig(
        model_id="anthropic/claude-sonnet-5.5", allowed_providers=("anthropic",)
    ),
    "opus-5.5": OpenAIModelConfig(
        model_id="anthropic/claude-opus-5.5", allowed_providers=("anthropic",)
    ),
    "fable-5.1": OpenAIModelConfig(
        model_id="anthropic/claude-fable-5.1", allowed_providers=("anthropic",)
    ),
    "gpt-6.1-sol-pro": OpenAIModelConfig(
        model_id="openai/gpt-6.1-sol-pro", caching=False, allowed_providers=("openai",)
    ),
    # "qwen-3.8": OpenAIModelConfig(model_id="qwen3.8:27b", base_url="http://localhost:11434/v1/"),
}


class OpenAIModelClient(ModelClient):
    """A thin wrapper class for an OpenAI API client."""

    def __init__(
        self,
        config: OpenAIModelConfig,
        env_api_key: str = "OPENROUTER_API_KEY",
        default_timeout_s: float = 90.0,
    ) -> None:
        """Initialize an OpenAI API client for the specified model.

        :param config: Configuration specifying the model identifier, base URL, etc.
        :param env_api_key: Environment variable storing the API key (default: "OPENROUTER_API_KEY")
        :param default_timeout_s: Default timeout (seconds) for all operations (default: 90 s)
        """
        self._api_key = getenv(env_api_key)
        assert self._api_key, f"Unable to find API key from environment variable '{env_api_key}'."

        self.config = config
        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self._api_key,
            timeout=httpx.Timeout(default_timeout_s, connect=5.0),
        )

        if self.config.caching and not self.config.allowed_providers:
            raise RuntimeError("Cannot cache model prompts without providers in the allowlist.")

        providers = list(self.config.allowed_providers) if self.config.allowed_providers else []
        self._extra_body = {"provider": {"only": providers, "allow_fallbacks": False}}

        self._usage_history: list[TokenUsageSnapshot] = []
        self._last_request: dict[str, Any] | None = None

    @property
    def usage_history(self) -> list[TokenUsageSnapshot]:
        """Retrieve a read-only copy of the model client's usage history."""
        return list(self._usage_history)

    def _record_usage(self, response: Response, started_at: str, start_ns: int) -> None:
        """Record token usage from the given OpenAI API response."""
        end_ns = time.monotonic_ns()
        duration_s = round((end_ns - start_ns) / 1e9, ndigits=3)

        # Record the token usage from this call (if those data are available)
        if response.usage:
            details = response.usage.input_tokens_details
            model_extra = response.usage.model_extra

            usage_snapshot = TokenUsageSnapshot(
                input_tokens=(response.usage.input_tokens or 0),
                tokens_read=(details.cached_tokens or 0),
                tokens_written=(details.cache_write_tokens or 0),
                cost_usd=(model_extra.get("cost", 0) if model_extra else 0),
                started_at=started_at,
                duration_s=duration_s,
            )
            self._usage_history.append(usage_snapshot)

    async def keep_alive(self) -> None:
        """Re-send the previous request with a minimal output cap to refresh the model cache."""
        if self._last_request is None:
            return
        started_at = datetime.now().astimezone().isoformat()
        start_ns = time.monotonic_ns()

        response = await self.client.responses.create(**self._last_request, max_output_tokens=16)
        self._record_usage(response=response, started_at=started_at, start_ns=start_ns)

    async def prompt(
        self, instructions: str, history: History, tools: list[MCPTool] | None = None
    ) -> History:
        """Prompt the model using the given conversation history and available MCP tools.

        :param instructions: System-level instructions to the model
        :param history: Conversation history upon which the model response is conditioned
        :param tools: Optional list of MCP tools available to the model (defaults to None)
        :return: Model response, represented as a list of additional conversation messages
        """
        started_at = datetime.now().astimezone().isoformat()
        start_ns = time.monotonic_ns()

        request: dict[str, Any] = {
            "model": self.config.model_id,
            "input": list(history),
            "tools": [mcp_tool_to_openai_tool(tool) for tool in tools] if tools else omit,
            "instructions": instructions,
            "prompt_cache_options": {"mode": "explicit"} if self.config.caching else omit,
            "extra_body": self._extra_body if self.config.caching else None,
        }
        self._last_request = request

        response: Response = await self.client.responses.create(
            **request, max_output_tokens=self.config.max_output_tokens
        )

        self._record_usage(response=response, started_at=started_at, start_ns=start_ns)

        return [_openai_output_to_input_item(item) for item in response.output]
