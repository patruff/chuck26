"""Custom LLM adapter wrapping any OpenAI-compatible chat completion API.

Bridges the smolagents agent framework to any OpenAI-compatible backend
(Cerebras, Together, local vLLM, etc.) by subclassing smolagents.Model
and delegating to the openai Python client with a configurable base_url.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import OpenAI
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole

if TYPE_CHECKING:
    from smolagents import Tool

    from chuckles_prime.config import AppConfig


class OpenAICompatibleModel(Model):
    """smolagents Model backed by any OpenAI-compatible chat completion API.

    Args:
        model_name: Model identifier (e.g. "qwen-3-32b", "llama-3.3-70b").
        api_base_url: Base URL of the OpenAI-compatible API endpoint.
        api_key_env_var: Name of the environment variable holding the API key.
        **kwargs: Forwarded to the base Model class (e.g. max_tokens, temperature).
    """

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key_env_var: str,
        **kwargs,
    ) -> None:
        super().__init__(model_id=model_name, **kwargs)

        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env_var} is not set. "
                f"Export it before using the model."
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        self.api_key_env_var = api_key_env_var

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Send messages to the OpenAI-compatible API and return a ChatMessage.

        Uses the base class _prepare_completion_kwargs to handle message
        conversion (including tool role mapping) and parameter merging.

        Args:
            messages: List of ChatMessage or dict messages.
            stop_sequences: Strings that stop generation if encountered.
            response_format: Response format hint (accepted but not used).
            tools_to_call_from: Tool list (accepted but ignored; CodeAgent
                uses Python code in text, not the OpenAI tools API).
            **kwargs: Additional overrides for the completion call.

        Returns:
            ChatMessage with role ASSISTANT and the model's response content.
        """
        # Leverage the base class helper for message cleaning, role conversion,
        # stop sequence handling, and kwargs merging.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            # Do NOT pass tools_to_call_from -- CodeAgent generates tool calls
            # as Python code in the text response, not via the OpenAI tools API.
            tools_to_call_from=None,
            **kwargs,
        )

        # Add the model identifier.
        completion_kwargs["model"] = self.model_id

        # Let openai.APIError and subclasses propagate up -- the caller
        # (smolagents) handles retries.
        response = self.client.chat.completions.create(**completion_kwargs)

        content = response.choices[0].message.content

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=None,
            raw=response,
        )


def create_model(config: AppConfig) -> OpenAICompatibleModel:
    """Create a model from an AppConfig instance.

    Args:
        config: Application configuration with model_name, api_base_url,
            and api_key_env_var fields.

    Returns:
        Configured OpenAICompatibleModel ready for inference.
    """
    return OpenAICompatibleModel(
        model_name=config.model_name,
        api_base_url=config.api_base_url,
        api_key_env_var=config.api_key_env_var,
        max_tokens=4096,
        temperature=0.7,
    )


def check_model_connectivity(model: OpenAICompatibleModel) -> str:
    """Send a simple prompt to verify the model connection works.

    Args:
        model: A configured OpenAICompatibleModel instance.

    Returns:
        The model's text response to a trivial prompt.
    """
    response = model([{"role": "user", "content": "Say 'hello' and nothing else."}])
    return response.content
