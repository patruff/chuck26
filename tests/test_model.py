"""Tests for the custom LLM model adapter and factory."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from smolagents import Model
from smolagents.models import ChatMessage, MessageRole

from chuckles_prime.model import (
    OpenAICompatibleModel,
    check_model_connectivity,
    create_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_VAR = "TEST_CHUCKLES_API_KEY"
FAKE_KEY = "fake-key-123"


def _make_mock_response(content: str = "Hello!") -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_model_subclasses_smolagents_model():
    """OpenAICompatibleModel is a proper smolagents Model subclass."""
    assert issubclass(OpenAICompatibleModel, Model)


def test_model_init_missing_env_var(monkeypatch):
    """Construction raises ValueError when the API key env var is not set."""
    monkeypatch.delenv(ENV_VAR, raising=False)

    with pytest.raises(ValueError, match=ENV_VAR):
        OpenAICompatibleModel(
            model_name="test-model",
            api_base_url="http://localhost:8000/v1",
            api_key_env_var=ENV_VAR,
        )


def test_model_init_success(monkeypatch):
    """Construction succeeds when the API key env var is set."""
    monkeypatch.setenv(ENV_VAR, FAKE_KEY)

    model = OpenAICompatibleModel(
        model_name="test-model",
        api_base_url="http://localhost:8000/v1",
        api_key_env_var=ENV_VAR,
    )

    assert model.model_id == "test-model"
    assert model.api_key_env_var == ENV_VAR
    # client should be an OpenAI instance
    from openai import OpenAI

    assert isinstance(model.client, OpenAI)


@patch("chuckles_prime.model.OpenAI")
def test_generate_returns_chat_message(mock_openai_cls, monkeypatch):
    """generate() returns a ChatMessage with ASSISTANT role and correct content."""
    monkeypatch.setenv(ENV_VAR, FAKE_KEY)

    # Set up mock client
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response("Hello!")

    model = OpenAICompatibleModel(
        model_name="test-model",
        api_base_url="http://localhost:8000/v1",
        api_key_env_var=ENV_VAR,
    )

    result = model.generate([{"role": "user", "content": "Hi"}])

    assert isinstance(result, ChatMessage)
    assert result.role == MessageRole.ASSISTANT
    assert result.content == "Hello!"
    assert result.tool_calls is None

    # Verify the client was called
    mock_client.chat.completions.create.assert_called_once()


@patch("chuckles_prime.model.OpenAI")
def test_generate_converts_chat_messages(mock_openai_cls, monkeypatch):
    """generate() properly converts ChatMessage objects for the API call."""
    monkeypatch.setenv(ENV_VAR, FAKE_KEY)

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response("OK")

    model = OpenAICompatibleModel(
        model_name="test-model",
        api_base_url="http://localhost:8000/v1",
        api_key_env_var=ENV_VAR,
    )

    # Pass ChatMessage objects instead of dicts
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Test"),
    ]
    result = model.generate(messages)

    assert result.content == "OK"

    # Verify the API was called with properly formatted messages
    call_kwargs = mock_client.chat.completions.create.call_args
    api_messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
    assert len(api_messages) == 2

    # Roles should be MessageRole enum values (which serialize as strings)
    assert api_messages[0]["content"] == "You are helpful."
    assert api_messages[1]["content"] == "Test"


@patch("chuckles_prime.model.OpenAI")
def test_create_model_from_config(mock_openai_cls, monkeypatch):
    """create_model() builds a model from AppConfig with correct defaults."""
    monkeypatch.setenv(ENV_VAR, FAKE_KEY)

    mock_openai_cls.return_value = MagicMock()

    # Use a simple namespace to mimic AppConfig
    config = SimpleNamespace(
        model_name="qwen-3-32b",
        api_base_url="https://api.cerebras.ai/v1",
        api_key_env_var=ENV_VAR,
    )

    model = create_model(config)

    assert isinstance(model, OpenAICompatibleModel)
    assert model.model_id == "qwen-3-32b"
    # Check default kwargs were applied
    assert model.kwargs.get("max_tokens") == 4096
    assert model.kwargs.get("temperature") == 0.7


# ---------------------------------------------------------------------------
# Integration test (optional, requires live API key)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("CEREBRAS_API_KEY"),
    reason="No CEREBRAS_API_KEY set -- skipping live integration test",
)
def test_live_connectivity():
    """Verify the model can connect to a real API and get a response."""
    model = OpenAICompatibleModel(
        model_name="qwen-3-32b",
        api_base_url="https://api.cerebras.ai/v1",
        api_key_env_var="CEREBRAS_API_KEY",
    )

    result = check_model_connectivity(model)
    assert isinstance(result, str)
    assert len(result) > 0
