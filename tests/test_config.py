import pytest
from pydantic import ValidationError

from agentforge.core.config import LLMConfig


def test_default_config():
    config = LLMConfig()

    assert config.provider == "gemini"
    assert config.temperature == 0.3
    assert config.max_retries == 2


def test_invalid_temperature():
    with pytest.raises(ValidationError):
        LLMConfig(temperature=-1)


def test_invalid_timeout():
    with pytest.raises(ValidationError):
        LLMConfig(timeout_seconds=0)


def test_config_is_immutable():
    config = LLMConfig()

    with pytest.raises(ValidationError):
        config.temperature = 0.8