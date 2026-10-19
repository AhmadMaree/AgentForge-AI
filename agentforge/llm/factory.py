
import os
from dotenv import load_dotenv

from agentforge.core.config import LLMConfig
from agentforge.core.exceptions import ConfigurationError
from agentforge.llm.gemini import GeminiLLM
from agentforge.llm.base import BaseLLM


class FactoryLLM :

    @staticmethod
    def create(
        provider: str | None = None,
        config: LLMConfig | None = None
    ) -> BaseLLM:
        load_dotenv()

        selected_provider = (
            provider
            or (config.provider if config.provider else None)
            or os.getenv("LLM_PROVIDER", "gemini")
        ).lower()

        if selected_provider != "gemini":
            raise ConfigurationError(
                f"This {selected_provider} not supported yet"
            )

        if config is None:
            config = LLMConfig(
                provider=selected_provider,
                model= os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
            )
        elif config.provider.lower() != selected_provider:
            raise ConfigurationError("The provider not match config provider")

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise("Gemini apu key is missing")

        return GeminiLLM(
            config=config,
            api_key=api_key
        )
