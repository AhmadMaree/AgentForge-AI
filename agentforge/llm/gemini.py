
from agentforge.llm.base import BaseLLM
from agentforge.core.config import LLMConfig
from agentforge.core.exceptions import ConfigurationError , LLMError
from agentforge.core.message import Message

from typing import Any

from google import genai
from google.genai import types

from collections.abc import Sequence, Iterator



class GeminiLLM(BaseLLM):

    def __init__(self, config: LLMConfig, api_key: str, client: Any | None = None):
        super().__init__(config)

        if config.provider.lower() != "gemini":
            raise ConfigurationError("Gemini required provider called gemini")

        if client is None and not api_key:
            raise ConfigurationError("Gemini api key is required")

        self.client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(config.timeout_seconds * 1000),
                retry_options= types.HttpRetryOptions(
                    attempts= config.max_retries + 1
                )
            )
        )

    def prepare_request(
        self,
        messages: Sequence[Message]
    ) -> tuple[list[types.Content], types.GenerateContentConfig]:
        if not messages:
            raise ValueError("at least one message")

        contents=[]
        system_parts=[]

        for message in messages:

            if message.role == "system":
                system_parts.append(message.content)
                continue

            if message.role == "tool":
                raise ValueError ("Tool still not supported")

            role = (
                "model"
                if message.role == "assistant"
                else "user"
            )

            contents.append(
                types.Content(
                    role=role,
                    parts= [
                        types.Part.from_text(
                            text=message.content
                        )
                    ]
                )
            )

        if not contents:
            raise ValueError("A user or assistant message required")

        generation_config = types.GenerateContentConfig(
            system_instruction=(
                "\n\n".join(system_parts)
                if system_parts
                else None
            ),
            temperature=self.config.temperature,
            max_output_tokens= self.config.max_output_token
        )

        return contents , generation_config


    def invoke(self, messages: Sequence[Message]) -> str:

        contents, generation_config = self.prepare_request(messages)

        try:
            response = self.client.models.generate_content(
               model=self.config.model,
               contents=contents,
               config=generation_config
            )

            if not response.text:
                raise LLMError("Gemini returned no text")
            return response.text
       
        except LLMError:
            raise

        except Exception as exc:
           raise LLMError(
               f"Gemini generation faild: {exc}"
            ) from exc

    def stream(self, messages):
        contents, generate_config = self.prepare_request(messages)

        try:

            response = self.client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=generate_config
            )

            recived_text = False

            for chunk in response:
                if chunk.text:
                    recived_text = True
                    yield chunk.text

            if not recived_text:
                raise LLMError(
                    "Gemini stream returned no text."
                )

        except LLMError:
            raise

        except Exception as exc:
            raise LLMError(
                f"Gemini streaming failed : {exc}"
            ) from exc
           
