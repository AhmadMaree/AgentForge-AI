
from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = "gemini"
    model: str = "gemini-2.5-flash"

    temperature: float = Field(
        default= 0.3,
        ge=0.0,
        le=2.0
    )

    max_output_token: int = Field(
        default=2048,
        gt=0
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        le=10
    )

    timeout_seconds: float = Field(
        default=60.0,
        gt=0
    )