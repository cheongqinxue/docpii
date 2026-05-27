"""Application configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOPII_",
        extra="ignore",
    )

    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible base URL for vLLM serving NuExtract3.",
    )
    vllm_api_key: str = Field(default="EMPTY", description="API key for the vLLM server.")
    vllm_model: str = Field(default="numind/NuExtract3", description="Model name on the server.")

    enable_thinking: bool = Field(
        default=False,
        description="Enable NuExtract reasoning mode (slower, for hard documents).",
    )
    temperature: float = Field(default=0.2, description="Sampling temperature.")
    max_new_tokens: int = Field(default=4096, description="Maximum generated tokens.")

    data_dir: str = Field(default="data", description="Root directory for datasets.")
    output_dir: str = Field(default="outputs", description="Directory for run artifacts.")
