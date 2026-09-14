"""Environment configuration; no implicit model calls or time budget."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOES_TECH_KG_", env_file=".env", extra="ignore")
    data_root: Path = Path("data")
    seed: int = Field(default=0, ge=0)
    api_key: SecretStr | None = None
    request_timeout_seconds: int = Field(default=45, gt=0)
