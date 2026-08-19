from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "postgresql://app:app@localhost:5433/app"
    log_level: str = "info"
    cache_dir: Path = Path("data/cache")

    # --- ledgerline ---
    edgar_user_agent: str = ""
    edgar_rate_limit_rps: float = 5.0

    # --- sightline ---
    mapillary_token: str = ""
    socrata_app_token: str = ""
    socrata_domain: str = "data.cityofchicago.org"
    socrata_dataset: str = "v6vf-nfxy"

    # --- providers ---
    anthropic_api_key: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    eval_dir: Path = Field(default=Path("evals/runs"))

    def resolved_cache_dir(self) -> Path:
        path = self.cache_dir
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
