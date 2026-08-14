from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../Backend, absolute

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None

    SPEAKER_VERIFICATION_MODEL_ROOT: Path = BACKEND_DIR / "pretrained_models" / "speaker_verification"
    SPEAKER_VERIFICATION_DATASET_ROOT: Path = BACKEND_DIR / "data" / "speaker_verification"

    @property
    def speaker_verification_ecapa_dir(self) -> Path:
        return self.SPEAKER_VERIFICATION_MODEL_ROOT / "ecapa-tdnn"

    @property
    def speaker_verification_hf_cache_dir(self) -> Path:
        return self.SPEAKER_VERIFICATION_MODEL_ROOT / "huggingface"

    @property
    def speaker_verification_demo_dataset_dir(self) -> Path:
        return self.SPEAKER_VERIFICATION_DATASET_ROOT / "vox_indian_demo_92"

settings = Settings()
