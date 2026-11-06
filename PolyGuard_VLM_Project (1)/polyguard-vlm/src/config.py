from dataclasses import dataclass, field
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    model: str = ""
    rpm: float = 6
    retries: int = 3
    timeout: float = 90
    max_side: int = 2400
    max_file_mb: int = 30
    max_pixels: int = 40_000_000

    def __post_init__(self):
        if self.rpm <= 0 or self.timeout <= 0 or self.retries < 0:
            raise ValueError("Rate/timeout must be positive and retries nonnegative")
        if min(self.max_side, self.max_file_mb, self.max_pixels) <= 0:
            raise ValueError("Image limits must be positive")

    @classmethod
    def from_env(cls):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            if (ROOT / ".env").exists():
                raise RuntimeError("Install requirements.txt to load .env")
        return cls(api_key=os.getenv("GEMINI_API_KEY", ""),
                   model=os.getenv("GEMINI_MODEL", ""),
                   rpm=float(os.getenv("REQUESTS_PER_MINUTE", "6")),
                   retries=int(os.getenv("MAX_RETRIES", "3")),
                   timeout=float(os.getenv("TIMEOUT_SECONDS", "90")),
                   max_side=int(os.getenv("MAX_IMAGE_SIDE", "2400")),
                   max_file_mb=int(os.getenv("MAX_FILE_MB", "30")),
                   max_pixels=int(os.getenv("MAX_PIXELS", "40000000")))

    def require_api(self):
        if not self.api_key or self.api_key == "your_key_here":
            raise ValueError("Set GEMINI_API_KEY in .env; see README. No API calls made.")
        if not self.model:
            raise ValueError("Set GEMINI_MODEL in .env; run models to check availability.")
