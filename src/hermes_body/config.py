"""Configuration management for hermes-body.

Loads environment variables from .env in the project root and exposes a
typed Config object plus a `validate()` method.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file in project root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Realtime (the brain / voice loop)
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_REALTIME_MODEL: str = field(
        default_factory=lambda: os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    )
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))

    # Hermes (the knowledge tool)
    HERMES_BASE_URL: str = field(
        default_factory=lambda: os.getenv("HERMES_BASE_URL", "http://localhost:8642/v1")
    )
    HERMES_API_KEY: str = field(default_factory=lambda: os.getenv("HERMES_API_KEY", ""))
    HERMES_MODEL: str = field(default_factory=lambda: os.getenv("HERMES_MODEL", "hermes-agent"))

    # Robot
    ROBOT_NAME: Optional[str] = field(default_factory=lambda: os.getenv("ROBOT_NAME") or None)

    # Feature flags
    ENABLE_HERMES: bool = field(
        default_factory=lambda: os.getenv("ENABLE_HERMES", "true").lower() == "true"
    )
    ENABLE_CAMERA: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true"
    )
    ENABLE_FACE_TRACKING: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true"
    )

    # Face tracking — "mediapipe" or "yolo"
    HEAD_TRACKER_TYPE: Optional[str] = field(
        default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "mediapipe")
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of error strings."""
        errors: list[str] = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        if self.ENABLE_HERMES and not self.HERMES_API_KEY:
            errors.append(
                "HERMES_API_KEY is required when ENABLE_HERMES is true "
                "(set in .env, or pass --no-hermes)"
            )
        return errors


# Global configuration instance
config = Config()


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_hermes_enabled(enabled: bool) -> None:
    """Enable or disable the Hermes bridge at runtime."""
    global config
    config.ENABLE_HERMES = enabled
