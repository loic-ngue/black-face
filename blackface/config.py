"""Configuration lue depuis l'environnement (et le fichier .env local)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=False)

# Modèles Hugging Face figés sur une révision précise (pas de code distant exécuté).
IMAGE_MODEL_ID = "haywoodsloan/ai-image-detector-deploy"
IMAGE_MODEL_REVISION = "4ae1822603cb1ec980c217cfde1ef1b8e0477a16"
TEXT_MODEL_ID = "fakespot-ai/roberta-base-ai-text-detection-v1"
TEXT_MODEL_REVISION = "f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db"
# Inférence textuelle multilingue : une source confirme / contredit / ignore l'affirmation.
NLI_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MODEL_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str | None
    google_factcheck_api_key: str | None


def get_settings() -> Settings:
    return Settings(
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        google_factcheck_api_key=os.getenv("GOOGLE_FACTCHECK_API_KEY") or None,
    )
