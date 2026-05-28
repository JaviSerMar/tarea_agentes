"""Carga de configuración desde entorno (.env soportado vía python-dotenv).

Centralizamos aquí la lectura de variables para que el resto del paquete no
toque ``os.environ`` directamente. Esto facilita cambiar Ollama local por el
endpoint UPV sin tocar el dominio.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "si", "sí"}


@dataclass(frozen=True)
class Settings:
    ollama_url: str
    llm_model: str
    embed_model: str
    verify_ssl: bool
    chroma_path: Path
    collection_name: str
    api_host: str
    api_port: int
    corpus_dir: Path
    llm_provider: str
    poligpt_base_url: str
    poligpt_api_key: str | None
    poligpt_model: str
    embedder_provider: str
    sentence_transformers_model: str
    vector_store_provider: str
    faiss_path: Path


    @classmethod
    def from_env(cls) -> Settings:
        repo_root = Path(__file__).resolve().parents[2]
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api"),
            llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            verify_ssl=_bool("VERIFY_SSL", True),
            chroma_path=Path(os.getenv("CHROMA_PATH", str(repo_root / "data" / "chroma"))),
            collection_name=os.getenv("COLLECTION_NAME", "dni"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            corpus_dir=Path(os.getenv("CORPUS_DIR", str(repo_root / "corpus"))),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
            poligpt_base_url=os.getenv(
                "POLIGPT_BASE_URL",
                "https://api.poligpt.upv.es/v1",
            ),
            poligpt_api_key=os.getenv("POLIGPT_API_KEY"),
            poligpt_model=os.getenv("POLIGPT_MODEL", "poligpt"),
            embedder_provider=os.getenv(
                "EMBEDDER_PROVIDER",
                "ollama",
            ).strip().lower(),
            sentence_transformers_model=os.getenv(
                "SENTENCE_TRANSFORMERS_MODEL",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ),
            vector_store_provider=os.getenv(
                "VECTOR_STORE_PROVIDER",
                "chroma",
            ).strip().lower(),
            faiss_path=Path(
                os.getenv("FAISS_PATH", str(repo_root / "data" / "dni.index"))
            ),
        )


SETTINGS = Settings.from_env()
