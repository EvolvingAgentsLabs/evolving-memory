"""FastAPI application factory for the evolving-memory server."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import CTEConfig
from ..dream.adapters.default_adapter import DefaultAdapter
from ..dream.adapters.robotics_adapter import RoboticsAdapter
from ..dream.domain_adapter import DreamDomainAdapter
from ..dream.engine import DreamEngine
from ..embeddings.encoder import EmbeddingEncoder
from ..llm.base import BaseLLMProvider
from ..router.cognitive_router import CognitiveRouter
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# Registry of domain adapters
ADAPTER_REGISTRY: dict[str, DreamDomainAdapter] = {
    "default": DefaultAdapter(),
    "software": DefaultAdapter(),
    "robotics": RoboticsAdapter(),
}


class MemoryServer:
    """Holds all shared state for the FastAPI application."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        config: CTEConfig | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.config = config or CTEConfig()
        if db_path is not None:
            self.config.db_path = Path(db_path)

        self.store = SQLiteStore(self.config.db_path)
        self.encoder = EmbeddingEncoder(self.config.embedding_model)
        self.index = VectorIndex(
            dim=self.config.embedding_dim,
            index_path=self.config.faiss_path,
        )
        self.router = CognitiveRouter(
            store=self.store,
            index=self.index,
            encoder=self.encoder,
            config=self.config.router,
        )
        # Build domain-scoped dream engines
        self._engines: dict[str, DreamEngine] = {}
        for name, adapter in ADAPTER_REGISTRY.items():
            self._engines[name] = DreamEngine(
                llm=llm,
                store=self.store,
                index=self.index,
                encoder=self.encoder,
                config=self.config,
                adapter=adapter,
            )

    def get_engine(self, domain: str = "default") -> DreamEngine:
        if domain not in self._engines:
            adapter = ADAPTER_REGISTRY.get(domain, DefaultAdapter())
            self._engines[domain] = DreamEngine(
                llm=self._engines["default"]._llm,
                store=self.store,
                index=self.index,
                encoder=self.encoder,
                config=self.config,
                adapter=adapter,
            )
        return self._engines[domain]

    def close(self) -> None:
        self.store.close()


def create_app(server: MemoryServer) -> "FastAPI":
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI

    from .routes import create_router

    app = FastAPI(
        title="Evolving Memory Server",
        description="REST/WebSocket API for the Cognitive Trajectory Engine",
        version="0.1.0",
    )
    app.state.server = server

    router = create_router(server)
    app.include_router(router)

    return app
