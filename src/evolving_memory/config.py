"""Configuration for the Cognitive Trajectory Engine."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ISAConfig(BaseModel):
    """Configuration for the Agentic ISA / VM."""
    max_instructions: int = 500
    enable_fallback: bool = True


class DreamConfig(BaseModel):
    """Configuration for the dream engine."""
    merge_similarity_threshold: float = 0.85
    max_traces_per_cycle: int = 50
    min_actions_for_trace: int = 2


class RouterConfig(BaseModel):
    """Configuration for the cognitive router."""
    similarity_weight: float = 0.5
    confidence_weight: float = 0.3
    success_rate_weight: float = 0.2
    composite_threshold: float = 0.4
    top_k: int = 5
    anomaly_threshold: float = 0.3


class CTEConfig(BaseModel):
    """Top-level configuration for the Cognitive Trajectory Engine."""
    db_path: Path = Field(default=Path("memory.db"))
    faiss_path: Path = Field(default=Path("memory.faiss"))
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    dream: DreamConfig = Field(default_factory=DreamConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    isa: ISAConfig = Field(default_factory=ISAConfig)
