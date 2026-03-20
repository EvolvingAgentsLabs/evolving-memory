"""Cognitive Migration Transforms — LLM-powered schema evolution for memory nodes.

When the ISA version changes, legacy memory nodes may need more than a version
re-stamp. Migration transforms allow the dream engine to use the LLM to
re-evaluate and enrich legacy nodes during Phase 0.

Example: ISA 1.0 nodes have no risk assessment. When upgrading to ISA 2.0,
a migration transform can ask the LLM to retroactively evaluate the risk
level of each legacy strategy and inject that into the node's content.

Usage::

    from evolving_memory.dream.migration import MigrationTransform

    class AddRiskLevel(MigrationTransform):
        from_version = "1.0"
        to_version = "2.0"

        async def transform(self, node, children, llm):
            # Ask LLM to assess risk
            assessment = await llm.complete(...)
            node.content = f"{node.content}\\n\\nRisk Assessment: {assessment}"
            return node, children

    engine.register_migration(AddRiskLevel())
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ..llm.base import BaseLLMProvider
from ..models.graph import ChildNode, ParentNode

logger = logging.getLogger(__name__)


class MigrationTransform(ABC):
    """A single cognitive migration that transforms legacy nodes.

    Subclass this to define a migration from one ISA version to another.
    The transform method receives the legacy node, its children, and an LLM
    provider, and returns the enriched node and children.
    """

    from_version: str  # Source ISA version (e.g. "1.0")
    to_version: str    # Target ISA version (e.g. "2.0")

    @abstractmethod
    async def transform(
        self,
        node: ParentNode,
        children: list[ChildNode],
        llm: BaseLLMProvider,
    ) -> tuple[ParentNode, list[ChildNode]]:
        """Transform a legacy node and its children to the new ISA version.

        This is called during Phase 0 of the dream cycle for each legacy node
        that matches `from_version`. The LLM can be used to re-evaluate,
        enrich, or restructure the node's content.

        Args:
            node: The legacy parent node to transform.
            children: The node's child steps.
            llm: LLM provider for re-evaluation.

        Returns:
            Tuple of (transformed_node, transformed_children).
        """
        ...
