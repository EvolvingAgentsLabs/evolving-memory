"""Phase 4 — Memory Compaction: LLM-powered summarization of verbose nodes.

Identifies parent nodes with low access counts and long summaries, then uses
the LLM to produce tighter summaries. Follows the TraceCurator/HierarchicalChunker
pattern: constructor takes (llm, store, config, adapter), async method returns count.
"""

from __future__ import annotations

import logging

from ..config import DreamConfig
from ..llm.base import BaseLLMProvider
from ..models.graph import ParentNode
from ..models.strategy import DreamJournalEntry
from ..storage.sqlite_store import SQLiteStore
from .domain_adapter import DreamDomainAdapter
from .prompt_builder import DreamPromptBuilder

logger = logging.getLogger(__name__)

COMPACTION_SYSTEM = (
    "You are a memory compaction engine. Your job is to rewrite a verbose "
    "memory summary into a concise version that preserves all key facts, "
    "decisions, and outcomes. Output ONLY the compacted summary, nothing else."
)

COMPACTION_PROMPT = (
    "Compact the following memory node summary into at most {max_len} characters.\n"
    "Preserve: goal, outcome, key actions, critical decisions, negative constraints.\n"
    "Drop: filler, redundant detail, verbose descriptions.\n\n"
    "Goal: {goal}\n"
    "Outcome: {outcome}\n"
    "Current summary ({current_len} chars):\n{summary}"
)


class MemoryCompactor:
    """Phase 4 — compacts verbose parent node summaries via LLM."""

    def __init__(
        self,
        llm: BaseLLMProvider,
        store: SQLiteStore,
        config: DreamConfig,
        adapter: DreamDomainAdapter | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._config = config
        self._adapter = adapter

    async def compact(
        self,
        journal: DreamJournalEntry,
        domain: str | None = None,
    ) -> int:
        """Compact verbose, low-access parent nodes. Returns count of nodes compacted."""
        nodes = self._store.get_all_parent_nodes(domain)
        candidates = [
            n for n in nodes
            if (
                n.access_count < self._config.compaction_min_access
                and len(n.summary) > self._config.compaction_max_summary_len
                and n.version < 5
            )
        ]

        if not candidates:
            journal.phase_log.append("Compaction: no candidates found")
            return 0

        journal.phase_log.append(
            f"Compaction: {len(candidates)} candidates from {len(nodes)} nodes"
        )

        compacted = 0
        system_prompt = self._build_system_prompt()

        for node in candidates:
            try:
                new_summary = await self._compact_node(node, system_prompt)
                if new_summary and len(new_summary) < len(node.summary):
                    node.summary = new_summary
                    node.version += 1
                    self._store.save_parent_node(node)
                    compacted += 1
                    logger.debug(
                        "Compacted node %s: %d -> %d chars",
                        node.node_id, len(node.summary), len(new_summary),
                    )
            except Exception:
                logger.debug("Compaction failed for node %s", node.node_id, exc_info=True)

        journal.phase_log.append(f"Compaction: {compacted} nodes compacted")
        return compacted

    def _build_system_prompt(self) -> str:
        if self._adapter is None:
            return COMPACTION_SYSTEM
        return (
            DreamPromptBuilder()
            .append_raw(COMPACTION_SYSTEM)
            .with_domain_context(self._adapter, "consolidation")
            .build()
        )

    async def _compact_node(self, node: ParentNode, system_prompt: str) -> str:
        prompt = COMPACTION_PROMPT.format(
            max_len=self._config.compaction_max_summary_len,
            goal=node.goal,
            outcome=node.outcome.value,
            current_len=len(node.summary),
            summary=node.summary,
        )
        return await self._llm.complete(prompt, system=system_prompt)
