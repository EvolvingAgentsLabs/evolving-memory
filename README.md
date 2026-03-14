# Evolving Memory

**Cognitive Trajectory Engine (CTE) — Topological memory graphs for LLM agents**

A bio-inspired memory system that captures agent execution traces, consolidates them through dream cycles (SWS/REM/Consolidation), and enables intelligent memory retrieval via topological graph traversal. Built on an **Agentic ISA** (Instruction Set Architecture) where LLMs emit structured opcodes instead of JSON.

[![Tests](https://img.shields.io/badge/tests-104%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)]()

---

## Why This Exists

Current LLM memory systems (LangChain, AutoGen, RAG) share a fundamental flaw: they treat memory as **document retrieval** — semantic search returns disconnected text fragments with no causal ordering, no procedural structure, and no learning from failure.

The human brain doesn't work this way. The hippocampus uses the **same neural machinery** for spatial navigation and conceptual reasoning. Grid cells that help you walk through a house are the same cells that help you solve a math problem step by step. Memory isn't a search engine — it's a **topological graph of causal trajectories**.

Evolving Memory implements this insight:

- **Semantic similarity is just the index** (the pointer into the library catalog)
- **Real knowledge lives in navigable causal sequences** (the actual books, read in order)
- **Learning happens through consolidation** (dreaming compresses experience into reusable procedures)

### The ISA Advantage

Traditional LLM integration uses JSON for all communication — the LLM receives prompts and responds with JSON objects. This is **token-bloated (~25-30 tokens per tool call)**, latency-heavy, and non-deterministic (hallucinated keys, missing brackets, conversational filler).

Evolving Memory replaces JSON with an **Agentic Assembly Language** — a Cognitive ISA where the LLM emits structured opcodes that a Python VM executes:

```
# JSON approach: ~80 tokens, fragile parsing
{"negative_constraints": [{"description": "Do not retry without backoff", "reasoning": "..."}]}

# ISA approach: ~15 tokens, deterministic parsing
EXTRACT_CONSTRAINT trace_001 "Do not retry without backoff"
HALT
```

| Metric | JSON | ISA | Improvement |
|--------|------|-----|-------------|
| Tokens per dream trace | ~590 | ~155 | **74% reduction** |
| LLM calls per trace | 8 | 3 | **62% reduction** |
| Parse failures | Common | Near-zero | Deterministic |

---

## Architecture

```
                    +-----------------------+
                    |   Waking State        |
                    |   (Trace Capture)     |
                    +-----------+-----------+
                                |
                         Raw Trace Log
                                |
                    +-----------v-----------+
                    |    Dream Engine       |
                    |  SWS -> REM -> Cons.  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Topological Memory   |
                    |  Graph + FAISS Index  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Cognitive Router     |
                    |  (Tripartite Path)    |
                    +-----------------------+
```

### The Three Subsystems

#### 1. Waking State (Capture Layer)

During active work, the agent records execution traces — structured logs of reasoning, actions, and results organized hierarchically (L1 Goals -> L2 Architecture -> L3 Tactics -> L4 Reactive):

```python
with cte.session("build authentication system") as logger:
    with logger.trace(HierarchyLevel.TACTICAL, "implement JWT") as ctx:
        ctx.action("research", "Read RFC 7519", result="Understood claims, signing")
        ctx.action("code", "Write jwt_utils.py", result="200 lines, HS256 + RS256")
        ctx.action("test", "Run pytest", result="8/8 passing")
```

#### 2. Dream Engine (Consolidation)

When the agent "sleeps" (or context saturates), the Dream Engine processes raw traces through three bio-inspired phases:

**Phase 1 — SWS (Slow-Wave Sleep): Curation**
- Analyzes failure traces to extract **negative constraints** (what NOT to do)
- Identifies the **critical path** — the minimal essential action sequence
- Prunes noise: retries, dead ends, and redundant steps are forgotten
- LLM emits `EXTRACT_CONSTRAINT` and `MARK_CRITICAL` / `MARK_NOISE` opcodes

**Phase 2 — REM: Hierarchical Chunking**
- Creates a **Parent Node** (high-level strategy summary with semantic embedding)
- Creates **Child Nodes** (individual steps with reasoning/action/result)
- Single LLM call emits `BUILD_PARENT` + `BUILD_CHILD` opcodes in one program
- `$LAST_PARENT` symbolic reference lets the LLM link children without tracking UUIDs

**Phase 3 — Consolidation: Topological Wiring**
- Generates embeddings for parent nodes and adds to FAISS index
- Detects **merges**: if a similar strategy already exists (cosine similarity > 0.85), the new experience is consolidated into the existing node — confidence increases, constraints accumulate
- Creates **edges**: `IS_CHILD_OF` (hierarchical), `NEXT_STEP` / `PREVIOUS_STEP` (temporal/causal)
- Repeated experiences compress into single consolidated nodes — **procedural memory emerges from episodic traces**

#### 3. Cognitive Router (Tripartite Decision)

When the agent encounters a new task, the router makes a three-way decision:

| Path | ISA Analog | When | What Happens |
|------|-----------|------|--------------|
| **ZERO_SHOT** | `EXEC_NATIVE` | No relevant memory found | LLM reasons from scratch using pre-trained weights |
| **MEMORY_TRAVERSAL** | `NAV_GRAPH` | Memory hit above threshold | Agent walks the graph step-by-step: read node, execute, follow `NEXT_STEP` edge |
| **CONTEXT_JUMP** | `JMP_PTR` | Semantic drift detected during traversal | Abandon current branch, search for new entry point |

The router scores candidates using a composite metric:

```
composite = similarity_weight * FAISS_score
          + confidence_weight * node.confidence
          + success_rate_weight * node.success_rate
```

---

## The Cognitive ISA

16 opcodes organized in three groups:

### Memory Traversal (0x10-0x1F)

```
MEM_PTR    <query>           Search FAISS, return closest parent node_id
MEM_READ   <node_id>         Return node summary text
MEM_NEXT   <node_id>         Return next sibling child node_id
MEM_PREV   <node_id>         Return previous sibling child node_id
MEM_PARENT <node_id>         Return parent node_id
MEM_JMP    <node_id>         Context jump — load node into accumulator
```

### Dream / Consolidation (0x20-0x2F)

```
EXTRACT_CONSTRAINT <trace_id> "<description>"     Extract a negative constraint
MARK_CRITICAL      <trace_id> <action_index>       Mark action as essential
MARK_NOISE         <trace_id> <action_index>       Mark action as noise (forget)
BUILD_PARENT       "<goal>" "<summary>" <confidence>  Create parent node
BUILD_CHILD        <parent_id> <step_idx> "<reasoning>" "<action>" "<result>"
LNK_NODE           <source_id> <target_id> <edge_type>  Create edge
GRP_NODE           <node_a> <node_b>               Mark nodes for merge
PRN_NODE           <node_id>                        Mark node for pruning
```

### System (0xF0-0xFF)

```
NOP                No operation
YIELD  <message>   Append message to output buffer
HALT               Stop VM execution
```

### Example: Full Dream Program

```asm
# Phase 1: SWS — Extract constraints and critical path
EXTRACT_CONSTRAINT trace_001 "Do not retry API calls without exponential backoff"
EXTRACT_CONSTRAINT trace_001 "Do not ignore rate limit headers"
MARK_CRITICAL trace_001 0
MARK_CRITICAL trace_001 2
MARK_CRITICAL trace_001 4
MARK_NOISE trace_001 1
MARK_NOISE trace_001 3
HALT
```

```asm
# Phase 2: REM — Build hierarchical memory
BUILD_PARENT "implement JWT auth" "Strategy for JWT authentication with HS256 signing" 0.85
BUILD_CHILD $LAST_PARENT 0 "Research JWT spec" "Read RFC 7519" "Understood claims and signing"
BUILD_CHILD $LAST_PARENT 1 "Implement encoder" "Write jwt_utils.py" "200 lines, HS256 + RS256"
BUILD_CHILD $LAST_PARENT 2 "Write tests" "Create test_jwt.py" "8/8 tests passing"
HALT
```

### Parser Fallback Strategy (RoClaw Pattern)

The parser uses three-mode fallback for maximum resilience:

1. **Primary**: `shlex.split()` — handles quoted strings correctly
2. **Fallback**: regex `r'("[^"]*"|\S+)'` — more lenient with malformed quotes
3. **Final**: `str.split()` — always succeeds, loses multi-word args

Unknown opcodes produce parse warnings but don't stop execution. Valid instructions still run.

### VM Safety

- **Max instructions limit** (default 500) prevents runaway programs
- **HALT stops execution** — anything after HALT is never reached
- **Accumulate, don't commit**: VM builds results in memory; the dream engine persists them only after successful completion. A VM error doesn't leave inconsistent database state.
- **Side effects audit log**: every handler logs what it did for debugging/replay

---

## Quick Start

### Installation

```bash
pip install evolving-memory

# With LLM providers
pip install evolving-memory[anthropic]   # or [openai] or [all]
```

### Usage

```python
import asyncio
from evolving_memory import CognitiveTrajectoryEngine, HierarchyLevel, RouterPath
from evolving_memory.llm.anthropic_provider import AnthropicProvider

async def main():
    llm = AnthropicProvider()
    cte = CognitiveTrajectoryEngine(llm=llm, db_path="memory.db")

    # 1. Capture traces during work
    with cte.session("build auth system") as logger:
        with logger.trace(HierarchyLevel.TACTICAL, "implement JWT") as ctx:
            ctx.action("research", "Read RFC 7519", result="Understood claims")
            ctx.action("code", "Write jwt_utils.py", result="200 lines")
            ctx.action("test", "Run pytest", result="8/8 passing")

    # 2. Dream — consolidate traces into memory graph
    journal = await cte.dream()
    print(f"Nodes created: {journal.nodes_created}")
    print(f"Constraints: {journal.constraints_extracted}")

    # 3. Query memory
    decision = cte.query("how to implement JWT authentication?")

    if decision.path == RouterPath.MEMORY_TRAVERSAL:
        # Walk the graph step by step
        state = cte.begin_traversal(decision.entry_point)
        while True:
            child, state = cte.next_step(state)
            if child is None:
                break
            print(f"Step {child.step_index}: {child.action} -> {child.result}")

    cte.close()

asyncio.run(main())
```

### Direct ISA/VM Usage

```python
from evolving_memory import InstructionParser, CognitiveVM

parser = InstructionParser()
program = parser.parse("""
BUILD_PARENT "deploy database" "Strategy for Docker Postgres deployment" 0.9
BUILD_CHILD $LAST_PARENT 0 "Pull image" "docker pull postgres:15" "Image downloaded"
BUILD_CHILD $LAST_PARENT 1 "Create volume" "docker volume create pgdata" "Volume ready"
BUILD_CHILD $LAST_PARENT 2 "Run container" "docker run -d postgres:15" "Container started"
HALT
""")

vm = CognitiveVM()
result = vm.execute(program)

print(f"Parents: {len(result.built_parents)}")
print(f"Children: {len(result.built_children)}")
print(f"Instructions executed: {result.instructions_executed}")
```

---

## Project Structure

```
src/evolving_memory/
    __init__.py              # Public facade: CognitiveTrajectoryEngine
    config.py                # CTEConfig, DreamConfig, RouterConfig, ISAConfig

    isa/                     # Agentic Instruction Set Architecture
        opcodes.py           # 16 opcodes (IntEnum), Instruction, Program
        parser.py            # Text-assembly parser, 3-mode fallback
        serializer.py        # Instruction -> text (debug/logging)

    vm/                      # Cognitive Virtual Machine
        context.py           # VMContext (execution state), VMResult
        handlers.py          # 16 handler functions, $LAST_PARENT resolution
        machine.py           # CognitiveVM dispatch loop

    llm/                     # LLM provider abstraction
        base.py              # BaseLLMProvider (complete, complete_json, emit_program)
        anthropic_provider.py
        openai_provider.py
        prompts.py           # ISA instruction templates

    dream/                   # 3-phase memory consolidation
        curator.py           # Phase 1 SWS: failure analysis, critical path
        chunker.py           # Phase 2 REM: hierarchical node creation
        connector.py         # Phase 3: edges, embeddings, merge detection
        engine.py            # DreamEngine orchestrator

    capture/                 # Trace capture
        session.py           # SessionManager
        trace_logger.py      # TraceLogger, TraceContext

    models/                  # Pydantic data models
        hierarchy.py         # HierarchyLevel, TraceOutcome, EdgeType, RouterPath
        trace.py             # TraceEntry, ActionEntry, TraceSession
        graph.py             # ParentNode, ChildNode, ThoughtEdge
        strategy.py          # NegativeConstraint, DreamJournalEntry
        query.py             # EntryPoint, RouterDecision, TraversalState

    router/                  # Query routing
        cognitive_router.py  # Tripartite router (Zero-Shot / Traversal / Jump)
        anomaly.py           # Semantic drift detection

    embeddings/
        encoder.py           # sentence-transformers wrapper

    storage/
        sqlite_store.py      # SQLite graph store (8 tables)
        vector_index.py      # FAISS vector index
```

---

## The Neuroscience Connection

This architecture directly mirrors how biological memory works:

| Biology | Evolving Memory | Purpose |
|---------|----------------|---------|
| **Waking experience** | Trace Capture | Record what happened |
| **Slow-Wave Sleep** | SWS Curator | Replay and evaluate experiences |
| **REM Sleep** | REM Chunker | Compress into abstract patterns |
| **Synaptic consolidation** | Topological Connector | Wire patterns into long-term graph |
| **Hippocampal grid cells** | FAISS Index + Graph Edges | Navigate both spatial and conceptual space |
| **Procedural memory** | Merged high-confidence nodes | Skills automated through repetition |
| **Forgetting** | MARK_NOISE + pruning | Remove noise, keep signal |
| **Negative constraints** | EXTRACT_CONSTRAINT | Learn from failure — "don't touch hot stove" |

The key insight: **navigating physical space (robotics) and navigating conceptual space (reasoning/code) are mathematically the same problem.** Both are causal trajectory traversals through a topological graph. This is why the same architecture powers both [RoClaw](https://github.com/EvolvingAgentsLabs) (robotics) and evolving-memory (LLM agents).

---

## Why Not RAG?

| | RAG | Evolving Memory |
|---|---|---|
| **What's stored** | Text chunks | Causal trajectories |
| **Retrieval** | Semantic similarity | Similarity + topological traversal |
| **Structure** | Flat document list | Hierarchical graph with edges |
| **Learning** | Static (re-index) | Continuous (dream consolidation) |
| **Failure handling** | None | Negative constraints extracted |
| **Ordering** | Lost | Preserved (NEXT_STEP edges) |
| **Compression** | Token-level | Experience-level (merge similar strategies) |
| **Context window** | Floods with fragments | Streams one step at a time |

---

## Testing

```bash
# Run the full test suite (104 tests)
pytest tests/ -v

# Individual modules
pytest tests/test_isa.py       # 29 tests — parser round-trips, all 16 opcodes
pytest tests/test_vm.py        # 25 tests — handlers, programs, safety limits
pytest tests/test_dream_engine.py  # 9 tests — dream cycle with ISA
pytest tests/test_integration.py   # 2 tests — full capture -> dream -> query
```

All tests use a `MockLLMProvider` that emits deterministic ISA opcodes — no API keys needed.

---

## Configuration

```python
from evolving_memory import CTEConfig

config = CTEConfig(
    db_path="memory.db",
    faiss_path="memory.faiss",
    embedding_model="all-MiniLM-L6-v2",
    embedding_dim=384,
    dream=DreamConfig(
        merge_similarity_threshold=0.85,  # When to merge similar nodes
        max_traces_per_cycle=50,
        min_actions_for_trace=2,
    ),
    router=RouterConfig(
        similarity_weight=0.5,
        confidence_weight=0.3,
        success_rate_weight=0.2,
        composite_threshold=0.4,   # Below this -> ZERO_SHOT
        top_k=5,
        anomaly_threshold=0.3,     # Semantic drift detection
    ),
    isa=ISAConfig(
        max_instructions=500,       # VM safety limit
        enable_fallback=True,
    ),
)
```

---

## Custom LLM Providers

Implement `BaseLLMProvider` to use any LLM:

```python
from evolving_memory import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    async def complete(self, prompt: str, system: str = "") -> str:
        ...

    async def complete_json(self, prompt: str, system: str = "") -> dict:
        ...

    async def emit_program(self, prompt: str, system: str = "") -> str:
        """Return raw text for ISA parsing. Use temperature=0.0."""
        ...
```

---

## Part of the EvolvingAgentsLabs Ecosystem

Evolving Memory is one component of a unified theory of **Agentic Control**:

- **RoClaw** — Robotics bytecode ISA (`AA 01 64 64 CB FF`) for hardware motor control
- **llmunix / DreamOS** — Operating system layer for LLM agent orchestration
- **Evolving Memory (CTE)** — Cognitive memory with ISA for software agents

All three share the same principle: **LLM as CPU, structured instructions as the interface, Python/firmware as the VM/executor.** The ISA is text-assembly instead of hex bytecode because Python has no hardware memory constraints — but the architecture is identical.

---

## License

Apache 2.0 -- see [LICENSE](LICENSE)
