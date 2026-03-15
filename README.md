# Evolving Memory

**Cognitive Trajectory Engine (CTE) — Topological memory graphs for LLM agents**

A bio-inspired memory system that captures agent execution traces, consolidates them through dream cycles (SWS/REM/Consolidation), and enables intelligent memory retrieval via topological graph traversal. Built on an **Agentic ISA** (Instruction Set Architecture) where LLMs emit structured opcodes instead of JSON.

[![Tests](https://img.shields.io/badge/tests-139%20passed-brightgreen)]()
[![Hypothesis](https://img.shields.io/badge/hypothesis-12%2F12%20validated-blue)]()
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)]()

---

## The Hypothesis

### What is Memory?

LLMs already have memory. They generate content from patterns memorized during training — the weights encode compressed statistical representations of language, reasoning, and world knowledge. In this sense, **memory and learning are inseparable**: what a model "knows" is what it has consolidated from exposure to data.

But this is only one kind of memory. Humans have at least three:

1. **Semantic memory** — facts and concepts (what LLM weights encode)
2. **Episodic memory** — specific experiences, ordered in time (what context windows hold temporarily)
3. **Procedural memory** — skills automated through repetition (what doesn't exist in current LLM systems)

Current LLM agents have strong semantic memory (pre-training) and weak, volatile episodic memory (context window). They have **no procedural memory at all**. Every conversation starts from zero. Every session is independent. There is no mechanism by which an agent improves at a task through repeated practice, learns to avoid past mistakes, or builds upon previous work sessions.

This is the gap Evolving Memory fills.

### How Memory Forms

Consider how human memory actually forms. A sequence of events occurs — interactions between a person and the world, between people, between a person and a problem. These interactions have temporal ordering, causal dependencies, and varying degrees of importance. Some steps are critical; others are noise.

This structure is remarkably similar to:

- **Dialogues with an LLM** — a sequence of prompts and responses building toward a goal
- **Chain of Thought reasoning** — a step-by-step reasoning trace where each step depends on the previous
- **Agent execution traces** — the full log of reasoning, actions, and results during a work session

In fact, this is exactly what we implemented in [RoClaw](https://github.com/EvolvingAgentsLabs/RoClaw#navigation-chain-of-thought) — a Chain of Thought for Robot Navigation:

> Each step builds on the previous one:
> 1. **Scene Analysis** — interpret the camera frame, extract location, features, navigation hints
> 2. **Location Matching** — compare current scene against known nodes in the topological map
> 3. **Navigation Planning** — reason about which motor action to take given the map, location, and destination
> 4. **Bytecode Compilation** — compile the VLM's text command into motor frame bytecode

The Semantic Map in RoClaw is the robot's **working memory** — a topological graph where nodes are locations and edges are navigation paths. It accumulates as the robot explores, enabling re-identification of visited places and multi-hop planning.

**The key insight: this same structure — topological graph of causal trajectories — works identically for conceptual navigation.** Navigating physical space (a robot finding its way through a building) and navigating conceptual space (an agent solving a math problem, writing code, or structuring a story) are **mathematically the same operation**: traversal through a directed graph of states connected by causal edges.

### From Experience to Knowledge

Through interaction, agents (biological or artificial) don't just accumulate raw experiences — they generate **knowledge**. This knowledge can take many forms:

- **Patterns** — "every time I do X, Y happens" (statistical regularities)
- **Behaviors / Habits** — "when facing situation S, do action A" (procedural memory)
- **Abstract concepts** — language, mathematics, physics (semantic memory)
- **Negative constraints** — "never do X in situation S" (learned from failure)

At some point, knowledge and memory become parts of the same thing. Humans have biological hardware that is "pretrained" (evolved neural circuits, instincts, sensory processing). LLMs have their pre-trained weights. In both cases, there is a static foundation plus a dynamic layer that accumulates through experience.

For humans, this dynamic layer is built over a lifetime through a biological process: experiences are captured during waking hours, then **consolidated during sleep** through a well-studied cycle of Slow-Wave Sleep (replay and evaluation) and REM sleep (abstraction and integration).

For LLMs and agents, **the missing piece is exactly this**: a smart management system for acquired experience and knowledge. And the most natural way to model and implement it is to mimic the biological mechanism that evolution has already optimized over millions of years.

### The Architecture of Consolidation

The Evolving Memory system implements this biological blueprint:

1. **Trace Capture (Waking)** — During active work, record complete chain-of-thought sequences: every reasoning step, every action, every result, organized hierarchically by goal level

2. **Dream Engine (Sleep)** — Between work sessions, consolidate raw traces through three phases:
   - **SWS (Slow-Wave Sleep)** — Curate: replay traces, identify the critical path (the minimal essential sequence), extract negative constraints from failures, prune noise (retries, dead ends, redundant steps). This is **intelligent forgetting** — the system learns what matters and what doesn't.
   - **REM** — Abstract: compress curated traces into hierarchical memory nodes (a parent strategy with child steps). This creates chunked, context-window-sized units of knowledge that the LLM can consume.
   - **Consolidation** — Connect: generate semantic embeddings, detect merge candidates (is this the same knowledge I already have?), wire causal and temporal edges between nodes, discover cross-domain links.

3. **Cognitive Router (Retrieval)** — When the agent faces a new task:
   - **Semantic similarity is just the index** — embeddings find candidate entry points (pointers into the knowledge graph), like a library catalog pointing to the right shelf
   - **Real knowledge lives in hierarchical traversal** — once a pointer is selected, the agent navigates the graph step by step: read the strategy, follow `NEXT_STEP` edges, execute each action in sequence
   - **Context jumps handle semantic drift** — if the current task diverges from the traversed strategy, the router detects the anomaly and jumps to a new entry point

This is the tripartite decision that an agent makes for every sub-task:

| Decision | Analog | When | What Happens |
|----------|--------|------|--------------|
| **ZERO_SHOT** | No relevant memory | Knowledge gap | LLM reasons from scratch using pre-trained weights |
| **MEMORY_TRAVERSAL** | Memory hit | Similar past experience | Agent walks the graph step-by-step, using consolidated knowledge as context |
| **CONTEXT_JUMP** | Semantic drift | Task evolved mid-traversal | Abandon current path, search for new entry point |

### Why Not RAG?

The fundamental difference: RAG treats memory as **document retrieval**. Evolving Memory treats memory as **experience replay**.

| | RAG | Evolving Memory |
|---|---|---|
| **What's stored** | Text chunks | Causal trajectories (reasoning + action + result sequences) |
| **Retrieval** | Semantic similarity → flat text | Similarity finds pointer → hierarchical graph traversal |
| **Structure** | Flat document list | Topological graph with typed edges (causal, temporal, hierarchical) |
| **Learning** | Static (re-index to update) | Continuous (dream cycles consolidate, merge, and prune) |
| **Failure handling** | None | Negative constraints extracted and attached to strategies |
| **Ordering** | Lost | Preserved (`NEXT_STEP` / `PREVIOUS_STEP` edges) |
| **Compression** | Token-level chunking | Experience-level merging (similar strategies fuse into one) |
| **Context window** | Floods with unordered fragments | Streams one step at a time, in causal order |
| **Improvement** | Never changes | Confidence increases with repeated success |

RAG answers "what documents are similar to this query?" Evolving Memory answers "have I done something like this before, and if so, what exactly did I do, step by step, and what should I avoid?"

### The Generality Claim

If the hypothesis is correct — that memory consolidation is a domain-agnostic process operating on causal trajectories — then the **exact same engine** should work for:

- A robot navigating a building (RoClaw)
- A software engineer implementing authentication
- A mathematician learning Fourier transforms
- A writer crafting a narrative arc
- A scientist designing experiments
- A chef perfecting sourdough bread

The consolidation mechanism doesn't care what the trajectories represent. It only cares about the **structure**: a sequence of (reasoning, action, result) tuples, organized hierarchically, with causal and temporal dependencies.

We tested this claim. **It holds.**

---

## Experimental Validation

### Setup

We ran 12 tests against the real production stack:

- **Embeddings**: Gemini Embedding 2 Preview (`gemini-embedding-2-preview`), 768 dimensions
- **LLM**: Gemini 2.5 Flash (for ISA opcode emission during dream cycles)
- **Storage**: In-memory SQLite + FAISS (no persistence, clean state per test)
- **Domains tested**: Software Engineering, Mathematics, Creative Writing, Scientific Reasoning, Cooking, Data Analysis, Machine Learning

All 12 tests passed. Total execution time: **143.87 seconds** (most spent on LLM API calls during dream cycles).

### Test 1: Semantic Embedding Quality

**Question**: Do the embeddings capture meaningful semantic relationships?

| Comparison | Cosine Similarity | Verdict |
|------------|-------------------|---------|
| "implement JWT authentication tokens" vs "create JSON web token auth system" | **0.9100** | Synonymous concepts recognized |
| "implement JWT authentication tokens" vs "bake chocolate chip cookies at 350 degrees" | **0.5407** | Unrelated concepts separated |
| "linear algebra matrix multiplication" vs "eigenvalue decomposition" | **0.7202** | Same-domain clustering |
| "linear algebra matrix multiplication" vs "quantum chromodynamics gluon interactions" | **0.5708** | Cross-domain separation |

**Result**: The embedding space correctly clusters related concepts and separates unrelated ones. The gap between same-domain similarity (0.72) and cross-domain similarity (0.57) provides sufficient signal for the router's composite scoring.

### Test 2: Multi-Domain Dream Cycles

**Question**: Does the full capture-dream-query-traverse cycle work identically across fundamentally different domains?

| Domain | Traces | Nodes Created | Edges Created | Query Confidence | Traversal Steps |
|--------|--------|---------------|---------------|------------------|-----------------|
| **Software Engineering** (JWT auth) | 1 | 1 | 7 | 0.884 | 3 steps |
| **Mathematics** (Fourier transforms) | 1 | 1 | 4 | 0.881 | 3 steps |
| **Creative Writing** (narrative arc) | 1 | 1 | 7 | 0.805 | 3 steps |
| **Scientific Reasoning** (experiment design) | 1 | 1 | 7 | 0.833 | 3 steps |

**Result**: The dream engine produced valid consolidated memory for every domain. The LLM (Gemini 2.5 Flash) correctly emitted `MARK_CRITICAL`, `BUILD_PARENT`, and `BUILD_CHILD` ISA opcodes for software, math, literature, and science traces alike. Query confidence ranged from 0.805 (creative writing) to 0.884 (software engineering), all well above the 0.4 router threshold.

**Detailed Software Engineering traversal** (JWT authentication):
```
Query: "how to implement JWT authentication?"
Path: memory_traversal (confidence: 0.884)

  Step 0: Read RFC 7519 and understand claims structure
  Step 1: Write jwt_utils.py with encode(payload, secret) function
  Step 2: Add decode(token, secret) with signature verification
```

The system didn't just find a relevant memory — it replayed the exact procedural sequence needed to implement JWT auth.

### Test 3: Cross-Domain Isolation

**Question**: When multiple domains coexist in the same memory graph, does the router correctly match queries to their own domain?

| Query | Found Memory | Correct? |
|-------|-------------|----------|
| "how to create JSON web tokens?" | "implement JWT tokens" | Yes |
| "how to bake sourdough bread?" | "master sourdough bread" | Yes |

**Result**: Perfect isolation. JWT queries find JWT memories; bread queries find bread memories. The semantic embedding space maintains sufficient separation between unrelated domains even when they share the same FAISS index.

### Test 4: Cross-Trace Linking

**Question**: When the agent has two related but distinct experiences, does the dream engine discover and create causal links between them?

**Setup**:
- Session 1: "understand gradient descent" — study loss functions, implement vanilla GD
- Session 2: "train neural network classifier" — prepare data, train with SGD optimizer

**Result**:
```
Session 1 (theory) — 1 node created
Session 2 (applied) — 1 node created, 1 cross-edge created
Cross-trace edges: gradient_descent → neural_network_classifier (causal)
```

The system correctly identified that "understanding gradient descent" is a **causal prerequisite** for "training a neural network classifier". This cross-link was discovered by the LLM during the consolidation phase — it compared the two strategies and emitted a `LNK_NODE <source_id> <target_id> "causal"` instruction.

This is how **knowledge graphs emerge from episodic traces**. The agent didn't explicitly create a curriculum — the dream engine inferred the dependency structure.

### Test 5: Failure Learning

**Question**: Does the system learn from failures and extract actionable constraints?

**Setup**: A trace of a failed database migration — running directly on production without backup, causing 2 hours of downtime and 8 hours of data loss.

**Result**: The dream engine extracted **4 negative constraints**:

```
1. "Do not run database migrations directly on production without a recent, verified backup."
2. "Do not execute database migrations on production without first testing them in a pre-production environment."
3. "Database migration scripts must be idempotent or include checks for existing schema elements."
4. "Do not perform database operations that lead to irreversible loss of recent user data."
```

These constraints are permanently attached to the memory node and will be surfaced whenever the agent encounters a similar task in the future. This is the artificial equivalent of "don't touch the hot stove" — the system transforms negative experiences into explicit avoidance rules.

### Test 6: Knowledge Accumulation

**Question**: When the agent performs the same task multiple times, does the memory consolidate rather than duplicate?

**Setup**: Two code review sessions with similar but not identical actions.

**Result**:
```
Session 1 — 1 node created, confidence: 1.000
Session 2 — 0 nodes created, 1 merged, confidence: 1.000
```

The second session was **merged** into the existing code review memory node. The system recognized that "review pull request for bugs and style" is the same strategy as "review pull request for bugs" and consolidated them. This is how **procedural memory emerges from episodic traces** — repeated experiences compress into a single, high-confidence skill node.

### Test 7: Multi-Domain Agent Lifecycle

**Question**: Can a single agent accumulate knowledge across different domains and retrieve it contextually?

**Setup**: An agent that (1) learns pytest on Day 1, (2) learns data analysis on Day 2, then (3) is asked "how to write tests for data processing code?" on Day 3.

**Result**:
```
Day 1 (testing) — 1 node, 7 edges
Day 2 (analysis) — 1 node, 7 edges

Query: "how to write tests for data processing code?"
Path: memory_traversal (confidence: 0.845)
Matched: "write pytest test suite" (similarity: 0.690)
Traversing 3 steps:
  [0] Create conftest.py with fixtures → Fixtures for db, client, auth
  [1] Test individual functions with edge cases → 20 tests, all passing
  [2] Test API endpoints end-to-end → 5 integration tests passing

Query: "pytest fixtures and test cases" → found: "write pytest test suite" (correct)
Query: "pandas data analysis trends" → found: "perform sales trend analysis" (correct)
```

The agent correctly retrieved its testing knowledge when faced with a task that combined both domains. Each subsequent query correctly routed to the appropriate domain. The accumulated memory graph functions as a **general-purpose knowledge base** that grows with every work session.

---

## Implications

### For AGI Architecture

These results suggest that the **missing piece for LLM agents** is not more parameters, longer context windows, or better retrieval augmentation. It is a **structured memory lifecycle**:

1. **Capture** structured traces during work (not just text logs — reasoning, actions, results, outcomes)
2. **Consolidate** through a biologically-inspired pipeline (curate, abstract, connect)
3. **Retrieve** through topological traversal (not flat similarity search)
4. **Learn continuously** through merge detection, confidence accumulation, and constraint extraction

This architecture is:

- **Domain-agnostic** — the same engine handles software, math, science, writing, cooking
- **Self-improving** — repeated experience strengthens memory nodes through merging
- **Failure-aware** — negative experiences produce explicit avoidance constraints
- **Compositional** — cross-trace linking discovers prerequisite relationships automatically
- **Context-efficient** — streams one step at a time instead of flooding the context window

### For the Cognitive Trinity

Evolving Memory is the **Hippocampus** in a three-part cognitive architecture:

```
    ┌──────────────────────────────────────┐
    │        llmos (Prefrontal Cortex)      │
    │   Agent Kernel · Applets · UI · LLM  │
    │   Decides what to do, plans, reasons │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │    Evolving Memory (Hippocampus)      │
    │   CTE · Dream Engine · FAISS · ISA   │
    │   Remembers, consolidates, retrieves │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼───────────────────────┐
    │        RoClaw (Cerebellum)            │
    │   Robot · Bytecode · Motor Control   │
    │   Executes physical actions          │
    └──────────────────────────────────────┘
```

The same memory server (REST/WebSocket on port 8420) serves all three layers. A robot's navigation experience and a software agent's coding experience consolidate through the exact same dream pipeline — different `TraceSource` fidelity weights, same ISA, same graph structure.

### What This First Draft Proves (and What It Doesn't)

**Proved**:
- The consolidation pipeline is domain-agnostic (tested across 6 domains)
- Real LLMs can reliably emit ISA opcodes for all three dream phases
- Semantic embeddings provide sufficient signal for routing and cross-linking
- Negative constraint extraction works (4 constraints from 1 failure trace)
- Merge detection works (repeated experiences consolidate correctly)
- Hierarchical traversal reproduces correct procedural sequences

**Not yet tested** (future work):
- Long-term knowledge evolution over hundreds of sessions
- Multi-agent shared memory (multiple agents writing to the same graph)
- Real-world robotics integration via the memory server
- Context jump behavior under real semantic drift
- Performance at scale (thousands of parent nodes in the FAISS index)
- The effect of accumulated constraints on actual task performance

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
- Generates embeddings (Gemini Embedding 2, 768-dim) for parent nodes and adds to FAISS index
- Detects **merges**: if a similar strategy already exists (cosine similarity > 0.85), the new experience is consolidated into the existing node — confidence increases, constraints accumulate
- Creates **edges**: `IS_CHILD_OF` (hierarchical), `NEXT_STEP` / `PREVIOUS_STEP` (temporal/causal)
- Discovers **cross-trace links** via LLM analysis — emits `LNK_NODE` opcodes to create `causal` or `context_jump` edges between related strategies
- Applies **fidelity weights** — real-world experiences (1.0) are trusted more than simulated (0.5) or dreamed (0.3) ones

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

This means a memory with moderate similarity but high confidence and success rate can outrank a highly similar but low-confidence memory — the system **trusts proven experience** over surface-level similarity.

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

### The ISA Advantage

Traditional LLM integration uses JSON for all communication. This is **token-bloated (~25-30 tokens per tool call)**, latency-heavy, and non-deterministic (hallucinated keys, missing brackets, conversational filler).

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
| **Epistemological trust** | Fidelity Weights | Real-world > simulated > dreamed experience |
| **Cross-modal association** | Cross-trace LNK_NODE | "Learning X helped me understand Y" |

The key insight from neuroscience: **the hippocampus uses the same neural machinery for spatial navigation and conceptual reasoning.** Grid cells that help you walk through a house are the same cells that help you solve a math problem step by step. Memory isn't a search engine — it's a **topological graph of causal trajectories**, and the traversal algorithm doesn't care what the trajectories represent.

This is why the same architecture powers both [RoClaw](https://github.com/EvolvingAgentsLabs/RoClaw) (robot navigation through physical space) and Evolving Memory (agent navigation through conceptual space).

---

## Quick Start

### Installation

```bash
pip install evolving-memory

# With LLM providers
pip install evolving-memory[openai]      # OpenAI
pip install evolving-memory[anthropic]   # Anthropic
pip install evolving-memory[all]         # All providers + server
```

### Environment

```bash
export GEMINI_API_KEY="your-key"  # Required for embeddings + Gemini LLM
```

### Usage

```python
import asyncio
from evolving_memory import CognitiveTrajectoryEngine, HierarchyLevel, RouterPath
from evolving_memory.llm.gemini_provider import GeminiProvider

async def main():
    llm = GeminiProvider()
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
        gemini_provider.py   # Gemini via OpenAI-compatible endpoint
        anthropic_provider.py
        openai_provider.py
        prompts.py           # ISA instruction templates

    dream/                   # 3-phase memory consolidation
        curator.py           # Phase 1 SWS: failure analysis, critical path
        chunker.py           # Phase 2 REM: hierarchical node creation
        connector.py         # Phase 3: edges, embeddings, merge detection
        engine.py            # DreamEngine orchestrator
        domain_adapter.py    # DreamDomainAdapter protocol
        adapters/            # Default + Robotics adapters

    capture/                 # Trace capture
        session.py           # SessionManager
        trace_logger.py      # TraceLogger, TraceContext

    models/                  # Pydantic data models
        hierarchy.py         # HierarchyLevel, TraceOutcome, EdgeType, RouterPath
        trace.py             # TraceEntry, ActionEntry, TraceSession
        graph.py             # ParentNode, ChildNode, ThoughtEdge
        strategy.py          # NegativeConstraint, DreamJournalEntry
        query.py             # EntryPoint, RouterDecision, TraversalState
        fidelity.py          # TraceSource fidelity weights

    router/                  # Query routing
        cognitive_router.py  # Tripartite router (Zero-Shot / Traversal / Jump)
        anomaly.py           # Semantic drift detection

    embeddings/
        encoder.py           # Gemini Embedding 2 (768-dim, google-genai SDK)

    storage/
        sqlite_store.py      # SQLite graph store (8 tables)
        vector_index.py      # FAISS vector index

    server/                  # REST/WebSocket API
        app.py               # MemoryServer + FastAPI factory
        routes.py            # HTTP/WS endpoints
```

---

## Testing

```bash
# Run the full test suite (139 tests, no API key needed)
PYTHONPATH=src:tests python3.12 -m pytest tests/ -v

# Run the hypothesis validation tests (12 tests, requires GEMINI_API_KEY)
PYTHONPATH=src:tests GEMINI_API_KEY=<key> python3.12 -m pytest tests/test_real_hypothesis.py -v -s

# Individual modules
pytest tests/test_isa.py              # 29 tests — parser round-trips, all 16 opcodes
pytest tests/test_vm.py               # 25 tests — handlers, programs, safety limits
pytest tests/test_dream_engine.py     # 9 tests — dream cycle with ISA
pytest tests/test_integration.py      # 3 tests — full capture -> dream -> query
pytest tests/test_real_hypothesis.py  # 12 tests — real LLM + real embeddings
```

All unit/integration tests use a `MockLLMProvider` that emits deterministic ISA opcodes — no API keys needed. The hypothesis validation tests (`test_real_hypothesis.py`) use real Gemini APIs.

---

## Configuration

```python
from evolving_memory import CTEConfig

config = CTEConfig(
    db_path="memory.db",
    faiss_path="memory.faiss",
    embedding_model="gemini-embedding-2-preview",
    embedding_dim=768,
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

- **[RoClaw](https://github.com/EvolvingAgentsLabs/RoClaw)** — Robotics bytecode ISA (`AA 01 64 64 CB FF`) for hardware motor control
- **[llmunix / DreamOS](https://github.com/EvolvingAgentsLabs/llmunix-dreamos)** — Operating system layer for LLM agent orchestration
- **Evolving Memory (CTE)** — Cognitive memory with ISA for software agents

All three share the same principle: **LLM as CPU, structured instructions as the interface, Python/firmware as the VM/executor.** The ISA is text-assembly instead of hex bytecode because Python has no hardware memory constraints — but the architecture is identical.

---

## License

Apache 2.0 -- see [LICENSE](LICENSE)
