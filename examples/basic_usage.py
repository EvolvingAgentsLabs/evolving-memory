"""
Evolving Memory — Basic Usage Example

Demonstrates the full CTE cycle:
  1. Capture traces during an agent work session
  2. Run a dream cycle to consolidate into memory graph
  3. Query memory and traverse the thought graph

Requires: An LLM provider API key (set ANTHROPIC_API_KEY or OPENAI_API_KEY)
For testing without API keys, see tests/test_integration.py which uses a mock LLM.
"""

import asyncio
import os
import sys

from evolving_memory import CognitiveTrajectoryEngine, HierarchyLevel, RouterPath


async def main():
    # Choose LLM provider based on available API keys
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from evolving_memory.llm.anthropic_provider import AnthropicProvider
        llm = AnthropicProvider(model="claude-sonnet-4-20250514")
        print("Using Anthropic provider")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from evolving_memory.llm.openai_provider import OpenAIProvider
            llm = OpenAIProvider(model="gpt-4o-mini")
            print("Using OpenAI provider")
        else:
            print("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
            print("For testing without API keys, run: pytest tests/test_integration.py")
            sys.exit(1)

    # Initialize the engine (uses in-memory SQLite for this demo)
    cte = CognitiveTrajectoryEngine(llm=llm, db_path=":memory:")

    # ── Step 1: Capture traces ──────────────────────────────────────
    print("\n=== Step 1: Capturing traces ===")

    with cte.session("build authentication system") as logger:
        with logger.trace(HierarchyLevel.GOAL, "implement auth") as ctx:
            ctx.action(
                "analyze requirements",
                "Read project spec",
                result="Need OAuth2 + JWT tokens",
            )

        with logger.trace(HierarchyLevel.TACTICAL, "implement JWT utilities") as ctx:
            ctx.action(
                "research JWT spec",
                "Read RFC 7519",
                result="Understood claims, signing algorithms",
            )
            ctx.action(
                "write JWT encoder/decoder",
                "Create jwt_utils.py with encode() and decode()",
                result="200 lines of code, supports HS256 and RS256",
            )
            ctx.action(
                "write unit tests",
                "Create test_jwt.py with 8 test cases",
                result="All 8 tests passing",
            )
            ctx.tag("jwt", "auth", "security")

        with logger.trace(HierarchyLevel.TACTICAL, "implement OAuth2 flow") as ctx:
            ctx.action(
                "design OAuth flow",
                "Draw sequence diagram for authorization code flow",
                result="Flow: redirect → code → token exchange → validate",
            )
            ctx.action(
                "implement OAuth client",
                "Create oauth_client.py with authorize() and callback()",
                result="150 lines, handles token exchange and refresh",
            )
            ctx.action(
                "integration testing",
                "Run full OAuth flow against test server",
                result="3/3 integration tests passing",
            )
            ctx.tag("oauth", "auth", "security")

    print(f"  Captured {len(logger.traces)} traces")

    # ── Step 2: Dream (consolidate into memory graph) ───────────────
    print("\n=== Step 2: Running dream cycle (ISA opcode engine) ===")

    journal = await cte.dream()
    print(f"  Traces processed: {journal.traces_processed}")
    print(f"  Nodes created: {journal.nodes_created}")
    print(f"  Nodes merged: {journal.nodes_merged}")
    print(f"  Edges created: {journal.edges_created}")
    print(f"  Constraints: {journal.constraints_extracted}")
    for log in journal.phase_log:
        print(f"  [{log}]")

    # ── Step 3: Query memory ────────────────────────────────────────
    print("\n=== Step 3: Querying memory ===")

    queries = [
        "how to implement JWT authentication?",
        "what is the OAuth2 authorization code flow?",
        "how to deploy a Kubernetes cluster?",  # Not in memory
    ]

    for q in queries:
        decision = cte.query(q)
        print(f"\n  Query: {q}")
        print(f"  Path: {decision.path}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Reasoning: {decision.reasoning}")

        if decision.path == RouterPath.MEMORY_TRAVERSAL and decision.entry_point:
            print(f"  Entry point: {decision.entry_point.parent_node.goal}")
            print(f"  Similarity: {decision.entry_point.similarity_score:.3f}")

            # Traverse the memory step by step
            state = cte.begin_traversal(decision.entry_point)
            print(f"  Traversing {state.total_steps} steps:")
            while True:
                child, state = cte.next_step(state)
                if child is None:
                    break
                print(f"    Step {child.step_index}: {child.summary}")
                print(f"      Action: {child.action}")
                print(f"      Result: {child.result}")

    cte.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
