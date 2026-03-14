"""CLI entry point: python -m evolving_memory.server"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolving Memory Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8420, help="Bind port")
    parser.add_argument("--db", default="memory.db", help="SQLite database path")
    parser.add_argument("--llm", default="mock", choices=["mock", "openai", "anthropic"],
                        help="LLM provider")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'evolving-memory[server]'", file=sys.stderr)
        sys.exit(1)

    from ..config import CTEConfig
    from .app import MemoryServer, create_app

    # Select LLM provider
    if args.llm == "mock":
        from ..llm.base import BaseLLMProvider

        class _MockLLM(BaseLLMProvider):
            async def complete(self, prompt: str, system: str = "") -> str:
                return "HALT"

            async def complete_json(self, prompt: str, system: str = "") -> dict:
                return {}

            async def emit_program(self, prompt: str, system: str = "") -> str:
                return "HALT"

        llm: BaseLLMProvider = _MockLLM()
    elif args.llm == "openai":
        from ..llm.openai_provider import OpenAIProvider
        llm = OpenAIProvider()
    elif args.llm == "anthropic":
        from ..llm.anthropic_provider import AnthropicProvider
        llm = AnthropicProvider()
    else:
        print(f"Unknown LLM provider: {args.llm}", file=sys.stderr)
        sys.exit(1)

    config = CTEConfig(db_path=args.db)
    server = MemoryServer(llm=llm, config=config)
    app = create_app(server)

    print(f"Starting Evolving Memory Server on {args.host}:{args.port}")
    print(f"  Database: {args.db}")
    print(f"  LLM: {args.llm}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
