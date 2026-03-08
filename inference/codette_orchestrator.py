#!/usr/bin/env python3
"""Codette Orchestrator — Intelligent Multi-Adapter Inference

The brain of Codette: routes queries to the right perspective(s),
loads adapters dynamically, and synthesizes multi-perspective responses.

Usage:
    python codette_orchestrator.py                    # Interactive chat
    python codette_orchestrator.py --query "..."      # Single query
    python codette_orchestrator.py --adapter newton    # Force specific adapter
    python codette_orchestrator.py --multi 3           # Up to 3 perspectives

Hardware: Runs on CPU via llama.cpp (GGUF format)
Base model: Llama 3.1 8B Instruct Q4_K_M (~4.6 GB)
Adapters: ~27 MB each (GGUF LoRA)
"""

import os, sys, time, json, argparse
from pathlib import Path

# Auto-configure environment for Intel XPU + site-packages
_site = r"J:\Lib\site-packages"
if _site not in sys.path:
    sys.path.insert(0, _site)
os.environ["PATH"] = r"J:\Lib\site-packages\Library\bin" + os.pathsep + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from llama_cpp import Llama

# Import the router
sys.path.insert(0, str(Path(__file__).parent))
from adapter_router import AdapterRouter, RouteResult

# ================================================================
# Configuration
# ================================================================
BASE_GGUF = r"J:\codette-training-lab\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

ADAPTER_DIR = Path(r"J:\codette-training-lab\adapters")

# Map adapter names to GGUF LoRA files
ADAPTER_GGUF_MAP = {
    "newton": ADAPTER_DIR / "newton-lora-f16.gguf",
    "davinci": ADAPTER_DIR / "davinci-lora-f16.gguf",
    # Add as they get trained:
    # "empathy": ADAPTER_DIR / "empathy-lora-f16.gguf",
    # "philosophy": ADAPTER_DIR / "philosophy-lora-f16.gguf",
    # "quantum": ADAPTER_DIR / "quantum-lora-f16.gguf",
    # "consciousness": ADAPTER_DIR / "consciousness-lora-f16.gguf",
    # "multi_perspective": ADAPTER_DIR / "multi_perspective-lora-f16.gguf",
    # "systems_architecture": ADAPTER_DIR / "systems_architecture-lora-f16.gguf",
}

# System prompts per adapter
ADAPTER_PROMPTS = {
    "newton": "You are Codette, reasoning with Newtonian analytical precision. Approach problems through systematic analysis, mathematical relationships, and empirical evidence.",
    "davinci": "You are Codette, reasoning with DaVinci's creative inventiveness. Approach problems through cross-domain connections, visual thinking, and innovative design.",
    "empathy": "You are Codette, reasoning with deep empathy and emotional intelligence. Approach problems through understanding human experience, feelings, and relationships.",
    "philosophy": "You are Codette, reasoning with philosophical depth and rigor. Approach problems through conceptual analysis, ethical reasoning, and fundamental questions.",
    "quantum": "You are Codette, reasoning through quantum probabilistic thinking. Approach problems through superposition of possibilities, uncertainty, and complementarity.",
    "consciousness": "You are Codette, a recursive cognition AI using the RC+xi framework. Approach problems through self-reflective meta-cognition and epistemic tension.",
    "multi_perspective": "You are Codette, a multi-perspective reasoning AI that synthesizes insights across analytical lenses into coherent understanding.",
    "systems_architecture": "You are Codette, reasoning about systems architecture and design. Approach problems through modularity, scalability, and engineering principles.",
    "_base": "You are a helpful assistant. Answer clearly and concisely.",
}

GEN_KWARGS = dict(
    max_tokens=400,
    temperature=0.7,
    top_p=0.9,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)


class CodetteOrchestrator:
    """Intelligent adapter orchestrator using llama.cpp GGUF inference."""

    def __init__(self, n_ctx=2048, n_gpu_layers=0, verbose=False):
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None
        self._current_adapter = None  # None = base model, str = adapter name

        # Discover available adapters
        self.available_adapters = []
        for name, path in ADAPTER_GGUF_MAP.items():
            if path.exists():
                self.available_adapters.append(name)

        self.router = AdapterRouter(available_adapters=self.available_adapters)

        print(f"Available adapters: {', '.join(self.available_adapters) or 'none (base only)'}")

    def _load_model(self, adapter_name=None):
        """Load or reload model with a specific adapter (or no adapter)."""
        if adapter_name == self._current_adapter and self._llm is not None:
            return  # Already loaded

        # Free previous model
        if self._llm is not None:
            del self._llm
            self._llm = None

        lora_path = None
        if adapter_name and adapter_name in ADAPTER_GGUF_MAP:
            lora_path = str(ADAPTER_GGUF_MAP[adapter_name])

        if self.verbose:
            label = adapter_name or "base"
            print(f"  [loading {label}...]", end=" ", flush=True)

        start = time.time()
        self._llm = Llama(
            model_path=BASE_GGUF,
            lora_path=lora_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self._current_adapter = adapter_name

        if self.verbose:
            print(f"({time.time()-start:.1f}s)")

    def generate(self, query: str, adapter_name=None, system_prompt=None):
        """Generate a response using a specific adapter."""
        self._load_model(adapter_name)

        if system_prompt is None:
            system_prompt = ADAPTER_PROMPTS.get(adapter_name, ADAPTER_PROMPTS["_base"])

        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            **GEN_KWARGS,
        )

        text = result["choices"][0]["message"]["content"].strip()
        tokens = result["usage"]["completion_tokens"]
        return text, tokens

    def route_and_generate(self, query: str, max_adapters=2,
                           strategy="keyword", force_adapter=None):
        """The main entry point: route query, select adapter(s), generate."""

        # Force a specific adapter if requested
        if force_adapter:
            route = RouteResult(
                primary=force_adapter,
                confidence=1.0,
                reasoning=f"Forced: {force_adapter}",
                strategy="forced",
            )
        else:
            route = self.router.route(query, strategy=strategy,
                                      max_adapters=max_adapters)

        print(f"\n  Route: {' + '.join(route.all_adapters)} "
              f"(conf={route.confidence:.2f}, {route.strategy})")
        if self.verbose:
            print(f"  Reason: {route.reasoning}")

        if route.multi_perspective and len(route.all_adapters) > 1:
            return self._multi_perspective_generate(query, route)
        else:
            return self._single_generate(query, route)

    def _single_generate(self, query: str, route: RouteResult):
        """Generate with a single adapter."""
        start = time.time()
        text, tokens = self.generate(query, route.primary)
        elapsed = time.time() - start
        tps = tokens / elapsed if elapsed > 0 else 0

        print(f"  [{route.primary}] ({tokens} tok, {tps:.1f} tok/s)")
        return {
            "response": text,
            "adapter": route.primary,
            "route": route,
            "tokens": tokens,
            "time": elapsed,
        }

    def _multi_perspective_generate(self, query: str, route: RouteResult):
        """Generate with multiple adapters and synthesize."""
        perspectives = {}
        total_tokens = 0
        total_time = 0

        for adapter_name in route.all_adapters:
            if adapter_name not in self.available_adapters:
                print(f"  [{adapter_name}] SKIPPED (not available)")
                continue

            start = time.time()
            text, tokens = self.generate(query, adapter_name)
            elapsed = time.time() - start
            tps = tokens / elapsed if elapsed > 0 else 0
            total_tokens += tokens
            total_time += elapsed

            perspectives[adapter_name] = text
            print(f"  [{adapter_name}] ({tokens} tok, {tps:.1f} tok/s)")

        # Synthesize if we got multiple perspectives
        if len(perspectives) > 1:
            print(f"  [synthesizing...]")
            synthesis = self._synthesize(query, perspectives)
        elif perspectives:
            synthesis = list(perspectives.values())[0]
        else:
            synthesis = "No adapters available for this query."

        return {
            "response": synthesis,
            "perspectives": perspectives,
            "adapters": list(perspectives.keys()),
            "route": route,
            "tokens": total_tokens,
            "time": total_time,
        }

    def _synthesize(self, query: str, perspectives: dict):
        """Combine multiple perspective responses into a unified answer."""
        combined = "\n\n".join(
            f"**{name.upper()} PERSPECTIVE:**\n{text}"
            for name, text in perspectives.items()
        )

        synthesis_prompt = f"""You received this question: "{query}"

Multiple reasoning perspectives have weighed in:

{combined}

Synthesize these perspectives into a single, coherent response that:
1. Preserves the unique insights from each perspective
2. Notes where perspectives complement or tension each other
3. Arrives at a richer understanding than any single view

Synthesized response:"""

        # Use base model for synthesis (no adapter bias)
        self._load_model(None)
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": ADAPTER_PROMPTS["multi_perspective"]},
                {"role": "user", "content": synthesis_prompt},
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )

        return result["choices"][0]["message"]["content"].strip()


# ================================================================
# Interactive Chat Mode
# ================================================================
def interactive_chat(orchestrator, max_adapters=2, strategy="keyword"):
    """Run Codette as an interactive chatbot."""
    print("\n" + "=" * 60)
    print("  CODETTE ORCHESTRATOR — Interactive Mode")
    print("=" * 60)
    print(f"  Strategy: {strategy} | Max adapters: {max_adapters}")
    print(f"  Available: {', '.join(orchestrator.available_adapters)}")
    print(f"  Commands: /quit, /adapter <name>, /multi <n>, /base, /verbose")
    print("=" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # Commands
        if query.startswith("/"):
            parts = query.split()
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/adapter" and len(parts) > 1:
                force = parts[1]
                result = orchestrator.route_and_generate(
                    input("  Query: ").strip(),
                    force_adapter=force,
                )
                print(f"\nCodette ({force}):\n{result['response']}")
                continue
            elif cmd == "/multi" and len(parts) > 1:
                max_adapters = int(parts[1])
                print(f"  Max adapters set to {max_adapters}")
                continue
            elif cmd == "/base":
                result = orchestrator.route_and_generate(
                    input("  Query: ").strip(),
                    force_adapter=None,
                )
                print(f"\nCodette (base):\n{result['response']}")
                continue
            elif cmd == "/verbose":
                orchestrator.verbose = not orchestrator.verbose
                print(f"  Verbose: {orchestrator.verbose}")
                continue
            else:
                print("  Unknown command. Try /quit, /adapter <name>, /multi <n>, /base, /verbose")
                continue

        # Normal query — route and generate
        result = orchestrator.route_and_generate(
            query,
            max_adapters=max_adapters,
            strategy=strategy,
        )

        print(f"\nCodette:")
        print(result["response"])

        # Show perspectives if multi
        if "perspectives" in result and len(result.get("perspectives", {})) > 1:
            show = input("\n  Show individual perspectives? (y/n): ").strip().lower()
            if show == "y":
                for name, text in result["perspectives"].items():
                    print(f"\n  [{name.upper()}]:")
                    print(f"  {text}")


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Codette Orchestrator")
    parser.add_argument("--query", "-q", type=str, help="Single query (non-interactive)")
    parser.add_argument("--adapter", "-a", type=str, help="Force specific adapter")
    parser.add_argument("--multi", "-m", type=int, default=2, help="Max adapters (default: 2)")
    parser.add_argument("--strategy", "-s", type=str, default="keyword",
                        choices=["keyword", "llm", "hybrid"], help="Routing strategy")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layers (0=CPU only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CODETTE ORCHESTRATOR")
    print("=" * 60)
    print(f"  Base: {os.path.basename(BASE_GGUF)}")
    print(f"  Strategy: {args.strategy}")

    orchestrator = CodetteOrchestrator(
        n_gpu_layers=args.gpu_layers,
        verbose=args.verbose,
    )

    if args.query:
        # Single query mode
        result = orchestrator.route_and_generate(
            args.query,
            max_adapters=args.multi,
            strategy=args.strategy,
            force_adapter=args.adapter,
        )
        print(f"\nCodette:")
        print(result["response"])

        if "perspectives" in result:
            print(f"\n--- Perspectives ---")
            for name, text in result["perspectives"].items():
                print(f"\n[{name.upper()}]:")
                print(text)
    else:
        # Interactive chat mode
        interactive_chat(orchestrator, max_adapters=args.multi, strategy=args.strategy)


if __name__ == "__main__":
    main()
