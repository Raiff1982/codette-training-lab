"""
Forge Engine - Main orchestrator for the multi-agent reasoning forge.

Coordinates the full forge cycle:
  concept -> problem_generator -> each agent analyzes -> critic evaluates
  -> synthesis_engine -> training example

Outputs JSONL training data in OpenAI chat format.
"""

import json
import os
import sys
import random
from typing import TextIO

from reasoning_forge.agents.newton_agent import NewtonAgent
from reasoning_forge.agents.quantum_agent import QuantumAgent
from reasoning_forge.agents.ethics_agent import EthicsAgent
from reasoning_forge.agents.philosophy_agent import PhilosophyAgent
from reasoning_forge.agents.davinci_agent import DaVinciAgent
from reasoning_forge.agents.empathy_agent import EmpathyAgent
from reasoning_forge.agents.critic_agent import CriticAgent
from reasoning_forge.synthesis_engine import SynthesisEngine
from reasoning_forge.problem_generator import ProblemGenerator


SYSTEM_PROMPT = (
    "You are Codette, a multi-perspective reasoning AI. You analyze concepts "
    "by examining them through multiple intellectual lenses -- physics, "
    "philosophy, ethics, creative invention, and human empathy -- then "
    "synthesize a unified understanding that is richer than any single "
    "perspective. You think carefully, acknowledge uncertainty, and connect "
    "abstract reasoning to concrete human experience."
)


class ForgeEngine:
    """Main orchestrator for multi-agent reasoning data generation."""

    def __init__(self):
        # Initialize all reasoning agents
        self.newton = NewtonAgent()
        self.quantum = QuantumAgent()
        self.ethics = EthicsAgent()
        self.philosophy = PhilosophyAgent()
        self.davinci = DaVinciAgent()
        self.empathy = EmpathyAgent()
        self.critic = CriticAgent()

        self.analysis_agents = [
            self.newton,
            self.quantum,
            self.ethics,
            self.philosophy,
            self.davinci,
            self.empathy,
        ]

        # Initialize supporting engines
        self.synthesis = SynthesisEngine()
        self.problem_generator = ProblemGenerator()

    def forge_single(self, concept: str) -> dict:
        """Run full forge cycle on one concept.

        The cycle:
        1. Generate reasoning problems from the concept.
        2. Each analysis agent produces its perspective.
        3. The critic evaluates the ensemble.
        4. The synthesis engine combines everything.
        5. Package as a training example.

        Args:
            concept: The concept text to forge.

        Returns:
            Training example dict in OpenAI chat format:
            {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ],
                "metadata": {
                    "concept": "...",
                    "agent_scores": {...},
                    "overall_quality": float,
                    "problems_generated": int,
                }
            }
        """
        # Step 1: Generate reasoning problems
        problems = self.problem_generator.generate_problems(concept)

        # Step 2: Each agent analyzes the concept
        analyses = {}
        for agent in self.analysis_agents:
            analyses[agent.name] = agent.analyze(concept)

        # Step 3: Critic evaluates the ensemble
        critique = self.critic.evaluate_ensemble(concept, analyses)

        # Step 4: Synthesis engine combines everything
        synthesized_response = self.synthesis.synthesize(
            concept, analyses, critique
        )

        # Step 5: Build the user prompt
        # Randomly choose between a direct concept prompt and a problem-based prompt
        if problems and random.random() < 0.5:
            problem_type, problem_text = random.choice(problems)
            user_content = problem_text
        else:
            user_content = (
                f"Analyze this concept from multiple perspectives:\n\n{concept}"
            )

        # Step 6: Package as training example
        training_example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": synthesized_response},
            ],
            "metadata": {
                "concept": concept,
                "agent_scores": critique.get("agent_scores", {}),
                "overall_quality": critique.get("overall_quality", 0.0),
                "problems_generated": len(problems),
                "problem_types": [p[0] for p in problems],
                "redundancies_found": len(critique.get("redundancies", [])),
                "missing_perspectives": len(
                    critique.get("missing_perspectives", [])
                ),
            },
        }

        return training_example

    def forge_batch(
        self, concept: str, variants: int = 3
    ) -> list[dict]:
        """Generate multiple training examples from one concept.

        Uses different problem framings and agent template selections
        to produce varied training data from the same concept.

        Args:
            concept: The concept text.
            variants: Number of variants to generate.

        Returns:
            List of training example dicts.
        """
        examples = []
        for _ in range(variants):
            example = self.forge_single(concept)
            examples.append(example)
        return examples

    def forge_dataset(
        self,
        concepts: list[str],
        output_path: str,
        variants_per_concept: int = 1,
        verbose: bool = False,
    ) -> dict:
        """Run forge on a list of concepts and write JSONL output.

        Args:
            concepts: List of concept strings.
            output_path: Path to output JSONL file.
            variants_per_concept: Number of training examples per concept.
            verbose: Whether to print progress.

        Returns:
            Summary dict with counts and quality statistics.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        total_examples = 0
        total_quality = 0.0
        quality_scores = []

        with open(output_path, "w", encoding="utf-8") as f:
            for i, concept in enumerate(concepts):
                if verbose:
                    print(
                        f"[{i + 1}/{len(concepts)}] Forging: "
                        f"{concept[:60]}{'...' if len(concept) > 60 else ''}",
                        file=sys.stderr,
                    )

                for variant in range(variants_per_concept):
                    example = self.forge_single(concept)
                    quality = example["metadata"]["overall_quality"]

                    # Write the messages (without metadata) for training
                    training_record = {"messages": example["messages"]}
                    f.write(json.dumps(training_record, ensure_ascii=False) + "\n")

                    total_examples += 1
                    total_quality += quality
                    quality_scores.append(quality)

        summary = {
            "total_examples": total_examples,
            "total_concepts": len(concepts),
            "variants_per_concept": variants_per_concept,
            "output_path": output_path,
            "avg_quality": round(total_quality / max(1, total_examples), 3),
            "min_quality": round(min(quality_scores) if quality_scores else 0, 3),
            "max_quality": round(max(quality_scores) if quality_scores else 0, 3),
        }

        if verbose:
            print(f"\nForge complete: {summary}", file=sys.stderr)

        return summary

    def forge_from_dataset(
        self,
        input_jsonl: str,
        output_path: str,
        concept_field: str = "text",
        variants_per_concept: int = 1,
        verbose: bool = False,
    ) -> dict:
        """Read an existing JSONL dataset and run forge on each entry.

        Expects each line to be a JSON object with a text field containing
        the concept. Supports common field names: 'text', 'concept',
        'content', 'input', 'question', 'prompt'.

        Args:
            input_jsonl: Path to input JSONL file.
            output_path: Path to output JSONL file.
            concept_field: Name of the field containing the concept text.
            variants_per_concept: Number of training examples per concept.
            verbose: Whether to print progress.

        Returns:
            Summary dict with counts and quality statistics.
        """
        # Candidate field names to try
        candidate_fields = [
            concept_field, "text", "concept", "content",
            "input", "question", "prompt",
        ]

        concepts = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if verbose:
                        print(
                            f"Warning: skipping malformed JSON on line {line_num}",
                            file=sys.stderr,
                        )
                    continue

                # Try candidate fields in order
                concept_text = None
                if isinstance(record, dict):
                    for field in candidate_fields:
                        if field in record and isinstance(record[field], str):
                            concept_text = record[field].strip()
                            break
                    # Fallback: if record has 'messages', extract user content
                    if concept_text is None and "messages" in record:
                        for msg in record["messages"]:
                            if msg.get("role") == "user":
                                concept_text = msg["content"].strip()
                                break
                elif isinstance(record, str):
                    concept_text = record.strip()

                if concept_text:
                    concepts.append(concept_text)

        if verbose:
            print(
                f"Loaded {len(concepts)} concepts from {input_jsonl}",
                file=sys.stderr,
            )

        return self.forge_dataset(
            concepts,
            output_path,
            variants_per_concept=variants_per_concept,
            verbose=verbose,
        )

    def forge_single_detailed(self, concept: str) -> dict:
        """Run forge cycle and return all intermediate outputs.

        Useful for debugging, inspection, and quality analysis.

        Args:
            concept: The concept text.

        Returns:
            Dict with all intermediate results:
            {
                "concept": str,
                "problems": [(type, text), ...],
                "analyses": {agent_name: analysis_text, ...},
                "critique": {...},
                "synthesis": str,
                "training_example": {...},
            }
        """
        problems = self.problem_generator.generate_problems(concept)

        analyses = {}
        for agent in self.analysis_agents:
            analyses[agent.name] = agent.analyze(concept)

        critique = self.critic.evaluate_ensemble(concept, analyses)
        synthesized = self.synthesis.synthesize(concept, analyses, critique)

        user_content = (
            f"Analyze this concept from multiple perspectives:\n\n{concept}"
        )

        training_example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": synthesized},
            ],
        }

        return {
            "concept": concept,
            "problems": problems,
            "analyses": analyses,
            "critique": critique,
            "synthesis": synthesized,
            "training_example": training_example,
        }
