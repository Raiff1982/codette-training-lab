"""
Forge Engine - Main orchestrator for the multi-agent reasoning forge.

Coordinates the full forge cycle:
  concept -> problem_generator -> each agent analyzes -> critic evaluates
  -> (feedback loop: weak agents revise) -> synthesis_engine -> training example

Supports three modes:
  1. forge_single()       — Original single-pass (fast, good for bulk generation)
  2. forge_with_feedback() — Closed critic loop (agents revise based on scores)
  3. forge_with_debate()   — Multi-turn debate (agents challenge each other)

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
from reasoning_forge.epistemic_metrics import EpistemicMetrics


SYSTEM_PROMPT = (
    "You are Codette, a multi-perspective reasoning AI. You analyze concepts "
    "by examining them through multiple intellectual lenses -- physics, "
    "philosophy, ethics, creative invention, and human empathy -- then "
    "synthesize a unified understanding that is richer than any single "
    "perspective. You think carefully, acknowledge uncertainty, and connect "
    "abstract reasoning to concrete human experience."
)

# Score below which an agent gets sent back for revision
_REVISION_THRESHOLD = 0.6


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
        self.epistemic = EpistemicMetrics()

    def forge_single(self, concept: str) -> dict:
        """Run full forge cycle on one concept (original single-pass mode).

        The cycle:
        1. Generate reasoning problems from the concept.
        2. Each analysis agent produces its perspective.
        3. The critic evaluates the ensemble.
        4. The synthesis engine combines everything.
        5. Package as a training example.

        Args:
            concept: The concept text to forge.

        Returns:
            Training example dict in OpenAI chat format.
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
        if problems and random.random() < 0.5:
            problem_type, problem_text = random.choice(problems)
            user_content = problem_text
        else:
            user_content = (
                f"Analyze this concept from multiple perspectives:\n\n{concept}"
            )

        # Step 6: Compute RC+xi epistemic metrics
        epistemic_report = self.epistemic.full_epistemic_report(
            analyses, synthesized_response
        )

        # Step 7: Package as training example
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
                "epistemic_tension": epistemic_report.get("tension_magnitude", 0),
                "ensemble_coherence": epistemic_report.get("ensemble_coherence", 0),
                "perspective_coverage": epistemic_report.get("perspective_coverage", {}),
                "tension_productivity": epistemic_report.get("tension_productivity", {}),
            },
        }

        return training_example

    # -- Closed Critic Feedback Loop (new) ---------------------------------

    def forge_with_feedback(
        self,
        concept: str,
        max_revisions: int = 2,
    ) -> dict:
        """Run forge with closed critic feedback loop.

        After initial analysis, the critic scores each agent. Agents scoring
        below the revision threshold are sent back with specific critique
        for a second attempt. The best version (original or revised) is kept.

        Args:
            concept: The concept text to forge.
            max_revisions: Maximum revision rounds per weak agent.

        Returns:
            Training example dict with revision metadata.
        """
        problems = self.problem_generator.generate_problems(concept)

        # Initial analysis pass
        analyses = {}
        for agent in self.analysis_agents:
            analyses[agent.name] = agent.analyze(concept)

        revision_counts = {agent.name: 0 for agent in self.analysis_agents}

        for revision_round in range(max_revisions):
            critique = self.critic.evaluate_ensemble(concept, analyses)
            agent_scores = critique.get("agent_scores", {})
            suggestions = critique.get("improvement_suggestions", [])

            # Find agents below threshold
            weak_agents = [
                agent for agent in self.analysis_agents
                if agent_scores.get(agent.name, {}).get("combined", 1.0) < _REVISION_THRESHOLD
            ]

            if not weak_agents:
                break  # All agents above threshold — converged

            for agent in weak_agents:
                score = agent_scores.get(agent.name, {})
                # Build revision directive from critic feedback
                directive = self._build_revision_directive(
                    agent.name, score, suggestions, concept
                )
                # Agent re-analyzes with the directive prepended to concept
                revised = agent.analyze(f"{directive}\n\n{concept}")

                # Keep revision only if it scores better
                old_score = score.get("combined", 0)
                new_critique = self.critic.evaluate_ensemble(
                    concept, {agent.name: revised}
                )
                new_score = new_critique.get("agent_scores", {}).get(
                    agent.name, {}
                ).get("combined", 0)

                if new_score > old_score:
                    analyses[agent.name] = revised
                revision_counts[agent.name] += 1

        # Final critique and synthesis
        final_critique = self.critic.evaluate_ensemble(concept, analyses)
        synthesized = self.synthesis.synthesize(concept, analyses, final_critique)
        epistemic_report = self.epistemic.full_epistemic_report(analyses, synthesized)

        if problems and random.random() < 0.5:
            problem_type, problem_text = random.choice(problems)
            user_content = problem_text
        else:
            user_content = f"Analyze this concept from multiple perspectives:\n\n{concept}"

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": synthesized},
            ],
            "metadata": {
                "concept": concept,
                "agent_scores": final_critique.get("agent_scores", {}),
                "overall_quality": final_critique.get("overall_quality", 0.0),
                "problems_generated": len(problems),
                "revision_counts": revision_counts,
                "total_revisions": sum(revision_counts.values()),
                "epistemic_tension": epistemic_report.get("tension_magnitude", 0),
                "ensemble_coherence": epistemic_report.get("ensemble_coherence", 0),
                "tension_productivity": epistemic_report.get("tension_productivity", {}),
                "forge_mode": "feedback_loop",
            },
        }

    # -- Multi-Turn Debate (new) -------------------------------------------

    def forge_with_debate(
        self,
        concept: str,
        debate_rounds: int = 2,
    ) -> dict:
        """Run forge with multi-turn agent debate.

        Each round:
        1. All agents produce their analysis.
        2. Random pairs are formed for cross-perspective challenge.
        3. Each agent in a pair sees the other's analysis and produces
           a response that engages with it.
        4. Epistemic tension is tracked per round.
        5. After all rounds, synthesis incorporates debate history.

        Args:
            concept: The concept text to forge.
            debate_rounds: Number of debate rounds.

        Returns:
            Training example with debate history and tension decay metrics.
        """
        problems = self.problem_generator.generate_problems(concept)

        # Round 0: initial analyses
        analyses = {}
        for agent in self.analysis_agents:
            analyses[agent.name] = agent.analyze(concept)

        round_analyses = [dict(analyses)]  # snapshot for tension tracking
        debate_log = []

        for round_num in range(debate_rounds):
            # Form random pairs
            agents_shuffled = list(self.analysis_agents)
            random.shuffle(agents_shuffled)
            pairs = []
            for i in range(0, len(agents_shuffled) - 1, 2):
                pairs.append((agents_shuffled[i], agents_shuffled[i + 1]))

            round_debates = []
            for agent_a, agent_b in pairs:
                # Agent A sees B's analysis and responds
                challenge_prompt = (
                    f"Another perspective on '{concept}' argues:\n\n"
                    f"{analyses[agent_b.name]}\n\n"
                    f"Respond to this from your {agent_a.perspective} perspective. "
                    f"Where do you agree, disagree, or see complementary insights?"
                )
                response_a = agent_a.analyze(challenge_prompt)

                # Agent B sees A's response
                counter_prompt = (
                    f"A {agent_a.perspective} perspective responded to your analysis "
                    f"of '{concept}':\n\n{response_a}\n\n"
                    f"Integrate their insights with your own view."
                )
                response_b = agent_b.analyze(counter_prompt)

                # Update analyses with debate-enriched versions
                analyses[agent_a.name] = response_a
                analyses[agent_b.name] = response_b

                round_debates.append({
                    "pair": f"{agent_a.name}_vs_{agent_b.name}",
                    "challenge": response_a[:200],
                    "counter": response_b[:200],
                })

            debate_log.append({
                "round": round_num + 1,
                "debates": round_debates,
            })
            round_analyses.append(dict(analyses))

        # Track tension decay across rounds
        convergence = self.epistemic.score_debate_convergence(round_analyses)

        # Final critique and synthesis
        critique = self.critic.evaluate_ensemble(concept, analyses)
        synthesized = self.synthesis.synthesize(concept, analyses, critique)
        epistemic_report = self.epistemic.full_epistemic_report(analyses, synthesized)

        if problems and random.random() < 0.5:
            problem_type, problem_text = random.choice(problems)
            user_content = problem_text
        else:
            user_content = f"Analyze this concept from multiple perspectives:\n\n{concept}"

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": synthesized},
            ],
            "metadata": {
                "concept": concept,
                "agent_scores": critique.get("agent_scores", {}),
                "overall_quality": critique.get("overall_quality", 0.0),
                "problems_generated": len(problems),
                "debate_rounds": debate_rounds,
                "debate_log": debate_log,
                "tension_decay": convergence,
                "epistemic_tension": epistemic_report.get("tension_magnitude", 0),
                "ensemble_coherence": epistemic_report.get("ensemble_coherence", 0),
                "tension_productivity": epistemic_report.get("tension_productivity", {}),
                "forge_mode": "debate",
            },
        }

    # -- Helpers -----------------------------------------------------------

    def _build_revision_directive(
        self,
        agent_name: str,
        score: dict,
        suggestions: list,
        concept: str,
    ) -> str:
        """Build a revision directive for a weak agent."""
        parts = [
            f"[REVISION REQUESTED for {agent_name}]",
            f"Your previous analysis scored {score.get('combined', 0):.2f}/1.00.",
        ]
        if score.get("logical_clarity", 1) < 0.5:
            parts.append(
                "Improve logical clarity: use connectives (therefore, because, however), "
                "avoid vague language, structure your argument explicitly."
            )
        if score.get("conceptual_accuracy", 1) < 0.5:
            parts.append(
                "Improve conceptual accuracy: engage directly with the specific concept, "
                "use domain vocabulary, avoid generic placeholder framing."
            )
        if suggestions:
            parts.append(f"Critic suggests: {suggestions[0]}")
        parts.append("Reanalyze with these improvements:")
        return " ".join(parts)

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
