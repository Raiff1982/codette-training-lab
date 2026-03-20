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
import logging
from typing import TextIO, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from reasoning_forge.token_confidence import TokenConfidenceEngine
from reasoning_forge.conflict_engine import ConflictEngine, ConflictTracker
from reasoning_forge.memory_weighting import MemoryWeighting
from reasoning_forge.coherence_field import CoherenceFieldGamma


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

    def __init__(self, living_memory=None, enable_memory_weighting=True):
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

        # Store living_memory for Phase 2
        self.living_memory = living_memory

        # Initialize Phase 1: Conflict detection engines (now with wired living_memory for Phase 2)
        self.token_confidence = TokenConfidenceEngine(living_memory=living_memory)

        # === Phase 6: Initialize Semantic Tension Engine ===
        # Replaces discrete opposition_score with embedding-based semantic tension
        try:
            from reasoning_forge.semantic_tension import SemanticTensionEngine
            # Try to use Llama embeddings if available, otherwise use dummy embeddings for testing
            llama_model = getattr(self, 'llama_model', None)
            self.semantic_tension_engine = SemanticTensionEngine(llama_model=llama_model)
        except Exception as e:
            logger.warning(f"Could not initialize SemanticTensionEngine: {e}, using heuristics only")
            self.semantic_tension_engine = None

        self.conflict_engine = ConflictEngine(
            token_confidence_engine=self.token_confidence,
            semantic_tension_engine=self.semantic_tension_engine  # Phase 6
        )

        # Initialize Phase 2: Memory-weighted adapter selection
        if enable_memory_weighting and living_memory:
            self.memory_weighting = MemoryWeighting(living_memory)
            # === Phase 4: Wire into conflict engine for experience-aware strength ===
            self.conflict_engine.memory_weighting = self.memory_weighting
        else:
            self.memory_weighting = None

        # === Phase 5A: Initialize Γ (Gamma) stabilization field ===
        # Real-time health monitoring to prevent weight drift, false convergence, and feedback lock-in
        self.coherence_field = CoherenceFieldGamma(memory_weighting=self.memory_weighting)

        # === Phase 6: Initialize Specialization Tracker ===
        # Track domain-specific performance to prevent semantic convergence
        try:
            from reasoning_forge.specialization_tracker import SpecializationTracker
            self.specialization = SpecializationTracker()
        except Exception as e:
            logger.warning(f"Could not initialize SpecializationTracker: {e}")
            self.specialization = None

        # === Phase 6: Initialize Pre-Flight Conflict Predictor ===
        # Predict conflicts before debate using Spiderweb injection
        try:
            from reasoning_forge.preflight_predictor import PreFlightConflictPredictor
            spiderweb = getattr(self, 'spiderweb', None)
            self.preflight_predictor = PreFlightConflictPredictor(
                spiderweb=spiderweb,
                memory_weighting=self.memory_weighting,
                semantic_engine=self.semantic_tension_engine
            )
        except Exception as e:
            logger.warning(f"Could not initialize PreFlightConflictPredictor: {e}")
            self.preflight_predictor = None

        # === Pre-compute adapter map for Phase 5A efficiency (avoid per-round recomputation) ===
        self._adapter_map = {agent.name.lower(): agent for agent in self.analysis_agents}

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

                # Keep revision only if it scores better (evaluate in full ensemble context)
                old_score = score.get("combined", 0)
                test_analyses = dict(analyses)
                test_analyses[agent.name] = revised
                new_critique = self.critic.evaluate_ensemble(
                    concept, test_analyses
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

    # === PATCH 5: Agent Relevance Gating Helper Methods ===
    def _classify_query_domain(self, query: str) -> str:
        """
        Classify the domain/intent of a query.
        Returns: 'physics', 'ethics', 'consciousness', 'creativity', 'systems', or 'general'
        """
        query_lower = query.lower()

        # Domain keywords
        domains = {
            'physics': ['speed', 'light', 'entropy', 'time', 'quantum', 'particle', 'force', 'energy', 'wave', 'matter'],
            'ethics': ['moral', 'right', 'wrong', 'ethical', 'should', 'ought', 'duty', 'consequence', 'virtue', 'lie', 'transparency', 'explain'],
            'consciousness': ['conscious', 'aware', 'mind', 'experience', 'qualia', 'sentient', 'machine', 'feel', 'perception'],
            'creativity': ['creative', 'invent', 'imagine', 'novel', 'original', 'artistic', 'design', 'innovate'],
            'systems': ['system', 'emerge', 'adapt', 'stability', 'complexity', 'feedback', 'balance', 'equilibrium'],
        }

        # Count keyword matches per domain
        matches = {}
        for domain, keywords in domains.items():
            matches[domain] = sum(1 for kw in keywords if kw in query_lower)

        # Return domain with most matches, or 'general'
        if max(matches.values()) > 0:
            return max(matches, key=matches.get)
        return 'general'

    def _get_agents_for_domain(self, domain: str) -> List:
        """
        Return agents relevant to the detected domain.
        Maps domains to agent specializations.
        """
        domain_agents = {
            'physics': ['Newton', 'Quantum'],
            'ethics': ['Philosophy', 'Empathy'],
            'consciousness': ['Philosophy', 'Quantum'],
            'creativity': ['DaVinci', 'Quantum'],
            'systems': ['Quantum', 'Philosophy'],
            'general': self.analysis_agents,  # Use all agents
        }

        selected_domain_agents = domain_agents.get(domain, self.analysis_agents)

        # Filter to only agents in analysis_agents list
        agent_names = {agent.name for agent in self.analysis_agents}
        active_agents = [
            agent for agent in self.analysis_agents
            if agent.name in selected_domain_agents
        ]

        # Always include critic/synthesizer if available
        return active_agents if active_agents else self.analysis_agents

    def _should_skip_further_rounds(self, gamma_metrics) -> bool:
        """
        === PATCH 4: Gamma Authority ===
        Check if system health is too poor to continue debate.
        If gamma < 0.3 (true collapse), system is failing and should stop.

        Threshold tuned: 0.3 allows healthy debate while preventing runaway.
        """
        if gamma_metrics is None:
            return False

        gamma_value = gamma_metrics.gamma if hasattr(gamma_metrics, 'gamma') else 0.5

        # Hard stop threshold: only stop if truly collapsing
        if gamma_value < 0.3:
            logger.warning(f"Gamma below collapse threshold ({gamma_value:.2f} < 0.3). Stopping debate.")
            return True

        return False

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

        # === PATCH 5: Agent Relevance Gating (Query-Specific Activation) ===
        detected_domain = self._classify_query_domain(concept)
        active_agents = self._get_agents_for_domain(detected_domain)
        logger.info(f"Domain-gated activation: detected '{detected_domain}' → {len(active_agents)} agents active")

        # Round 0: initial analyses
        analyses = {}
        for agent in active_agents:
            analyses[agent.name] = agent.analyze(concept)

        round_analyses = [dict(analyses)]  # snapshot for tension tracking
        debate_log = []

        # === Phase 3/4: Initialize conflict tracker ===
        if not hasattr(self, "conflict_tracker"):
            self.conflict_tracker = ConflictTracker(self.conflict_engine)
        else:
            # Reset for new debate cycle
            self.conflict_tracker.evolution_data = {}

        # === Phase 1: Conflict Detection Round 0 ===
        # Detect initial conflicts before debate begins
        conflicts_round_0 = self.conflict_engine.detect_conflicts(analyses)
        conflicts_summary_r0 = self.conflict_engine.summarize_conflicts(conflicts_round_0)

        # === Phase 3: Initialize conflict tracker ===
        tracker = ConflictTracker(self.conflict_engine)
        tracker.track_round(0, analyses, [])  # Log Round 0 conflicts

        # === Phase 2: Store conflicts in memory for learning ===
        if self.living_memory and conflicts_round_0:
            for conflict in conflicts_round_0[:5]:  # Store top 5 by strength
                resolution_outcome = {
                    "original_concept": concept,
                    "conflict_type": conflict.conflict_type,
                    "agents": [conflict.agent_a, conflict.agent_b],
                    "initial_strength": conflict.conflict_strength,
                }
                self.living_memory.store_conflict(conflict.to_dict(), resolution_outcome)

        debate_log.append({
            "round": 0,
            "type": "initial_analysis",
            "conflicts_detected": len(conflicts_round_0),
            "conflict_strength_summary": conflicts_summary_r0,
            "conflicts": [c.to_dict() for c in conflicts_round_0] if conflicts_round_0 else [],
        })

        # === Phase 6: Pre-Flight Conflict Prediction ===
        # Predict which adapter pairs will clash BEFORE debate starts
        preflight_prediction = None
        agent_names = [agent.name for agent in self.analysis_agents]

        if self.preflight_predictor:
            try:
                preflight_prediction = self.preflight_predictor.predict_conflicts(concept, agent_names)
                debate_log[0]["preflight_prediction"] = preflight_prediction.to_dict()
            except Exception as e:
                logger.debug(f"Pre-flight prediction failed: {e}")

        # === Phase 6: Update specialization tracking (after Round 0) ===
        if self.specialization:
            for agent in self.analysis_agents:
                try:
                    cohesion = 0.5  # Default, will be updated after synthesis
                    self.specialization.record_adapter_performance(agent.name, concept, cohesion)
                except Exception as e:
                    logger.debug(f"Specialization tracking failed: {e}")

        # Multi-round debate loop (Phase 3: supports up to 4 rounds)
        current_round_conflicts = conflicts_round_0

        for round_num in range(1, min(debate_rounds + 1, 4)):
            # Form random pairs (odd agent out debates the first agent)
            agents_shuffled = list(active_agents)
            random.shuffle(agents_shuffled)
            pairs = []
            for i in range(0, len(agents_shuffled) - 1, 2):
                pairs.append((agents_shuffled[i], agents_shuffled[i + 1]))
            if len(agents_shuffled) % 2 == 1:
                pairs.append((agents_shuffled[-1], agents_shuffled[0]))

            round_debates = []
            for agent_a, agent_b in pairs:
                # Check if these agents had conflicts in previous round
                conflict_context = ""
                for conflict in current_round_conflicts:
                    if (conflict.agent_a == agent_a.name and conflict.agent_b == agent_b.name) or \
                       (conflict.agent_a == agent_b.name and conflict.agent_b == agent_a.name):
                        if conflict.conflict_strength > 0.2:  # Only mention significant conflicts
                            conflict_context += (
                                f"\nNote: A {conflict.conflict_type} was detected on: "
                                f"\"{conflict.claim_a if conflict.agent_a == agent_a.name else conflict.claim_b}\" "
                                f"(your confidence: {conflict.confidence_a if conflict.agent_a == agent_a.name else conflict.confidence_b:.2f})"
                            )

                # Agent A sees B's analysis and responds
                challenge_prompt = (
                    f"Another perspective on '{concept}' argues:\n\n"
                    f"{analyses[agent_b.name]}\n\n"
                    f"Respond to this from your {agent_a.perspective} perspective. "
                    f"Where do you agree, disagree, or see complementary insights?"
                    f"{conflict_context}"
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

            # === Phase 3: Track conflict evolution for this round ===
            round_evolutions = self.conflict_tracker.track_round(round_num, analyses, current_round_conflicts)

            # Detect conflicts for next round
            new_round_conflicts = self.conflict_engine.detect_conflicts(analyses)

            # ========================================================================
            # === Phase 4: Self-Correcting Feedback Loop ===
            # ========================================================================

            # A. Update adapter weights based on resolution performance
            if self.memory_weighting:
                for evolution in round_evolutions:
                    self.memory_weighting.update_from_evolution(evolution)

            # B. Dynamic rerouting: inject best adapter if conflicts remain high
            override = self._dynamic_reroute(new_round_conflicts)
            if override and override not in analyses:
                try:
                    analyses[override] = self._run_adapter(override, concept)
                    # Re-detect conflicts with new perspective
                    new_round_conflicts = self.conflict_engine.detect_conflicts(analyses)
                except Exception as e:
                    logger.debug(f"Dynamic reroute failed: {e}")

            # C. Runaway detection: if tensions increasing, inject stabilizer
            if new_round_conflicts and current_round_conflicts:
                avg_old = sum(c.conflict_strength for c in current_round_conflicts) / len(current_round_conflicts)
                avg_new = sum(c.conflict_strength for c in new_round_conflicts) / len(new_round_conflicts)

                if avg_new > avg_old * 1.1:  # 10% increase = runaway
                    try:
                        # Inject multi-perspective synthesis
                        analyses["multi_perspective"] = self._run_adapter("multi_perspective", concept)
                        new_round_conflicts = self.conflict_engine.detect_conflicts(analyses)
                    except Exception as e:
                        logger.debug(f"Runaway stabilization failed: {e}")

            # === Phase 5A: Gamma Stabilization Field ===
            # Monitor system health and intervene if drifting toward failure modes
            gamma_metrics = self.coherence_field.compute_health(
                conflicts=new_round_conflicts,
                responses=analyses,
                adapter_weights=(
                    self.memory_weighting.adapter_weights if self.memory_weighting else None
                ),
            )

            # Override resolution_rate with actual evolution data if available
            if round_evolutions:
                avg_resolution = sum(e.resolution_rate for e in round_evolutions) / len(round_evolutions)
                # Manually update the metrics (no method exists, so we replace the field)
                gamma_metrics.resolution_rate = avg_resolution

            # Check if intervention is needed
            gamma_intervention = self.coherence_field.get_intervention(
                gamma_metrics,
                available_adapters=list(self._adapter_map.keys()),
            )

            if gamma_intervention:
                logger.info(f"Gamma intervention triggered: {gamma_intervention.intervention_type} "
                           f"(γ={gamma_metrics.gamma:.2f})")
                # Execute intervention if recommended adapter is valid
                if gamma_intervention.recommended_adapter and ";" not in gamma_intervention.recommended_adapter:
                    try:
                        override = gamma_intervention.recommended_adapter
                        if override not in analyses:
                            analyses[override] = self._run_adapter(override, concept)
                            new_round_conflicts = self.conflict_engine.detect_conflicts(analyses)
                            gamma_intervention.result = f"Injected {override} perspective"
                    except Exception as e:
                        gamma_intervention.result = f"Injection failed: {e}"
                        logger.debug(f"Gamma intervention injection failed: {e}")

            # === PATCH 4: Gamma Authority (Hard Stop If System Collapsing) ===
            if self._should_skip_further_rounds(gamma_metrics):
                logger.warning(f"SYSTEM COLLAPSE DETECTED: Stopping debate at round {round_num}. Forcing synthesis.")
                debate_log.append({
                    "round": round_num,
                    "type": "emergency_stop",
                    "reason": "gamma < 0.5 threshold",
                    "gamma_score": gamma_metrics.gamma,
                })
                break  # Exit the debate loop early

            debate_log.append({
                "round": round_num,
                "type": "debate",
                "debates": round_debates,
                "conflicts_detected": len(new_round_conflicts),
                "conflict_evolution": [
                    {
                        "agents": f"{e.original_conflict.agent_a}_vs_{e.original_conflict.agent_b}",
                        "initial_strength": e.original_conflict.conflict_strength,
                        "current_strength": e.round_trajectories.get(round_num, {}).get("strength", 0),
                        "resolution_type": e.resolution_type,
                        "resolution_rate": e.resolution_rate,
                    }
                    for e in round_evolutions
                ],
                "resolution_metrics": {
                    "conflicts_before": len(current_round_conflicts),
                    "conflicts_after": len(new_round_conflicts),
                    "resolution_rate": (len(current_round_conflicts) - len(new_round_conflicts)) / max(len(current_round_conflicts), 1),
                } if current_round_conflicts else {},
                # === Phase 5A: Gamma health monitoring ===
                "gamma_health": {
                    "gamma": round(gamma_metrics.gamma, 3),
                    "status": gamma_metrics.status if hasattr(gamma_metrics, 'status') else "unknown",
                    "conflict_strength": round(gamma_metrics.avg_conflict_strength, 3),
                    "perspective_diversity": round(gamma_metrics.perspective_diversity, 3),
                    "weight_variance": round(gamma_metrics.adapter_weight_variance, 3),
                    "intervention": (
                        {
                            "type": gamma_intervention.intervention_type,
                            "reason": gamma_intervention.reason,
                            "adapter": gamma_intervention.recommended_adapter,
                            "result": gamma_intervention.result,
                        }
                        if gamma_intervention else None
                    ),
                },
            })

            current_round_conflicts = new_round_conflicts
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
                "debate_rounds": min(debate_rounds, 3),  # Cap for Phase 3
                "debate_log": debate_log,
                "tension_decay": convergence,
                "epistemic_tension": epistemic_report.get("tension_magnitude", 0),
                "ensemble_coherence": epistemic_report.get("ensemble_coherence", 0),
                "tension_productivity": epistemic_report.get("tension_productivity", {}),
                "forge_mode": "debate",
                # === Phase 1: Conflict Detection Metrics ===
                "conflicts_round_0_count": len(conflicts_round_0),
                "conflicts_detected": [c.to_dict() for c in conflicts_round_0[:5]],  # Top 5
                "conflict_summary": conflicts_summary_r0,
                # === Phase 3: Conflict Evolution Metrics ===
                "phase_3_metrics": self.conflict_tracker.get_summary(),
                "evolution_data": [
                    {
                        "agents": key,
                        "resolved_in_round": e.resolved_in_round,
                        "resolution_rate": e.resolution_rate,
                        "resolution_type": e.resolution_type,
                        "trajectory": list(e.round_trajectories.values()),
                    }
                    for key, e in self.conflict_tracker.evolution_data.items()
                ],
                # === Phase 4: Self-Correcting Feedback Metrics ===
                "phase_4_active": self.memory_weighting is not None,
                "adapter_weights": (
                    self.memory_weighting.get_all_weights()
                    if self.memory_weighting
                    else {}
                ),
                # === Phase 5A: Gamma Stabilization Metrics ===
                "phase_5a_active": True,
                "gamma_metrics": self.coherence_field.export_metrics(),
                # === Phase 6: Semantic Tension & Specialization Metrics ===
                "phase_6_active": True,
                "semantic_tension_engine_ready": self.semantic_tension_engine is not None,
                "specialization_metrics": (
                    self.specialization.export_summary() if self.specialization else {}
                ),
                "preflight_prediction": preflight_prediction.to_dict() if preflight_prediction else {},
            },
        }

    # -- Helpers -----------------------------------------------------------

    def _dynamic_reroute(self, conflicts: List) -> Optional[str]:
        """
        Dynamically select best-performing adapter when conflicts are high.

        Phase 4: Real-time adaptation - inject the strongest adapter when
        conflicts exceed threshold.

        Args:
            conflicts: List of Conflict objects from current round

        Returns:
            Best adapter name to inject, or None if not needed
        """
        if not conflicts or not self.memory_weighting:
            return None

        # Find high-conflict situations
        high_conflicts = [c for c in conflicts if c.conflict_strength > 0.2]

        if not high_conflicts:
            return None

        weights = self.memory_weighting.get_all_weights()

        if not weights:
            return None

        # Select best-performing adapter
        best_adapter = max(weights.items(), key=lambda x: x[1]["weight"])[0]

        return best_adapter

    def _run_adapter(self, adapter_name: str, concept: str) -> str:
        """
        Run a specific adapter/agent to generate analysis.

        Phase 4: Helper for dynamic rerouting.

        Args:
            adapter_name: Name of adapter to run
            concept: Concept to analyze

        Returns:
            Analysis text
        """
        for agent in self.analysis_agents:
            if agent.name.lower() == adapter_name.lower():
                return agent.analyze(concept)

        # Fallback: synthesis engine as generic perspective
        return f"Generic perspective on {concept[:50]}..."

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
