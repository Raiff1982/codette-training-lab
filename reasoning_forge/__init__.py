"""
Reasoning Forge - Multi-Agent Reasoning Training Data Generator

The reasoning forge takes concepts and generates high-quality multi-perspective
reasoning training data. Each agent analyzes from its unique perspective, a critic
evaluates the ensemble, and a synthesis engine combines them into coherent training examples.
"""

from reasoning_forge.forge_engine import ForgeEngine
from reasoning_forge.agents.base_agent import ReasoningAgent
from reasoning_forge.agents.newton_agent import NewtonAgent
from reasoning_forge.agents.quantum_agent import QuantumAgent
from reasoning_forge.agents.ethics_agent import EthicsAgent
from reasoning_forge.agents.philosophy_agent import PhilosophyAgent
from reasoning_forge.agents.davinci_agent import DaVinciAgent
from reasoning_forge.agents.empathy_agent import EmpathyAgent
from reasoning_forge.agents.critic_agent import CriticAgent
from reasoning_forge.synthesis_engine import SynthesisEngine
from reasoning_forge.problem_generator import ProblemGenerator

__all__ = [
    "ForgeEngine",
    "ReasoningAgent",
    "NewtonAgent",
    "QuantumAgent",
    "EthicsAgent",
    "PhilosophyAgent",
    "DaVinciAgent",
    "EmpathyAgent",
    "CriticAgent",
    "SynthesisEngine",
    "ProblemGenerator",
]

__version__ = "1.0.0"
