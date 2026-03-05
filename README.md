# Codette Adapter Training Lab

Codette is an experimental AI research system for **recursive reasoning, multi-perspective cognition, and ethical AI alignment**.

This repository contains the complete training pipeline for developing Codette LoRA adapters on Llama 3.1 8B.

## Architecture

```
codette-training-lab/
├── dataset_engine/          # Dataset generation pipeline
│   ├── template_registry.py # Rich template pools per adapter
│   ├── answer_generator.py  # Structured educational answer generation
│   ├── dataset_generator.py # Main generator with dedup + validation
│   └── templates/           # JSON template definitions
│
├── reasoning_forge/         # Multi-agent reasoning dataset refinement
│   ├── agents/              # Newton, Quantum, Ethics, Philosophy, DaVinci, Empathy
│   ├── critic_agent.py      # Quality evaluation agent
│   ├── synthesis_engine.py  # Multi-perspective synthesis
│   ├── problem_generator.py # Reasoning problem generation
│   └── forge_engine.py      # Orchestrator
│
├── training/                # LoRA training scripts
│   ├── train_adapter.py     # Single adapter training (4-bit LoRA)
│   ├── train_all_adapters.py# Sequential multi-adapter training
│   ├── merge_adapters.py    # Merge LoRA into base model
│   └── configs/             # Training hyperparameters
│
├── evaluation/              # Benchmarks and quality assurance
│   ├── reasoning_metrics.py # Multi-dimensional scoring
│   ├── benchmark_runner.py  # Automated evaluation
│   ├── dataset_validator.py # Dataset quality checks
│   ├── failure_analyzer.py  # Weakness detection
│   └── prompts/             # Benchmark test sets
│
├── observatory/             # Experiment tracking and monitoring
│   ├── metrics_logger.py    # Training run logging
│   ├── performance_tracker.py # Improvement trends
│   ├── dataset_quality_monitor.py
│   └── dashboard.py         # ASCII status dashboard
│
├── research/                # Source research documents
│   ├── papers/              # Published manuscripts
│   ├── frameworks/          # RC+xi, quantum equations, perspectives
│   └── experiments/         # Cocoon simulations, logs
│
├── datasets/                # Generated training datasets (JSONL)
├── adapters/                # Trained LoRA adapters
├── scripts/                 # Pipeline orchestration
│   ├── run_full_pipeline.py # End-to-end pipeline
│   └── hf_job.yaml          # HuggingFace job config
└── configs/                 # System configuration
    ├── adapter_registry.yaml
    └── pipeline_config.yaml
```

## Adapters

| Adapter | Domain | Target Examples | System Prompt |
|---------|--------|----------------|---------------|
| Newton | Analytical physics reasoning | 3000 | Newtonian analytical precision |
| DaVinci | Creative invention thinking | 2500 | Creative inventiveness |
| Empathy | Emotional understanding | 2500 | Deep empathy and EQ |
| Philosophy | Conceptual reasoning | 2000 | Philosophical depth |
| Quantum | Probabilistic thinking | 2000 | Quantum probabilistic thinking |
| RC+xi | Recursive cognition | 3000 | RC+xi framework reasoning |
| Multi-Perspective | Synthesis across lenses | 2500 | Multi-perspective synthesis |
| Systems | AI architecture | 2000 | System architecture design |

## Training Pipeline

```
research documents
      ↓
dataset extraction (template-based generation)
      ↓
synthetic reasoning expansion (counterexamples, variations)
      ↓
dataset validation (dedup, quality filter)
      ↓
reasoning forge (multi-agent critique + refinement)
      ↓
adapter training (4-bit LoRA on Llama 3.1 8B)
      ↓
benchmark evaluation (multi-dimensional reasoning metrics)
      ↓
observatory logging (track improvement over time)
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate all datasets

```bash
python -m dataset_engine.generate_all
```

### Run full pipeline

```bash
python scripts/run_full_pipeline.py --all
```

### Generate + validate only

```bash
python scripts/run_full_pipeline.py --generate --validate
```

### Train a single adapter

```bash
python -m training.train_adapter \
  --dataset datasets/newton_reasoning.jsonl \
  --adapter-name newton \
  --output-dir adapters/newton
```

### Run benchmarks

```bash
python -m evaluation.benchmark_runner --prompts evaluation/prompts/reasoning_tests.json
```

### View dashboard

```bash
python -m observatory.dashboard
```

## Dataset Format

All datasets use chat-format JSONL:

```json
{
  "messages": [
    {"role": "system", "content": "You are Codette, a recursive multi-perspective reasoning AI."},
    {"role": "user", "content": "Explain the conservation of momentum using a real-world example."},
    {"role": "assistant", "content": "Conservation of momentum states that in a closed system..."}
  ]
}
```

## Reasoning Forge

The Reasoning Forge refines training data through multi-agent debate:

```
concept → problem generator → agent analysis → critic evaluation → synthesis → training example
```

Agents: Newton (physics), Quantum (probability), Ethics (alignment), Philosophy (meaning), DaVinci (creativity), Empathy (emotion)

Each agent analyzes from its perspective, the critic scores quality, and the synthesis engine produces a unified multi-perspective response.

## Base Model

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Method**: QLoRA (4-bit quantization)
- **LoRA config**: rank=16, alpha=32, target=q/k/v/o projections

## Research Background

Codette implements the RC+xi (Recursive Convergence + Epistemic Tension) framework for structured multi-perspective reasoning. The system coordinates 11 reasoning perspectives in parallel before synthesizing a final response.

Key research documents in `research/`:
- RC+xi Framework specification
- Quantum Cosmic Multicore experiment
- Codette Research Equations (8 core quantum mathematics)
- Multi-perspective reasoning architecture

## Requirements

- Python 3.10+
- PyTorch 2.1+
- 16GB+ RAM (CPU training) or GPU with 8GB+ VRAM
- ~1-3 hours per adapter (CPU) or 20-40 min (A10/A100 GPU)

## License

Research project - experimental AI development.
