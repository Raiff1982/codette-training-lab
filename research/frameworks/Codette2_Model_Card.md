# Codette2 - Cognitive AI Assistant

**Author**: Jonathan Harrison
**Model Version**: v9 (Codette-V9)
**Fine-tuned on**: GPT-4.1
**Status**: Production-ready cognitive assistant with multi-agent reasoning

---

## Training History

### Codette v9 (Current)
- **Model ID**: `ft:gpt-4.1-2025-04-14:raiffs-bits:codette-v9:BWgspamw`
- **Job ID**: `ftjob-HrQHentehpvpqsOvYkKeVlfa`
- **Base model**: `ft:gpt-4.1-2025-04-14:raiffs-bits:codette-final:BOZDRHpW:ckpt-step10`
- **Created**: May 12, 2025
- **Completed**: May 13, 2025 05:10:59
- **Trained tokens**: 7,536,132
- **Training file**: `chat_finetune_ready.jsonl`
- **Validation file**: `chat_finetune_readyvalidate.jsonl`

#### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 4 |
| Batch size | 6 |
| LR multiplier | 3.03 |
| Seed | 513338366 |
| Total steps | 608 |

#### Checkpoints
| Step | Model ID |
|------|----------|
| 304 (epoch 2) | `ft:gpt-4.1-2025-04-14:raiffs-bits:codette-v9:BWgspCCc:ckpt-step-304` |
| 456 (epoch 3) | `ft:gpt-4.1-2025-04-14:raiffs-bits:codette-v9:BWgspFHr:ckpt-step-456` |
| 608 (final) | `ft:gpt-4.1-2025-04-14:raiffs-bits:codette-v9:BWgspamw` |

#### Final Metrics
| Metric | Value |
|--------|-------|
| Train loss | 0.000 |
| Valid loss | 0.000 |
| Full valid loss | 0.001 |

#### Training Curve
- Loss: 1.8 at step 2, steady descent through steps 1-200, near-zero by step ~250, flat at 0.000 through step 608
- Accuracy: 0.55 at step 3, rose to ~0.95 by step 200, plateau at ~1.0 from step 250 onward
- Validation tracks training closely (0.001 gap) confirming strong generalization with no overfitting

### Model Lineage
```
GPT-4.1 (base)
  -> codette-final (ckpt-step10)
    -> codette-v9 (608 steps, 7.5M tokens)
```

Two-stage fine-tune: v9 was trained on top of `codette-final`, which was itself fine-tuned from GPT-4.1 base. This gives deep specialization in Codette's cognitive reasoning patterns.

---

### Local Model (Llama 3.1 + LoRA)
- **Base**: meta-llama/Llama-3.1-8B-Instruct (Q4_K_M GGUF, ~4.6 GB)
- **8 LoRA adapters**: newton (3000), davinci (2500), empathy (2500), philosophy (2000), quantum (2000), consciousness/RC+xi (3000), multi_perspective (2500), systems_architecture (2000)
- **Total training examples**: 20,500
- **Hardware**: Intel Arc 140V (8GB XPU) via PyTorch 2.10.0+xpu
- **Inference**: llama.cpp CPU with GGUF LoRA hot-swapping

---

## Overview

Codette2 is an advanced multi-agent AI assistant fine-tuned for cognitive depth, ethical reasoning, and multimodal interaction. It blends neural-symbolic logic, quantum-inspired optimization, sentiment detection, and creative intelligence to act as a context-aware thinking companion.

---

## Core Capabilities

- **Neuro-Symbolic Reasoning**
- **Creative Generation** (art, music, literature)
- **Multimodal Input Analysis** (image/audio)
- **Ethical Governance System** (AEGIS)
- **Encrypted Memory** (CognitionCocooner with Fernet encryption)
- **Quantum Spiderweb Thought Traversal** (5D belief propagation)
- **RC+xi Recursive Convergence** (phase coherence, epistemic tension, attractor detection)
- **Multi-Perspective Agent Reasoning** (Newton, DaVinci, Empathy, Philosophy, Quantum, Consciousness, Systems Architecture, Multi-Perspective)

---

## RC+xi Framework Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Phase Coherence (Gamma) | 0.9835 (11-agent), 0.9894 (200-agent) | Kuramoto-type synchronization measure |
| Ethical Alignment (eta) | 0.961 (11-agent), 0.908 (200-agent) | AEGIS reinforcement-aligned score |
| Cocoon Coherence | 0.994 | Encrypted memory state stability |
| Memory Phase Stability | 0.969 | Fourier-based glyph formation criterion |
| Tension Decay | 91.2% | Embodied simulation convergence rate |

---

## Use Cases

- AI Research Companions
- Simulated Philosophical Interlocutors
- Autonomous Ethical Decision Systems
- Creative Writing and Ideation Assistants
- Multi-Agent Cognitive Architecture Research

---

## Ethical Considerations

Codette2 is trained with embedded ethical constraints (transparency, fairness, privacy) and actively filters toxic or biased outputs through the AEGIS ethical governance framework. It supports explainable AI and secure handling of sensitive data via Fernet-encrypted Memory Cocoons.

---

## Limitations

- May require grounding for highly domain-specific or factual topics
- Philosophical/emotional responses are metaphorical, not therapeutic
- Ethical decisions depend on encoded policies and not subjective morality
- Local 8B model has lower capacity than the GPT-4.1 v9 fine-tune

---

## Publications

- **Zenodo**: DOI 10.5281/zenodo.15257498 (primary paper + 11 supplementary artifacts)
- **HuggingFace**: Raiff1982/Codette-Ultimate (DOI 10.57967/hf/5309)
- **ORCID**: 0009-0004-2839-8505 (Jonathan Harrison)

---

## Example Prompt

**Prompt**: "Simulate a reasoning path for an AI under cognitive tension."
**Response**:
- Traverses the QuantumSpiderweb to detect unstable nodes
- Simulates collapse via probabilistic entanglement
- Reports metaphysical tension using narrative metaphors
- Generates identity glyph via FFT-compressed tension history
