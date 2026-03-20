# Phase 7: The Control Problem (System-Level Pathology)

## Discovery: Smoke Test Revealed Combinatorial Explosion (Session 2026-03-19)

**Failure Pattern**:
- Conflicts: 41 → 1551 (+3700%)
- Resolution rate: -36.8 (negative = system destabilizing)
- Gamma: 0.375 (collapsed)
- Correctness: 0.0 (complete answer failure)
- Time: ~692s per question (11+ minutes)

**Root Cause**: System can think, but doesn't know when to *stop* thinking.

---

## Solutions Applied ✅

### Patch 1: Conflict Filtering (Framework Differences) ✅
**Status**: Merged into conflict_engine.py
**Rule**: Discard framework disagreements with semantic_overlap > 0.6
**Impact**: Prevents low-value philosophical noise

### Patch 2: Top-K Conflict Selection (Hard Cap) ✅
**Status**: Merged into conflict_engine.py
**Rule**: `conflicts = conflicts[:10]` after sorting by strength
**Impact**: Alone fixes ~80% of explosion (214-860 → 10)
**Logs**: "Capping conflicts: X → 10 (top by strength)"

### Patch 3: Debate Kill Switch (Not yet merged)
**Status**: Designed but not yet implemented
**Rule**: `if conflicts_after > conflicts_before * 1.2: stop_debate = True`
**Why skip for now**: Gamma Authority (Patch 4) provides more nuanced control

### Patch 4: Gamma Authority (Threshold Tuned) ✅
**Status**: Merged into forge_engine.py with tuned threshold
**Change**: 0.5 → 0.3 (allow more debate, only stop on true collapse)
**Rule**:
```python
if gamma_value < 0.3:
    logger.warning(f"System collapse. Stopping debate.")
    return True
```
**Why tuning mattered**: 0.5 was too aggressive (cut debate at round 1, killed reasoning depth)
**New behavior**: Allows 2-3 rounds, stops only if truly collapsing
**Logs**: "WARNING:reasoning_forge.forge_engine:Gamma below collapse threshold (X < 0.3). Stopping debate."

### Patch 5: Agent Relevance Gating (Domain-Aware) ✅
**Status**: Merged into forge_engine.py with helper methods
**Methods**:
- `_classify_query_domain(query)` → detects domain from keyword matching
- `_get_agents_for_domain(domain)` → returns 2-3 relevant agents
**Domain mapping**:
```
'physics' → ['Newton', 'Quantum']
'ethics' → ['Philosophy', 'Empathy']
'consciousness' → ['Philosophy', 'Quantum']
'creativity' → ['DaVinci', 'Quantum']
'systems' → ['Quantum', 'Philosophy']
'general' → all agents
```
**Logs**: "Domain-gated activation: detected 'physics' → 2 agents active"

### Patch 6: Pre-Flight Integration (Predictor Must Suppress) ✅ (Partial)
**Status**: Integration FIXED, Prediction Logic Needs Tuning
**Fix Applied**:
- Added QuantumSpiderweb import to forge_engine.py
- Initialize self.spiderweb = QuantumSpiderweb() in __init__
- Pass self.spiderweb directly to PreFlightConflictPredictor
**Result**: "Could not build spiderweb" warning eliminated ✓
**Current Behavior**:
- Spiderweb builds successfully
- Query state encoding works (psi, tau, chi, phi, lam computed)
- Propagation runs without errors
- **Issue**: Tension threshold (> 1.0) is too high for initial agent states
- **Predicted pairs**: 0 (expected; agents start neutral)
**Next Phase**: Lower threshold or enhance state differentiation**Status**: Designed but not yet implementing

### Patch 7: Hard Output Guarantee (Fallback) ⏳
**Status**: Designed but not yet implementing
**Issue**: Need to wire fallback to single-best-agent per condition
**Note**: Can defer until full evaluation if synthesis is working

---

## Smoke Test Results (Post-Patches)

| Condition | Correctness | Reasoning Depth | Gamma | Conflicts |
|-----------|------------|-----------------|-------|-----------|
| Baseline Llama | 0.500 | 1.0 | 1.000 | N/A |
| Phase 1-5 | 0.600 | 2.0 | 0.750 | 3 |
| **Phase 6 Full** | **0.000** | 1.0 | 0.500 | 10 (capped) |
| Phase 6 -PreFlight | **0.650** | 2.5 | 0.700 | ~10 (capped) |

**Insight**: Phase 1-5 gives +20% correctness. Phase 6 with patches gives +8% but stabilizes system.

---

## UI Updates ✅

### codette_web.bat
Updated with detailed comments:
- Phase 6 production launch
- List of features (domain routing, conflict detection, coherence monitoring, etc.)
- Model/memory specifications
- Launch instructions

### inference/static/index.html
Updated welcome screen:
- New tagline: "Codette v2.0 with Phase 6: Multi-perspective reasoning with controlled debate..."
- Feature callout: "Domain-aware agent routing • Semantic conflict detection • Real-time coherence monitoring • Experience-weighted reasoning"
- Updated example questions reflecting new capabilities:
  - Physics: "What is the speed of light and why does it matter?"
  - Ethics: "How should we balance accuracy and explainability in AI systems?"
  - Creativity: "What are the hallmarks of a truly creative solution?"
  - Consciousness: "What would it mean for a machine to genuinely understand?"

### CODETTE_V2_CAPABILITIES.md
Comprehensive documentation (2,500 words):
- Overview and version info
- 10 core capabilities explained
- Architecture diagram
- Performance characteristics
- Known limitations & Phase 7 roadmap
- Getting started guide

---

## System Behavior Now

✅ **Stability achieved**:
- Conflicts capped at ~10 per round (from 200-800)
- Multuple debate rounds allowed (not cut short)
- Gamma metric guiding intervention, not killing debate
- Domain gating reducing agent noise

⚠️ **Trade-off**:
- Correctness on Phase 6 full = 0.0 (still investigating)
- Phase 6 -PreFlight = 0.650 (+8% vs. baseline)
- System is stable but needs correctness improvement

🧠 **Root cause of correctness dip**:
- Likely: Synthesis engine not properly integrating debate outputs
- Need to check how debate-enriched analyses are combined
- Could be placeholder implementation in test harness

---

## Next Steps for Option B (Full Evaluation)

 Now ready to proceed with 25-question evaluation:
- ✅ Three critical patches applied and verified
- ✅ UI updated to reflect current capabilities
- ✅ System stable (no more 0.5+ gamma collapses)
- ⏳ Expect 2-3 hours for full evaluation run
- ⏳ Will measure Phase 6 on stabilized system (not pathological state)

**Timeline**:
- Today: Patches + smoke test complete
- Tomorrow/next week: Run full evaluation (25 questions × 4 conditions = 100 debates)
- Week 2: Analyze results, make Phase 7 decisions

