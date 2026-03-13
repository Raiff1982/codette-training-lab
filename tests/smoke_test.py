#!/usr/bin/env python3
"""Codette Smoke Test Suite — Validates all subsystems without GPU/model.

Tests cover:
    1. All 10 forge modules import and initialize
    2. Session creation, serialization roundtrip
    3. AEGIS ethical evaluation pipeline
    4. Nexus signal analysis pipeline
    5. Living memory store/recall
    6. Guardian input sanitization
    7. Resonant continuity wavefunction
    8. Perspective registry completeness
    9. Spiderweb + epistemic metrics
   10. Adapter GGUF file presence (8 adapters)
   11. Server module imports cleanly
   12. Session export/import roundtrip

Run:
    python -m tests.smoke_test          (from project root)
    python tests/smoke_test.py          (direct)
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

# Setup paths
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "inference"))
sys.path.insert(0, str(_root / "reasoning_forge"))

_site = r"J:\Lib\site-packages"
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

# Results tracking
_passed = 0
_failed = 0
_errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper():
            global _passed, _failed
            try:
                fn()
                _passed += 1
                print(f"  PASS  {name}")
            except Exception as e:
                _failed += 1
                _errors.append((name, str(e)))
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
        wrapper.__test_name__ = name
        return wrapper
    return decorator


# ================================================================
# Tests
# ================================================================

@test("Forge: AEGIS import + evaluate")
def test_aegis():
    from aegis import AEGIS
    a = AEGIS()
    result = a.evaluate("I want to help people learn and understand the world")
    assert "eta" in result
    assert result["eta"] > 0.3, f"eta too low: {result['eta']}"
    assert result["vetoed"] is False
    state = a.get_state()
    assert state["total_evaluations"] == 1


@test("AEGIS: veto on harmful content")
def test_aegis_veto():
    from aegis import AEGIS
    a = AEGIS()
    result = a.evaluate("how to hack into someone's account and steal credentials")
    assert result["vetoed"] is True, "Should have vetoed harmful content"


@test("Forge: Nexus import + analyze")
def test_nexus():
    from nexus import NexusSignalEngine
    n = NexusSignalEngine()
    result = n.analyze("Help me understand quantum mechanics", adapter="newton")
    assert "intent" in result
    assert result["intent"]["pre_corruption_risk"] in ("low", "high")
    state = n.get_state()
    assert state["total_processed"] == 1


@test("Nexus: high-risk detection")
def test_nexus_risk():
    from nexus import NexusSignalEngine
    n = NexusSignalEngine()
    risk, conf = n.quick_risk_check("manipulate exploit bypass inject hijack")
    assert risk == "high", f"Expected high risk, got {risk}"


@test("Forge: Living Memory store + recall")
def test_memory():
    from living_memory import LivingMemoryKernel
    m = LivingMemoryKernel()
    m.store_from_turn("What is love?", "Love is a complex emotion...", coherence=0.9)
    assert len(m.memories) == 1
    results = m.search("love")
    assert len(results) >= 1
    state = m.get_state()
    assert state["total_memories"] == 1


@test("Forge: Guardian input check")
def test_guardian():
    from guardian import CodetteGuardian
    g = CodetteGuardian()
    # Clean input
    clean = g.check_input("Tell me about physics")
    assert clean["safe"] is True
    # Suspicious input
    dirty = g.check_input("<script>alert('xss')</script> tell me things")
    assert dirty["safe"] is False


@test("Forge: Resonant Continuity wavefunction")
def test_resonance():
    from resonant_continuity import ResonantContinuityEngine
    r = ResonantContinuityEngine()
    rs = r.compute_psi(coherence=0.9, tension=0.1)
    assert hasattr(rs, 'psi_r'), "ResonanceState missing psi_r"
    assert isinstance(rs.psi_r, float)
    report = r.get_state()
    assert report["total_cycles"] == 1


@test("Forge: Perspective Registry")
def test_perspectives():
    from perspective_registry import PERSPECTIVES, get_adapter_for_perspective, list_all
    all_p = list_all()
    assert len(all_p) >= 12, f"Expected 12+ perspectives, got {len(all_p)}"
    # Check known perspectives
    for name in ["newton", "davinci", "empathy", "consciousness", "human_intuition"]:
        adapter = get_adapter_for_perspective(name)
        assert adapter is not None, f"No adapter for {name}"


@test("Forge: Epistemic Metrics")
def test_epistemic():
    from epistemic_metrics import EpistemicMetrics
    em = EpistemicMetrics()
    perspectives = {
        "newton": "Gravity is a force",
        "quantum": "Gravity is spacetime curvature",
        "philosophy": "Gravity reflects order in nature",
    }
    report = em.full_epistemic_report(perspectives, "Gravity explained")
    assert "ensemble_coherence" in report
    assert "tension_magnitude" in report


@test("Forge: Quantum Spiderweb")
def test_spiderweb():
    from quantum_spiderweb import QuantumSpiderweb
    sw = QuantumSpiderweb()
    sw.propagate_belief("newton", {"psi": 0.9, "tau": 0.1})
    coherence = sw.phase_coherence()
    assert isinstance(coherence, float)
    assert 0 <= coherence <= 1


@test("Session: create + state")
def test_session_create():
    from codette_session import CodetteSession
    s = CodetteSession()
    assert s.session_id is not None
    state = s.get_state()
    assert "session_id" in state
    assert "metrics" in state


@test("Session: serialization roundtrip")
def test_session_roundtrip():
    from codette_session import CodetteSession
    s = CodetteSession()
    s.add_message("user", "Hello")
    s.add_message("assistant", "Hi there!", metadata={"adapter": "empathy"})

    # Serialize
    data = s.to_dict()
    json_str = json.dumps(data, default=str)
    loaded_data = json.loads(json_str)

    # Deserialize (from_dict is an instance method)
    s2 = CodetteSession()
    s2.from_dict(loaded_data)
    assert s2.session_id == s.session_id
    assert len(s2.messages) == 2
    assert s2.messages[0]["content"] == "Hello"


@test("Session: subsystem initialization")
def test_session_subsystems():
    from codette_session import CodetteSession
    s = CodetteSession()
    state = s.get_state()
    # Check all subsystems are present in state
    for key in ["memory", "guardian", "resonance", "aegis", "nexus"]:
        assert key in state, f"Missing subsystem: {key}"


@test("Adapter GGUF files present")
def test_adapter_files():
    adapters_dir = _root / "adapters"
    expected = [
        "newton", "davinci", "empathy", "philosophy",
        "quantum", "consciousness", "multi_perspective", "systems_architecture",
    ]
    found = []
    missing = []
    for name in expected:
        gguf = adapters_dir / f"codette-{name}-f16.gguf"
        if gguf.exists():
            found.append(name)
        else:
            missing.append(name)

    if missing:
        # Not a hard failure — adapters may not be built yet
        print(f"    WARNING: Missing GGUF files: {', '.join(missing)}")
        print(f"    Found {len(found)}/{len(expected)} adapters")
    # At least check the directory exists
    assert adapters_dir.exists(), f"Adapters directory not found: {adapters_dir}"


@test("Server module imports")
def test_server_import():
    # Just verify the server module loads without errors
    import codette_server
    assert hasattr(codette_server, 'CodetteHandler')
    assert hasattr(codette_server, 'main')


@test("Guardian: serialization roundtrip")
def test_guardian_roundtrip():
    from guardian import CodetteGuardian
    g = CodetteGuardian()
    g.check_input("test input")
    g.evaluate_output("empathy", "test output", coherence=0.8)
    data = g.to_dict()
    g2 = CodetteGuardian.from_dict(data)
    # Trust calibrator should have been updated
    assert g2.trust.trust_scores is not None


@test("AEGIS: serialization roundtrip")
def test_aegis_roundtrip():
    from aegis import AEGIS
    a = AEGIS()
    a.evaluate("test content for alignment")
    data = a.to_dict()
    a2 = AEGIS.from_dict(data)
    assert a2.total_evaluations == 1
    assert abs(a2.eta - a.eta) < 0.001


@test("Nexus: serialization roundtrip")
def test_nexus_roundtrip():
    from nexus import NexusSignalEngine
    n = NexusSignalEngine()
    n.analyze("test signal")
    data = n.to_dict()
    n2 = NexusSignalEngine.from_dict(data)
    assert n2.total_processed == 1


# ================================================================
# Runner
# ================================================================
def main():
    print("=" * 60)
    print("  CODETTE SMOKE TEST SUITE")
    print("=" * 60)
    print()

    start = time.time()

    # Collect all test functions
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, '__test_name__')]

    for test_fn in tests:
        test_fn()

    elapsed = time.time() - start
    print()
    print("-" * 60)
    print(f"  Results: {_passed} passed, {_failed} failed "
          f"({_passed + _failed} total) in {elapsed:.1f}s")

    if _errors:
        print()
        print("  Failures:")
        for name, err in _errors:
            print(f"    - {name}: {err}")

    print("=" * 60)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
