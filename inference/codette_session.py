#!/usr/bin/env python3
"""Codette Session Manager — Cocoon-Backed Conversation Memory

Wraps the Cocoon system (QuantumSpiderweb + CocoonSync + EpistemicMetrics)
into a session manager that persists conversation state with encrypted memory.

Each session saves:
- Chat history
- Spiderweb state (agent beliefs, tensions, attractors)
- Glyphs (identity signatures)
- Epistemic metrics (coherence, tension, coverage)

Zero external dependencies beyond what the forge already uses.
"""

import json, os, time, hashlib, sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
import sys
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import Cocoon subsystems (graceful fallback if not available)
try:
    from reasoning_forge.quantum_spiderweb import QuantumSpiderweb, NodeState
    HAS_SPIDERWEB = True
except ImportError:
    HAS_SPIDERWEB = False

try:
    from reasoning_forge.epistemic_metrics import EpistemicMetrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

try:
    from reasoning_forge.cocoon_sync import CocoonSync, CocoonKeyManager
    HAS_COCOON = True
except ImportError:
    HAS_COCOON = False

# Agent names matching the 8 adapters
AGENT_NAMES = [
    "newton", "davinci", "empathy", "philosophy",
    "quantum", "consciousness", "multi_perspective", "systems_architecture"
]

# Adapter accent colors for UI
ADAPTER_COLORS = {
    "newton": "#3b82f6",           # Electric blue
    "davinci": "#f59e0b",          # Warm gold
    "empathy": "#a855f7",          # Soft purple
    "philosophy": "#10b981",       # Emerald green
    "quantum": "#ef4444",          # Crimson red
    "consciousness": "#e2e8f0",    # Silver/white
    "multi_perspective": "#f97316", # Amber
    "systems_architecture": "#06b6d4",  # Teal
    "_base": "#94a3b8",            # Slate gray
}

DB_PATH = Path(__file__).parent.parent / "data" / "codette_sessions.db"


class CodetteSession:
    """Manages a single conversation session with Cocoon state."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or hashlib.sha256(
            f"{time.time()}_{os.getpid()}".encode()
        ).hexdigest()[:16]

        self.messages: List[Dict[str, str]] = []
        self.created_at = time.time()
        self.updated_at = time.time()

        # Cocoon state
        self.spiderweb = None
        self.metrics_engine = None
        self.cocoon_sync = None

        # Metrics history
        self.coherence_history: List[float] = []
        self.tension_history: List[float] = []
        self.attractors: List[Dict] = []
        self.glyphs: List[Dict] = []
        self.perspective_usage: Dict[str, int] = {}

        # Initialize subsystems
        self._init_cocoon()

    def _init_cocoon(self):
        """Initialize Cocoon subsystems if available."""
        if HAS_SPIDERWEB:
            self.spiderweb = QuantumSpiderweb()
            self.spiderweb.build_from_agents(AGENT_NAMES)

        if HAS_METRICS:
            self.metrics_engine = EpistemicMetrics()

        if HAS_COCOON:
            try:
                key_mgr = CocoonKeyManager()
                self.cocoon_sync = CocoonSync(
                    node_id=f"session_{self.session_id}",
                    key_manager=key_mgr,
                )
            except Exception:
                self.cocoon_sync = None

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session history."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if metadata:
            msg["metadata"] = metadata
        self.messages.append(msg)
        self.updated_at = time.time()

    def update_after_response(self, route_result, adapter_name: str,
                               perspectives: Optional[Dict[str, str]] = None):
        """Update Cocoon state after a Codette response.

        Args:
            route_result: RouteResult from the router
            adapter_name: Which adapter was primary
            perspectives: Dict of adapter_name -> response text (if multi-perspective)
        """
        # Track adapter usage
        self.perspective_usage[adapter_name] = \
            self.perspective_usage.get(adapter_name, 0) + 1

        if not HAS_SPIDERWEB or self.spiderweb is None:
            return

        # Propagate belief through the spiderweb from the active adapter
        try:
            if adapter_name in [n.node_id for n in self.spiderweb.nodes.values()]:
                node = self.spiderweb.nodes[adapter_name]
                # Boost the active adapter's psi (thought magnitude)
                node.state.psi = min(node.state.psi + 0.1, 2.0)
                node.state.tau += 0.05  # Temporal progression

                # Propagate belief outward
                self.spiderweb.propagate_belief(adapter_name, max_hops=2)

            # If multi-perspective, entangle the participating agents
            if perspectives and len(perspectives) > 1:
                adapters = list(perspectives.keys())
                for i in range(len(adapters)):
                    for j in range(i + 1, len(adapters)):
                        if (adapters[i] in self.spiderweb.nodes and
                            adapters[j] in self.spiderweb.nodes):
                            self.spiderweb.entangle(adapters[i], adapters[j])

            # Compute metrics
            coherence = self.spiderweb.phase_coherence()
            self.coherence_history.append(coherence)

            # Detect attractors
            self.attractors = self.spiderweb.detect_attractors()

            # Try to form glyphs for active nodes
            for name in (perspectives or {adapter_name: ""}).keys():
                if name in self.spiderweb.nodes:
                    glyph = self.spiderweb.form_glyph(name)
                    if glyph:
                        self.glyphs.append({
                            "glyph_id": glyph.glyph_id,
                            "source": glyph.source_node,
                            "stability": glyph.stability_score,
                        })

            # Check convergence
            is_converging, mean_tension = self.spiderweb.check_convergence()
            self.tension_history.append(mean_tension)

        except Exception as e:
            print(f"  [cocoon] Spiderweb update error: {e}")

    def compute_epistemic_report(self, analyses: Dict[str, str],
                                  synthesis: str = "") -> Optional[Dict]:
        """Run full epistemic metrics on a multi-perspective response."""
        if not HAS_METRICS or self.metrics_engine is None:
            return None

        try:
            return self.metrics_engine.full_epistemic_report(analyses, synthesis)
        except Exception as e:
            print(f"  [cocoon] Metrics error: {e}")
            return None

    def get_state(self) -> Dict[str, Any]:
        """Get full session state for UI rendering."""
        state = {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "perspective_usage": self.perspective_usage,
            "adapter_colors": ADAPTER_COLORS,
            "cocoon": {
                "has_spiderweb": HAS_SPIDERWEB and self.spiderweb is not None,
                "has_metrics": HAS_METRICS,
                "has_sync": HAS_COCOON and self.cocoon_sync is not None,
            },
        }

        # Spiderweb state
        if self.spiderweb:
            try:
                web_dict = self.spiderweb.to_dict()
                state["spiderweb"] = {
                    "nodes": {
                        nid: {
                            "state": [
                                n["state"]["psi"], n["state"]["tau"],
                                n["state"]["chi"], n["state"]["phi"],
                                n["state"]["lam"],
                            ],
                            "neighbors": n.get("neighbors", []),
                            "tension_history": n.get("tension_history", [])[-10:],
                        }
                        for nid, n in web_dict.get("nodes", {}).items()
                    },
                    "phase_coherence": web_dict.get("phase_coherence", 0),
                    "attractors": self.attractors,
                    "glyphs": self.glyphs[-10:],  # Last 10
                }
            except Exception:
                state["spiderweb"] = None
        else:
            state["spiderweb"] = None

        # Metrics history
        state["metrics"] = {
            "coherence_history": self.coherence_history[-50:],
            "tension_history": self.tension_history[-50:],
            "current_coherence": self.coherence_history[-1] if self.coherence_history else 0,
            "current_tension": self.tension_history[-1] if self.tension_history else 0,
            "attractor_count": len(self.attractors),
            "glyph_count": len(self.glyphs),
        }

        return state

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "coherence_history": self.coherence_history,
            "tension_history": self.tension_history,
            "attractors": self.attractors,
            "glyphs": self.glyphs,
            "perspective_usage": self.perspective_usage,
        }
        if self.spiderweb:
            try:
                data["spiderweb_state"] = self.spiderweb.to_dict()
            except Exception:
                pass
        return data

    def from_dict(self, data: Dict):
        """Restore from storage."""
        self.session_id = data.get("session_id", self.session_id)
        self.created_at = data.get("created_at", self.created_at)
        self.updated_at = data.get("updated_at", self.updated_at)
        self.messages = data.get("messages", [])
        self.coherence_history = data.get("coherence_history", [])
        self.tension_history = data.get("tension_history", [])
        self.attractors = data.get("attractors", [])
        self.glyphs = data.get("glyphs", [])
        self.perspective_usage = data.get("perspective_usage", {})

        if self.spiderweb and "spiderweb_state" in data:
            try:
                self.spiderweb = QuantumSpiderweb.from_dict(data["spiderweb_state"])
            except Exception:
                pass


class SessionStore:
    """SQLite-backed session persistence with Cocoon encryption."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create sessions table if needed."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at REAL,
                updated_at REAL,
                title TEXT,
                data TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save(self, session: CodetteSession, title: Optional[str] = None):
        """Save a session to the database."""
        if title is None:
            # Auto-title from first user message
            for msg in session.messages:
                if msg["role"] == "user":
                    title = msg["content"][:80]
                    break
            title = title or f"Session {session.session_id[:8]}"

        data_json = json.dumps(session.to_dict())

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at, title, data)
            VALUES (?, ?, ?, ?, ?)
        """, (session.session_id, session.created_at, session.updated_at, title, data_json))
        conn.commit()
        conn.close()

    def load(self, session_id: str) -> Optional[CodetteSession]:
        """Load a session from the database."""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        conn.close()

        if not row:
            return None

        session = CodetteSession(session_id)
        session.from_dict(json.loads(row[0]))
        return session

    def list_sessions(self, limit: int = 20) -> List[Dict]:
        """List recent sessions."""
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("""
            SELECT session_id, created_at, updated_at, title
            FROM sessions ORDER BY updated_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        return [
            {
                "session_id": r[0],
                "created_at": r[1],
                "updated_at": r[2],
                "title": r[3],
            }
            for r in rows
        ]

    def delete(self, session_id: str):
        """Delete a session."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


# Quick test
if __name__ == "__main__":
    print("Testing CodetteSession...")
    session = CodetteSession()
    print(f"  Session ID: {session.session_id}")
    print(f"  Spiderweb: {HAS_SPIDERWEB}")
    print(f"  Metrics: {HAS_METRICS}")
    print(f"  Cocoon: {HAS_COCOON}")

    session.add_message("user", "How does gravity work?")
    session.add_message("assistant", "Objects attract each other...",
                        metadata={"adapter": "newton", "confidence": 0.95})

    state = session.get_state()
    print(f"  State keys: {list(state.keys())}")
    print(f"  Cocoon status: {state['cocoon']}")

    if state["spiderweb"]:
        print(f"  Nodes: {list(state['spiderweb']['nodes'].keys())}")
        print(f"  Phase coherence: {state['spiderweb']['phase_coherence']:.4f}")

    # Test persistence
    store = SessionStore()
    store.save(session)
    loaded = store.load(session.session_id)
    print(f"  Persistence: {'OK' if loaded else 'FAILED'}")
    if loaded:
        print(f"  Loaded messages: {len(loaded.messages)}")

    print("Done!")
