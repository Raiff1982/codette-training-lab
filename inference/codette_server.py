#!/usr/bin/env python3
"""Codette Web Server — Zero-Dependency Local AI Chat

Pure Python stdlib HTTP server with SSE streaming.
No Flask, no FastAPI, no npm, no node — just Python.

Usage:
    python codette_server.py                    # Start on port 7860
    python codette_server.py --port 8080        # Custom port
    python codette_server.py --no-browser       # Don't auto-open browser

Architecture:
    - http.server for static files + REST API
    - Server-Sent Events (SSE) for streaming responses
    - Threading for background model loading/inference
    - CodetteOrchestrator for routing + generation
    - CodetteSession for Cocoon-backed memory
"""

import os, sys, json, time, threading, queue, argparse, webbrowser, traceback
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# Auto-configure environment
_site = r"J:\Lib\site-packages"
if _site not in sys.path:
    sys.path.insert(0, _site)
os.environ["PATH"] = r"J:\Lib\site-packages\Library\bin" + os.pathsep + os.environ.get("PATH", "")
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Project imports
_inference_dir = str(Path(__file__).parent)
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)

from codette_session import (
    CodetteSession, SessionStore, ADAPTER_COLORS, AGENT_NAMES
)

# Lazy import orchestrator (heavy — loads llama_cpp)
_orchestrator = None
_orchestrator_lock = threading.Lock()
_orchestrator_status = {"state": "idle", "message": "Not loaded"}
_load_error = None

# Current session
_session: CodetteSession = None
_session_store: SessionStore = None

# Request queue for thread-safe model access
_request_queue = queue.Queue()
_response_queues = {}  # request_id -> queue.Queue


def _get_orchestrator():
    """Lazy-load the orchestrator (first call takes ~60s)."""
    global _orchestrator, _orchestrator_status, _load_error
    if _orchestrator is not None:
        return _orchestrator

    with _orchestrator_lock:
        if _orchestrator is not None:
            return _orchestrator

        _orchestrator_status = {"state": "loading", "message": "Loading Codette model..."}
        print("\n  Loading CodetteOrchestrator...")

        try:
            from codette_orchestrator import CodetteOrchestrator
            _orchestrator = CodetteOrchestrator(verbose=True)
            _orchestrator_status = {
                "state": "ready",
                "message": f"Ready — {len(_orchestrator.available_adapters)} adapters",
                "adapters": _orchestrator.available_adapters,
            }
            print(f"  Orchestrator ready: {_orchestrator.available_adapters}")
            return _orchestrator
        except Exception as e:
            _load_error = str(e)
            _orchestrator_status = {"state": "error", "message": f"Load failed: {e}"}
            print(f"  ERROR loading orchestrator: {e}")
            traceback.print_exc()
            return None


def _worker_thread():
    """Background worker that processes inference requests."""
    while True:
        try:
            request = _request_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if request is None:
            break  # Shutdown signal

        req_id = request["id"]
        response_q = _response_queues.get(req_id)
        if not response_q:
            continue

        try:
            orch = _get_orchestrator()
            if orch is None:
                response_q.put({"error": _load_error or "Model failed to load"})
                continue

            query = request["query"]
            adapter = request.get("adapter")  # None = auto-route
            max_adapters = request.get("max_adapters", 2)

            # Send "thinking" event
            response_q.put({"event": "thinking", "adapter": adapter or "auto"})

            # Route and generate
            force = adapter if adapter and adapter != "auto" else None
            result = orch.route_and_generate(
                query,
                max_adapters=max_adapters,
                strategy="keyword",
                force_adapter=force,
            )

            # Update session
            if _session:
                route = result.get("route")
                adapter_used = result.get("adapter", result.get("adapters", ["base"])[0]
                                          if isinstance(result.get("adapters"), list) else "base")
                perspectives = result.get("perspectives")

                # For single-adapter responses, create a perspectives dict
                # so the spiderweb always updates
                if not perspectives:
                    perspectives = {adapter_used: result["response"]}

                _session.add_message("user", query)
                _session.add_message("assistant", result["response"], metadata={
                    "adapter": adapter_used,
                    "confidence": route.confidence if route else 0,
                    "tokens": result.get("tokens", 0),
                    "time": result.get("time", 0),
                })

                _session.update_after_response(route, adapter_used, perspectives)

                # Compute epistemic metrics (always, not just multi-perspective)
                epistemic = None
                if perspectives and len(perspectives) >= 1:
                    epistemic = _session.compute_epistemic_report(
                        perspectives, result["response"]
                    )

                # Save session periodically
                if _session_store and len(_session.messages) % 4 == 0:
                    try:
                        _session_store.save(_session)
                    except Exception:
                        pass

            # Build response
            response_data = {
                "event": "complete",
                "response": result["response"],
                "adapter": result.get("adapter",
                    result.get("adapters", ["base"])[0] if isinstance(result.get("adapters"), list) else "base"),
                "confidence": route.confidence if route else 0,
                "reasoning": route.reasoning if route else "",
                "tokens": result.get("tokens", 0),
                "time": round(result.get("time", 0), 2),
                "multi_perspective": route.multi_perspective if route else False,
            }

            # Add perspectives if available
            if perspectives:
                response_data["perspectives"] = perspectives

            # Add cocoon state
            if _session:
                response_data["cocoon"] = _session.get_state()

            # Add epistemic report if available
            if epistemic:
                response_data["epistemic"] = epistemic

            # Add tool usage info if any tools were called
            tools_used = result.get("tools_used", [])
            if tools_used:
                response_data["tools_used"] = tools_used

            response_q.put(response_data)

        except Exception as e:
            traceback.print_exc()
            response_q.put({"event": "error", "error": str(e)})


class CodetteHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for Codette API + static files."""

    # Serve static files from inference/static/
    def __init__(self, *args, **kwargs):
        static_dir = str(Path(__file__).parent / "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def log_message(self, format, *args):
        """Quieter logging — skip static file requests."""
        msg = format % args
        if not any(ext in msg for ext in [".css", ".js", ".ico", ".png", ".woff"]):
            print(f"  [{time.strftime('%H:%M:%S')}] {msg}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API routes
        if path == "/api/status":
            self._json_response(_orchestrator_status)
        elif path == "/api/session":
            self._json_response(_session.get_state() if _session else {})
        elif path == "/api/sessions":
            sessions = _session_store.list_sessions() if _session_store else []
            self._json_response({"sessions": sessions})
        elif path == "/api/adapters":
            self._json_response({
                "colors": ADAPTER_COLORS,
                "agents": AGENT_NAMES,
                "available": _orchestrator.available_adapters if _orchestrator else [],
            })
        elif path == "/api/chat":
            # SSE endpoint for streaming
            self._handle_chat_sse(parsed)
        elif path == "/":
            # Serve index.html
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            self._handle_chat_post()
        elif path == "/api/session/new":
            self._handle_new_session()
        elif path == "/api/session/load":
            self._handle_load_session()
        elif path == "/api/session/save":
            self._handle_save_session()
        else:
            self.send_error(404, "Not found")

    def _json_response(self, data, status=200):
        """Send a JSON response."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        """Read and parse JSON POST body."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _handle_chat_post(self):
        """Handle chat request — queue inference, return via SSE or JSON."""
        data = self._read_json_body()
        query = data.get("query", "").strip()
        adapter = data.get("adapter")
        max_adapters = data.get("max_adapters", 2)

        if not query:
            self._json_response({"error": "Empty query"}, 400)
            return

        # Guardian input check
        if _session and _session.guardian:
            check = _session.guardian.check_input(query)
            if not check["safe"]:
                query = check["cleaned_text"]

        # Check if orchestrator is loading
        if _orchestrator_status.get("state") == "loading":
            self._json_response({
                "error": "Model is still loading, please wait...",
                "status": _orchestrator_status,
            }, 503)
            return

        # Queue the request
        req_id = f"{time.time()}_{id(self)}"
        response_q = queue.Queue()
        _response_queues[req_id] = response_q

        _request_queue.put({
            "id": req_id,
            "query": query,
            "adapter": adapter,
            "max_adapters": max_adapters,
        })

        # Wait for response (with timeout)
        try:
            # First wait for thinking event
            thinking = response_q.get(timeout=120)
            if "error" in thinking and thinking.get("event") != "thinking":
                self._json_response(thinking, 500)
                return

            # Wait for complete event (multi-perspective can take 15+ min on CPU)
            result = response_q.get(timeout=1200)  # 20 min max for inference
            self._json_response(result)

        except queue.Empty:
            self._json_response({"error": "Request timed out"}, 504)
        finally:
            _response_queues.pop(req_id, None)

    def _handle_chat_sse(self, parsed):
        """Handle SSE streaming endpoint."""
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        adapter = params.get("adapter", [None])[0]

        if not query:
            self.send_error(400, "Missing query parameter 'q'")
            return

        # Set up SSE headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Queue request
        req_id = f"sse_{time.time()}_{id(self)}"
        response_q = queue.Queue()
        _response_queues[req_id] = response_q

        _request_queue.put({
            "id": req_id,
            "query": query,
            "adapter": adapter,
            "max_adapters": 2,
        })

        try:
            # Stream events
            while True:
                try:
                    event = response_q.get(timeout=300)
                except queue.Empty:
                    self._send_sse("error", {"error": "Timeout"})
                    break

                event_type = event.get("event", "message")
                self._send_sse(event_type, event)

                if event_type in ("complete", "error"):
                    break
        finally:
            _response_queues.pop(req_id, None)

    def _send_sse(self, event_type, data):
        """Send a Server-Sent Event."""
        try:
            payload = f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
        except Exception:
            pass

    def _handle_new_session(self):
        """Create a new session."""
        global _session
        # Save current session first
        if _session and _session_store and _session.messages:
            try:
                _session_store.save(_session)
            except Exception:
                pass

        _session = CodetteSession()
        self._json_response({"session_id": _session.session_id})

    def _handle_load_session(self):
        """Load a previous session."""
        global _session
        data = self._read_json_body()
        session_id = data.get("session_id")

        if not session_id or not _session_store:
            self._json_response({"error": "Invalid session ID"}, 400)
            return

        loaded = _session_store.load(session_id)
        if loaded:
            _session = loaded
            self._json_response({
                "session_id": _session.session_id,
                "messages": _session.messages,
                "state": _session.get_state(),
            })
        else:
            self._json_response({"error": "Session not found"}, 404)

    def _handle_save_session(self):
        """Manually save current session."""
        if _session and _session_store:
            _session_store.save(_session)
            self._json_response({"saved": True, "session_id": _session.session_id})
        else:
            self._json_response({"error": "No active session"}, 400)


def main():
    global _session, _session_store

    parser = argparse.ArgumentParser(description="Codette Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    print("=" * 60)
    print("  CODETTE WEB UI")
    print("=" * 60)

    # Initialize session
    _session_store = SessionStore()
    _session = CodetteSession()
    print(f"  Session: {_session.session_id}")
    print(f"  Cocoon: spiderweb={_session.spiderweb is not None}, "
          f"metrics={_session.metrics_engine is not None}")

    # Start worker thread
    worker = threading.Thread(target=_worker_thread, daemon=True)
    worker.start()

    # Start model loading in background
    threading.Thread(target=_get_orchestrator, daemon=True).start()

    # Start server
    server = HTTPServer(("127.0.0.1", args.port), CodetteHandler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Server: {url}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        # Save session
        if _session and _session_store and _session.messages:
            _session_store.save(_session)
            print(f"  Session saved: {_session.session_id}")
        _request_queue.put(None)  # Shutdown worker
        server.shutdown()
        print("  Goodbye!")


if __name__ == "__main__":
    main()
