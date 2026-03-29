"""
HTTP server for the broken-pipeline-env.

Exposes a simple REST API so external agents can interact with the
environment over HTTP.

Endpoints
---------
POST /reset          – Reset the environment for a given task.
POST /step           – Execute one step.
GET  /result         – Retrieve the current episode result.
GET  /health         – Health check.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from env.core import BrokenPipelineEnv
from env.models import Action

_env: Dict[str, BrokenPipelineEnv] = {}


def _read_body(handler: "EnvHandler") -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)
    return json.loads(raw) if raw else {}


def _send_json(handler: "EnvHandler", status: int, data: Dict[str, Any]) -> None:
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class EnvHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            _send_json(self, 200, {"status": "ok"})
        elif path == "/result":
            session_id = self.headers.get("X-Session-Id", "default")
            env = _env.get(session_id)
            if env is None:
                _send_json(self, 400, {"error": "No active session. Call /reset first."})
                return
            try:
                result = env.get_episode_result()
                _send_json(self, 200, result.to_dict())
            except RuntimeError as exc:
                _send_json(self, 400, {"error": str(exc)})
        else:
            _send_json(self, 404, {"error": "Not found."})

    def do_POST(self):
        path = urlparse(self.path).path
        session_id = self.headers.get("X-Session-Id", "default")

        if path == "/reset":
            try:
                body = _read_body(self)
                task_id = body.get("task_id", "task1_audit")
                scenario_path = body.get("scenario_path")
                env = BrokenPipelineEnv(task_id, scenario_path)
                _env[session_id] = env
                obs = env.reset()
                _send_json(self, 200, obs.to_dict())
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                _send_json(self, 400, {"error": str(exc)})

        elif path == "/step":
            env = _env.get(session_id)
            if env is None:
                _send_json(self, 400, {"error": "No active session. Call /reset first."})
                return
            try:
                body = _read_body(self)
                action = Action(
                    action_type=body.get("action_type", ""),
                    payload=body.get("payload", {}),
                )
                result = env.step(action)
                _send_json(self, 200, result.to_dict())
            except RuntimeError as exc:
                _send_json(self, 400, {"error": str(exc)})
            except json.JSONDecodeError as exc:
                _send_json(self, 400, {"error": f"Invalid JSON: {exc}"})

        else:
            _send_json(self, 404, {"error": "Not found."})


def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the HTTP server (blocking)."""
    server = HTTPServer((host, port), EnvHandler)
    print(f"broken-pipeline-env server listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
