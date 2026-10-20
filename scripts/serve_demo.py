"""Tiny local web demo: chat with one ArguBot system and watch its claim log and retrieved passages.

    python scripts/serve_demo.py --dummy        # fake components, no downloads, checks the page works
    python scripts/serve_demo.py                # real adapter, passages and NLI model (needs a GPU or patience)

Then open http://127.0.0.1:8000. On the GPU cluster add --host 0.0.0.0 and tunnel the port (docs/RUN_ON_VIOLET.md). It uses only the standard library, so nothing extra to install.
The chat goes through ArguBotPipeline.step, the same code path the evaluated systems use.
"""
import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from argubot.config import load_config
from argubot.schemas import Turn
from argubot.systems import ArguBotPipeline, build_systems
from argubot.systems.components import load_components

PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


class Demo:
    def __init__(self, cfg, dummy: bool) -> None:
        self.cfg = cfg
        generator, retriever, nli = load_components(cfg, dummy=dummy)
        specs = build_systems(cfg["decoding"]["n_candidates_rerank"])
        self.pipelines = {
            name: ArguBotPipeline(spec, generator, cfg, retriever=retriever, nli=nli)
            for name, spec in specs.items()
        }
        self.sessions = {}  # session id -> (system name, history, tracker, turn count)
        self.dummy = dummy

    def new_session(self, system: str) -> str:
        sid = uuid.uuid4().hex[:8]
        self.sessions[sid] = {"system": system, "history": [], "tracker": self.pipelines[system].tracker_factory(), "n": 0}
        return sid

    def chat(self, sid: str, text: str) -> dict:
        s = self.sessions[sid]
        record = self.pipelines[s["system"]].step(s["history"], s["tracker"], Turn("user", text), sid, s["n"])
        s["n"] += 1
        return {
            "reply": record.bot_text,
            "claims_shown_to_bot": record.claims_in_prompt,
            "claim_log_now": list(s["tracker"].claims),
            "passages": record.passages_in_prompt,
        }


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self._json(200, {"systems": list(demo.pipelines), "dummy": demo.dummy})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            try:
                if self.path == "/api/new":
                    system = data.get("system", "C1")
                    if system not in demo.pipelines:
                        return self._json(400, {"error": f"unknown system {system}"})
                    return self._json(200, {"session": demo.new_session(system)})
                if self.path == "/api/chat":
                    text = (data.get("message") or "").strip()
                    if not text or data.get("session") not in demo.sessions:
                        return self._json(400, {"error": "missing message or unknown session"})
                    return self._json(200, demo.chat(data["session"], text))
            except Exception as e:  # keep the page alive and show the problem
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dummy", action="store_true", help="fake components, replies are canned")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach it through an SSH tunnel, e.g. on the GPU cluster")
    args = ap.parse_args()

    demo = Demo(load_config(args.config), args.dummy)
    print(f"ArguBot demo on http://{args.host}:{args.port}" + ("  (DUMMY components)" if args.dummy else ""))
    ThreadingHTTPServer((args.host, args.port), make_handler(demo)).serve_forever()


if __name__ == "__main__":
    main()
