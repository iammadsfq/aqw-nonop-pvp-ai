#!/usr/bin/env python3
"""
Lightweight embedded HTTP Telemetry Server for AQW 1v1 PvP Reinforcement Learning.
Serves live match replays and training progress curves to the web visualizer on port 8088.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

PORT = 8088
HTML_PATH = Path(__file__).resolve().parent / "visualizer.html"
DATA_PATH = Path(__file__).resolve().parent / "telemetry.json"

# Global thread-safe telemetry store
TELEMETRY_LOCK = threading.Lock()
TELEMETRY_DATA: dict[str, Any] = {
    "status": "idle", # "idle", "training", "finished"
    "current_update": 0,
    "total_updates": 60,
    "current_step": 0,
    "total_steps": 122880,
    "win_rate": 0.0,
    "loss_rate": 1.0,
    "avg_win_hp": 0.0,
    "history": [],
    "matches": {}, # update -> match_data
    "latest_match": None,
}


def save_telemetry_to_disk():
    try:
        tmp_path = DATA_PATH.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(TELEMETRY_DATA, f)
        tmp_path.replace(DATA_PATH)
    except Exception as e:
        pass


LAST_LOADED_MTIME = 0.0


def load_telemetry_from_disk():
    global LAST_LOADED_MTIME
    if not DATA_PATH.exists():
        return
    try:
        mtime = DATA_PATH.stat().st_mtime
        if mtime <= LAST_LOADED_MTIME:
            return
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            TELEMETRY_DATA.update(data)
            LAST_LOADED_MTIME = mtime
    except Exception:
        pass


def update_training_status(
    update: int,
    total_updates: int,
    step: int,
    total_steps: int,
    metrics: dict[str, float],
    status: str = "training",
):
    with TELEMETRY_LOCK:
        TELEMETRY_DATA["status"] = status
        TELEMETRY_DATA["current_update"] = update
        TELEMETRY_DATA["total_updates"] = total_updates
        TELEMETRY_DATA["current_step"] = step
        TELEMETRY_DATA["total_steps"] = total_steps
        TELEMETRY_DATA["win_rate"] = metrics.get("win_rate", 0.0)
        TELEMETRY_DATA["loss_rate"] = metrics.get("loss_rate", 1.0)
        TELEMETRY_DATA["avg_win_hp"] = metrics.get("avg_win_hp_ratio", 0.0)

        TELEMETRY_DATA["history"].append({
            "update": update,
            "step": step,
            "win_rate": metrics.get("win_rate", 0.0),
            "loss_rate": metrics.get("loss_rate", 1.0),
            "avg_win_hp": metrics.get("avg_win_hp_ratio", 0.0),
        })
        save_telemetry_to_disk()


def record_eval_match(match_data: dict[str, Any]):
    with TELEMETRY_LOCK:
        up = match_data.get("update", 0)
        TELEMETRY_DATA["matches"][up] = match_data
        TELEMETRY_DATA["latest_match"] = match_data
        save_telemetry_to_disk()


class TelemetryHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed_path = self.path.split("?")[0]
        with TELEMETRY_LOCK:
            load_telemetry_from_disk()

        if parsed_path == "/api/status":
            with TELEMETRY_LOCK:
                summary = {
                    "status": TELEMETRY_DATA["status"],
                    "current_update": TELEMETRY_DATA["current_update"],
                    "total_updates": TELEMETRY_DATA["total_updates"],
                    "current_step": TELEMETRY_DATA["current_step"],
                    "total_steps": TELEMETRY_DATA["total_steps"],
                    "win_rate": TELEMETRY_DATA["win_rate"],
                    "loss_rate": TELEMETRY_DATA["loss_rate"],
                    "avg_win_hp": TELEMETRY_DATA["avg_win_hp"],
                    "history": TELEMETRY_DATA["history"],
                    "has_latest_match": TELEMETRY_DATA["latest_match"] is not None,
                    "available_matches": sorted(list(TELEMETRY_DATA["matches"].keys())),
                }
            payload = json.dumps(summary).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed_path == "/api/latest_match":
            with TELEMETRY_LOCK:
                match = TELEMETRY_DATA["latest_match"]
            if match is None:
                payload = json.dumps({"error": "No matches recorded yet"}).encode("utf-8")
                self.send_response(404)
            else:
                payload = json.dumps(match).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed_path.startswith("/api/match/"):
            try:
                up_id = int(parsed_path.split("/")[-1])
                with TELEMETRY_LOCK:
                    match = TELEMETRY_DATA["matches"].get(up_id)
                if match:
                    payload = json.dumps(match).encode("utf-8")
                    self.send_response(200)
                else:
                    payload = json.dumps({"error": f"Match update {up_id} not found"}).encode("utf-8")
                    self.send_response(404)
            except Exception as e:
                payload = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # Serve visualizer HTML
        if HTML_PATH.exists():
            html_bytes = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence per-request terminal logging to prevent terminal clutter
        pass


def start_visualizer_server(port: int = PORT) -> tuple[HTTPServer | None, threading.Thread | None]:
    load_telemetry_from_disk()
    try:
        server = HTTPServer(("0.0.0.0", port), TelemetryHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[Visualizer Server] Live PvP Arena Dashboard available at http://localhost:{port}")
        return server, thread
    except OSError:
        print(f"[Visualizer Server] Server already active on port {port}")
        return None, None


if __name__ == "__main__":
    load_telemetry_from_disk()
    print(f"[Visualizer Server] Starting standalone server on http://localhost:{PORT}")
    server = HTTPServer(("0.0.0.0", PORT), TelemetryHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Visualizer Server] Stopped.")
