import argparse
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import webview
from loguru import logger

from cubesight_ml.visualization.backend import SolverAPI

STATIC_DIR = Path(__file__).parent / "static"


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving 3D visualizer static assets and REST API endpoints."""

    api: SolverAPI

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._send_json(self.api.get_state())
        elif self.path == "/api/checkpoints":
            self._send_json(self.api.auto_discover_checkpoints())
        elif self.path == "/api/presets":
            self._send_json(self.api.get_penalties_presets())
        else:
            # Default static file handler
            super().do_GET()

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = (
            self.rfile.read(content_len).decode("utf-8")
            if content_len > 0
            else "{}"
        )
        try:
            body = json.loads(post_body) if post_body else {}
        except json.JSONDecodeError:
            body = {}

        try:
            if self.path == "/api/reset":
                res = self.api.reset()
                self._send_json(res)
            elif self.path == "/api/move":
                move_name = body.get("move", "R")
                res = self.api.apply_move(move_name)
                self._send_json(res)
            elif self.path == "/api/scramble":
                depth = int(body.get("depth", 30))
                custom = body.get("custom_moves")
                res = self.api.scramble(depth=depth, custom_moves=custom)
                self._send_json(res)
            elif self.path == "/api/precondition":
                stage = body.get("stage", "cross")
                res = self.api.precondition(stage)
                self._send_json(res)
            elif self.path == "/api/solve":
                stage = body.get("stage", "all")
                beam_width = int(body.get("beam_width", 100))
                max_steps = int(body.get("max_steps", 40))
                move_penalties = body.get("move_penalties")
                res = self.api.solve(
                    stage=stage,
                    beam_width=beam_width,
                    max_steps=max_steps,
                    move_penalties=move_penalties,
                )
                self._send_json(res)
            elif self.path == "/api/model":
                stage = body.get("stage", "cross")
                checkpoint = body.get("checkpoint")
                res = self.api.set_stage_checkpoint(stage, checkpoint)
                self._send_json(res)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error handling {self.path}: {e}")
            self._send_json({"error": str(e)}, status=500)

    def _send_json(self, data: Any, status: int = 200) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP request logging
        pass


def start_server(
    api: SolverAPI, host: str = "127.0.0.1", port: int = 0
) -> tuple[ThreadingHTTPServer, str]:
    """Starts the visualizer HTTP server on a daemon thread."""
    handler = type("ConfiguredHandler", (VisualizerHandler,), {"api": api})
    server = ThreadingHTTPServer((host, port), handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, url


def launch_visualizer(
    checkpoints: dict[str, str | Path | None] | None = None,
    move_penalties: str | dict | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    browser: bool = False,
    device: str = "auto",
) -> None:
    """
    Launches the 3D Rubik's Cube Solver Studio.
    Opens a native desktop window via pywebview (or standard web browser if --browser is specified).
    """
    logger.info("Initializing CubeSight-ML 3D Solver Studio...")
    api = SolverAPI(
        checkpoints=checkpoints,
        move_penalties=move_penalties,
        device=device,
    )

    server, url = start_server(api, host=host, port=port)
    logger.info(f"Visualizer server running at: {url}")

    if browser:
        logger.info(f"Opening visualizer in web browser: {url}")
        webbrowser.open(url)
        print("\nVisualizer is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down visualizer server...")
            server.shutdown()
        return

    # Try launching native pywebview desktop window
    try:
        logger.info("Launching native 3D visualizer window...")
        webview.create_window(
            title="CubeSight-ML | 3D Rubik's Cube Solver Studio",
            url=url,
            width=1280,
            height=820,
            min_size=(960, 640),
            background_color="#09090b",
        )
        webview.start()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not open native desktop window ({e}). Falling back to web browser..."
        )
        webbrowser.open(url)
        print("\nVisualizer is running in browser. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    finally:
        server.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch 3D Rubik's Cube Solver Studio for interactive Beam Search inference."
    )
    parser.add_argument(
        "--cross_ckpt",
        type=str,
        default=None,
        help="Path to trained Cross model checkpoint (.pt).",
    )
    parser.add_argument(
        "--f2l_ckpt",
        type=str,
        default=None,
        help="Path to trained F2L model checkpoint (.pt).",
    )
    parser.add_argument(
        "--oll_ckpt",
        type=str,
        default=None,
        help="Path to trained OLL model checkpoint (.pt).",
    )
    parser.add_argument(
        "--pll_ckpt",
        type=str,
        default=None,
        help="Path to trained PLL model checkpoint (.pt).",
    )
    parser.add_argument(
        "--move_penalties",
        type=str,
        default="presets/zero_penalties.json",
        help="Path to JSON file or JSON string containing move execution penalties.",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in default web browser instead of native desktop window.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind server to (0 for random free port).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server to.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'cuda', or 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = {
        "cross": args.cross_ckpt,
        "f2l": args.f2l_ckpt,
        "oll": args.oll_ckpt,
        "pll": args.pll_ckpt,
    }
    launch_visualizer(
        checkpoints=checkpoints,
        move_penalties=args.move_penalties,
        host=args.host,
        port=args.port,
        browser=args.browser,
        device=args.device,
    )


if __name__ == "__main__":
    main()
