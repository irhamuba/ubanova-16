"""Web GUI server for UbaNova pixel art generation.
All frontend (HTML, CSS, JS) and server logic reside inside the web/ directory.
"""
import argparse
import json
import logging
import mimetypes
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer import SpriteGenerator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_server")

WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class SpriteHTTPRequestHandler(BaseHTTPRequestHandler):
    generator: SpriteGenerator = None

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            info = {
                "status": "ready",
                "device": self.generator.device,
                "model_path": self.generator.model_path,
                "palette_path": self.generator.palette_path,
            }
            body = json.dumps(info).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path in ("/", "/index.html"):
            target_file = os.path.join(WEB_DIR, "index.html")
        else:
            rel_path = path.lstrip("/")
            target_file = os.path.abspath(os.path.join(WEB_DIR, rel_path))
            if not target_file.startswith(WEB_DIR + os.sep):
                self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
                return

        if os.path.isfile(target_file):
            content_type, _ = mimetypes.guess_type(target_file)
            content_type = content_type or "application/octet-stream"
            with open(target_file, "rb") as f:
                content = f.read()

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/generate":
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._send_json_error(HTTPStatus.BAD_REQUEST, "Empty request body")
                return

            try:
                body = self.rfile.read(length).decode("utf-8")
                params = json.loads(body)
            except Exception as e:
                self._send_json_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {e}")
                return

            prompt = params.get("prompt", "").strip()
            if not prompt:
                self._send_json_error(HTTPStatus.BAD_REQUEST, "Prompt is required")
                return

            count = min(max(int(params.get("n", 4)), 1), 16)
            cfg = float(params.get("cfg", 3.0))
            temperature = float(params.get("temperature", 1.0))
            top_p = float(params.get("top_p", 0.9))
            seed = params.get("seed")
            if seed is not None:
                seed = int(seed)

            logger.info("Generating: '%s' (n=%d, cfg=%.1f, temp=%.1f, seed=%s)", prompt, count, cfg, temperature, seed)
            start_time = time.time()

            try:
                images = self.generator.generate(
                    prompt=prompt,
                    n=count,
                    cfg=cfg,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
                elapsed = time.time() - start_time

                grid = self.generator.create_grid(images, scale=8)
                grid_uri = self.generator.to_data_uri(grid)
                image_uris = [self.generator.to_data_uri(img, scale=8) for img in images]

                response_data = {
                    "prompt": prompt,
                    "count": count,
                    "cfg": cfg,
                    "temperature": temperature,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "grid_image": grid_uri,
                    "images": image_uris,
                }
                body_bytes = json.dumps(response_data).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
            except Exception as e:
                logger.exception("Error generating sprites")
                self._send_json_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Generation failed: {e}")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Path not found")

    def _send_json_error(self, status: int, message: str):
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="UbaNova Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument("--palette", default=None, help="Path to palette.npy")
    parser.add_argument("--device", default=None, help="Inference device: cpu or cuda")
    args = parser.parse_args()

    logger.info("Initializing SpriteGenerator...")
    generator = SpriteGenerator(
        model_path=args.model,
        palette_path=args.palette,
        device=args.device,
    )
    logger.info("Generator ready on %s (model: %s)", generator.device, generator.model_path)

    SpriteHTTPRequestHandler.generator = generator
    server_address = (args.host, args.port)
    server = ThreadingHTTPServer(server_address, SpriteHTTPRequestHandler)

    url = f"http://{args.host}:{args.port}"
    logger.info("Serving UbaNova GUI at %s", url)
    print(f"Server running at {url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()
