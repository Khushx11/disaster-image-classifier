"""Serve the capstone evaluation and optional verified model comparison locally."""

import argparse
import base64
import binascii
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "portfolio"
FILES = {
    "/": (ASSETS / "index.html", "text/html; charset=utf-8"),
    "/style.css": (ASSETS / "style.css", "text/css; charset=utf-8"),
    "/app.js": (ASSETS / "app.js", "text/javascript; charset=utf-8"),
}
MAX_IMAGE = 8 * 1024 * 1024
MAX_REQUEST = (MAX_IMAGE + 2) // 3 * 4 + 8192
MODELS = {"late_fusion", "image", "concat_drop", "gated", "text"}


def study():
    with (RESULTS / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = [
            {"model": row["model"], "condition": row["condition"], "f1": float(row["macro_f1"])}
            for row in csv.DictReader(stream) if row["model"] in MODELS
        ]
    audit = json.loads((RESULTS / "data_audit.json").read_text())
    return {
        "rows": rows,
        "test_size": sum(group["test"] for group in audit["splits"].values()),
        "label_disagreements": audit["mirror_label_disagreements"],
        "test": json.loads((RESULTS / "statistics.json").read_text())["clean"],
    }


def verify_inputs(evidence, manifest=RESULTS / "demo_inputs.sha256"):
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        with (evidence / name).open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Evidence differs from the published capstone: {name}")


def serve(port, report, predictor=None, sample=None):
    inference_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, result):
            self.send(status, json.dumps(result, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            if self.path == "/api/study":
                self.json(200, report)
            elif self.path == "/example.jpg" and sample:
                self.send(200, sample, "image/jpeg")
            elif self.path in FILES:
                path, content_type = FILES[self.path]
                self.send(200, path.read_bytes(), content_type)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/predict" or predictor is None:
                self.send_error(404)
                return
            origin = self.headers.get("Origin")
            allowed = {f"http://{host}:{self.server.server_port}" for host in ("127.0.0.1", "localhost")}
            if origin and origin not in allowed:
                self.send_error(403)
                return
            from PIL import Image

            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST:
                    self.json(413, {"error": "Image request exceeds the size limit or is empty."})
                    return
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an image and optional caption.")
                caption = payload.get("text", "")
                if not isinstance(caption, str) or len(caption) > 1000:
                    raise ValueError("Use a caption of at most 1,000 characters.")
                if payload.get("sample") is True and sample:
                    data = sample
                else:
                    encoded = payload.get("image")
                    if not isinstance(encoded, str):
                        raise ValueError("Choose an image before running inference.")
                    data = base64.b64decode(encoded, validate=True)
                if not 0 < len(data) <= MAX_IMAGE:
                    raise ValueError("Choose an image no larger than 8 MiB.")
                with inference_lock:
                    result = predictor.run(BytesIO(data), caption)
                self.json(200, result)
            except (UnicodeError, binascii.Error, OSError, ValueError, Image.DecompressionBombError):
                # Invalid uploads can include local image metadata. Keep the response generic.
                self.json(400, {"error": "Use a valid JPEG, PNG or WebP up to 8 MiB and 20 megapixels, with a caption up to 1,000 characters."})

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Open http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--evidence", type=Path, help="Trusted local original COMP6001 project folder")
    args = parser.parse_args()
    report = study()
    predictor = sample = None
    if args.evidence is not None:
        try:
            verify_inputs(args.evidence)
            from src.predict import Predictor
            predictor = Predictor(args.evidence, compare=True)
            image_path = args.evidence / "data/example.jpg"
            caption_path = args.evidence / "data/example.json"
            if image_path.is_file() and caption_path.is_file():
                sample = image_path.read_bytes()
                report["example_caption"] = json.loads(caption_path.read_text())["text"]
        except (OSError, ValueError, ImportError, KeyError) as error:
            parser.exit(1, f"Unable to load the original capstone evidence: {error}\n")
    report["live"] = predictor is not None
    serve(args.port, report, predictor, sample)


if __name__ == "__main__":
    main()
