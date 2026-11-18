"""Bounded JSON transport for the local processing library."""
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .errors import DomainError

MAX_BODY = 2 * 1024 * 1024

def _constant(value):
    raise ValueError("Non-finite JSON values are not accepted")

def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result

def make_server(host, port, dispatch, routes):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def _reply(self, status, result):
            try:
                data = json.dumps(result, allow_nan=False, ensure_ascii=True).encode()
            except (ValueError, OverflowError):
                status = 500
                data = b'{"error":{"code":"non_finite_result","message":"Numeric result is not representable"}}'
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"status": "ok", "routes": sorted(routes)})
            else:
                self._reply(404, {"error": {"code": "not_found", "message": "Unknown endpoint"}})

        def do_POST(self):
            if self.path not in routes:
                self._reply(404, {"error": {"code": "not_found", "message": "Unknown endpoint"}})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                self._reply(400, {"error": {"code": "invalid_framing", "message": "One Content-Length header is required"}})
                return
            try:
                size = int(lengths[0])
            except ValueError:
                self._reply(400, {"error": {"code": "invalid_length", "message": "Invalid Content-Length"}})
                return
            if size < 0 or size > MAX_BODY:
                # Drain a bounded, slightly oversized body so clients can receive the error.
                if MAX_BODY < size <= 2 * MAX_BODY:
                    remaining = size
                    try:
                        while remaining:
                            chunk = self.rfile.read(min(remaining, 65536))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                    except (TimeoutError, OSError):
                        pass
                self._reply(413, {"error": {"code": "body_limit", "message": "Request body exceeds the supported limit"}})
                return
            if self.headers.get_content_type() != "application/json":
                self._reply(415, {"error": {"code": "content_type", "message": "Use application/json"}})
                return
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete JSON body")
                payload = json.loads(raw, parse_constant=_constant, object_pairs_hook=_object)
            except (ValueError, UnicodeError, RecursionError, TimeoutError, OSError):
                self._reply(400, {"error": {"code": "invalid_json", "message": "Expected one complete, finite JSON value with unique keys"}})
                return
            try:
                result = dispatch(self.path, payload)
            except DomainError as exc:
                self._reply(422, {"error": {"code": exc.code, "message": str(exc)}})
            except (ArithmeticError, RecursionError):
                self._reply(422, {"error": {"code": "numeric_range", "message": "Input exceeds the supported numerical range"}})
            except Exception:
                self.log_error("Unhandled processing error")
                self._reply(500, {"error": {"code": "internal_error", "message": "Processing failed"}})
            else:
                self._reply(200, {"result": result})

        def log_message(self, format, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False
        request_queue_size = 32

    return Server((host, port), Handler)
