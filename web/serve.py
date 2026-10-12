"""Local server for the web demo that tells the browser not to cache anything.

    python3 web/serve.py          (run from the repository root, then open http://localhost:8765/web/)
"""
import http.server
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"http://localhost:{port}/web/")
    http.server.ThreadingHTTPServer(("", port), NoCache).serve_forever()
