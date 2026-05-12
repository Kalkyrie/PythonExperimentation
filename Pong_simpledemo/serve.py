#!/usr/bin/env python3
"""Serves pong.html on localhost:8080. Open http://localhost:8080/pong.html in your browser."""
import http.server, os

PORT = 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))

handler = http.server.SimpleHTTPRequestHandler
with http.server.HTTPServer(("", PORT), handler) as httpd:
    print(f"Serving at http://localhost:{PORT}/pong.html  (Ctrl+C to stop)")
    httpd.serve_forever()
