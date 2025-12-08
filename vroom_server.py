#!/usr/bin/env python3
"""
Simple HTTP server wrapper for VROOM CLI tool.
Provides HTTP API compatible with vroom-express.
"""

import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

VROOM_BINARY = "/usr/local/bin/vroom"
# Extract host and port from OSRM_ADDRESS
OSRM_ADDRESS = os.getenv("OSRM_ADDRESS", "http://localhost:5001")
# Parse OSRM address for vroom CLI (needs host:port format)
from urllib.parse import urlparse
osrm_parsed = urlparse(OSRM_ADDRESS)
OSRM_HOST = osrm_parsed.hostname or "localhost"
OSRM_PORT = osrm_parsed.port or (5001 if osrm_parsed.hostname else 5001)
PORT = int(os.getenv("VROOM_PORT", "3000"))


class VroomHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests to solve VRP problems."""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse JSON input
            data = json.loads(post_data.decode('utf-8'))
            
            # Convert to VROOM CLI format
            # VROOM expects: vehicles, jobs, shipments (optional)
            vroom_input = {
                "vehicles": data.get("vehicles", []),
                "jobs": data.get("jobs", []),
            }
            
            if "shipments" in data:
                vroom_input["shipments"] = data["shipments"]
            
            # Run VROOM
            cmd = [
                VROOM_BINARY,
                "-a", OSRM_HOST,
                "-p", str(OSRM_PORT),
                "-r", "osrm",
                "-g"  # Add geometry
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=json.dumps(vroom_input))
            
            if process.returncode != 0:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "code": process.returncode,
                    "error": stderr or "VROOM execution failed"
                }
                self.wfile.write(json.dumps(error_response).encode())
                return
            
            # Parse VROOM output
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "code": 1,
                    "error": "Failed to parse VROOM output",
                    "output": stdout[:500]
                }
                self.wfile.write(json.dumps(error_response).encode())
                return
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except json.JSONDecodeError as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"code": 2, "error": f"Invalid JSON: {str(e)}"}
            self.wfile.write(json.dumps(error_response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"code": 1, "error": str(e)}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_GET(self):
        """Handle GET requests - health check."""
        if self.path == '/' or self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                "status": "ok",
                "service": "vroom",
                "osrm_address": OSRM_ADDRESS
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    """Start the HTTP server."""
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, VroomHandler)
    print(f"VROOM HTTP server starting on port {PORT}")
    print(f"OSRM address: {OSRM_ADDRESS}")
    print(f"Access at: http://localhost:{PORT}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == '__main__':
    main()

