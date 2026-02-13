#!/usr/bin/env python3
"""
Simple HTTP server to serve trajectory files.
"""

import http.server
import socketserver
import json
import os
from pathlib import Path
from urllib.parse import unquote

PORT = 8000

# Source data directory (relative to this script)
# Should contain stage3_trajectories_* folders and stage2_tasks/
DATA_DIR = "../synthetic_data/demo"

def get_available_models():
    """Scan DATA_DIR for stage3_trajectories_* directories."""
    root_path = Path(__file__).parent / DATA_DIR
    models = []

    if root_path.exists():
        for item in root_path.iterdir():
            if item.is_dir() and item.name.startswith('stage3_trajectories_'):
                # Extract model name from directory name
                model_name = item.name.replace('stage3_trajectories_', '')
                models.append({
                    'id': model_name,
                    'name': model_name,
                    'path': item.name
                })

    return sorted(models, key=lambda x: x['id'])

class TrajectoryHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/models':
            # Return list of available models
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            models = get_available_models()
            self.wfile.write(json.dumps(models).encode())
            return

        elif self.path.startswith('/trajectories'):
            # Return list of all trajectory files for specified model
            # Parse query string for model parameter
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            model_id = query.get('model', [None])[0]

            if not model_id:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'model parameter required'}).encode())
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            base_path = Path(__file__).parent / DATA_DIR / f"stage3_trajectories_{model_id}"
            trajectory_files = []

            if base_path.exists():
                for root, dirs, files in os.walk(base_path):
                    if "trajectory.jsonl" in files:
                        rel_path = os.path.relpath(root, base_path)
                        trajectory_files.append(rel_path)

            self.wfile.write(json.dumps(trajectory_files).encode())
            return

        elif self.path.startswith('/trajectory/'):
            # Return specific trajectory with validation data and component initializations
            # Format: /trajectory/MODEL_ID/path/to/trajectory
            path = unquote(self.path[len('/trajectory/'):])
            parts = path.split('/', 1)

            if len(parts) < 2:
                self.send_response(400)
                self.end_headers()
                return

            model_id = parts[0]
            trajectory_path_rel = parts[1]

            base_path = Path(__file__).parent / DATA_DIR / f"stage3_trajectories_{model_id}"
            trajectory_path = base_path / trajectory_path_rel / "trajectory.jsonl"
            validation_path = base_path / trajectory_path_rel / "validation.json"

            # Extract environment_id (first part of path, e.g., "desert_ecology_station")
            environment_id = trajectory_path_rel.split('/')[0] if '/' in trajectory_path_rel else trajectory_path_rel.split(os.sep)[0]
            stage2_path = Path(__file__).parent / DATA_DIR / "stage2_tasks" / environment_id / "environment.json"

            if trajectory_path.exists():
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Load trajectory
                with open(trajectory_path, 'r') as f:
                    data = json.loads(f.readline().strip())

                # Load validation if exists
                if validation_path.exists():
                    with open(validation_path, 'r') as f:
                        validation = json.load(f)
                        data['validation'] = validation

                # Load component_initializations from environment.json if exists
                if stage2_path.exists():
                    with open(stage2_path, 'r') as f:
                        env_data = json.load(f)
                        data['component_initializations'] = env_data.get('component_initializations', {})

                self.wfile.write(json.dumps(data).encode())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # Serve static files (HTML, etc.)
        return super().do_GET()

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), TrajectoryHandler) as httpd:
        print(f"✓ Server running at http://localhost:{PORT}")
        print(f"✓ Open http://localhost:{PORT}/viewer.html in your browser")
        httpd.serve_forever()
