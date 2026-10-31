"""
TrafficVision AI - High Performance Multi-Threaded Video & Telemetry Server
Built to stream heavy 1080p highway video and handle multi-node geocoding with zero hang.
"""
import http.server
import socketserver
import webbrowser
import urllib.request
import urllib.parse
import json
import os
import sys

# Configure UTF-8 for Windows PowerShell terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MultiThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

class TacticalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Concise logging to keep server fast
        pass

    def end_headers(self):
        # Enable CORS for all local requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            
            # Proxy Geocoding Endpoint: /api/geocode?q=RoadName
            if parsed_url.path == '/api/geocode':
                query_params = urllib.parse.parse_qs(parsed_url.query)
                q = query_params.get('q', [''])[0].strip()
                
                if not q:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Missing query parameter q'}).encode('utf-8'))
                    return

                results = []
                
                # 1. Try Photon Komoot API (Fast, OpenStreetMap based)
                try:
                    photon_url = f"https://photon.komoot.io/api/?q={urllib.parse.quote(q)}&limit=3"
                    req = urllib.request.Request(photon_url, headers={'User-Agent': 'TrafficVisionAI/2.0'})
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        for feat in data.get('features', []):
                            coords = feat.get('geometry', {}).get('coordinates', [0, 0])
                            props = feat.get('properties', {})
                            name = props.get('name') or props.get('street') or q
                            city = props.get('city') or props.get('state') or props.get('country') or 'Highway Corridor'
                            results.append({
                                'name': name,
                                'city': city,
                                'road': props.get('street') or name,
                                'lat': coords[1],
                                'lng': coords[0],
                                'display_name': f"{name}, {city}"
                            })
                except Exception:
                    pass

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'results': results, 'query': q}).encode('utf-8'))
                return

            # Default static file serving (with broken socket protection)
            return super().do_GET()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # Normal client abort during video streaming seek
            pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

def launch():
    os.chdir(DIRECTORY)
    url = f"http://localhost:{PORT}/dashboard.html"
    print("=" * 65)
    print("[*] TRAFFICVISION AI - MULTI-THREADED STREAMING SERVER")
    print("=" * 65)
    print(f"[*] Serving Directory: {DIRECTORY}")
    print(f"[*] Access URL: {url}")
    print("=" * 65)
    
    with MultiThreadedTCPServer(("", PORT), TacticalHandler) as httpd:
        print(f"[*] Listening on port {PORT} (Multi-threaded & Video Streaming Optimized)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")

if __name__ == "__main__":
    launch()
