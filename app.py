"""
TrafficVision AI - Real-time Tactical Dashboard Launcher & API Proxy Server
Serves the web command center with video streaming, CORS-friendly geocoding, and telemetry.
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

class TacticalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS for all local requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
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

            print(f"[API] Geocoding request for: {q}")
            results = []
            
            # 1. Try Photon Komoot API (Fast, OpenStreetMap based, highly reliable)
            try:
                photon_url = f"https://photon.komoot.io/api/?q={urllib.parse.quote(q)}&limit=3"
                req = urllib.request.Request(photon_url, headers={'User-Agent': 'TrafficVisionAI/2.0'})
                with urllib.request.urlopen(req, timeout=4) as response:
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
            except Exception as e:
                print(f"[API] Photon lookup failed: {e}")

            # 2. Fallback to Nominatim API (with required User-Agent)
            if not results:
                try:
                    nom_url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(q)}&limit=3"
                    req = urllib.request.Request(nom_url, headers={'User-Agent': 'TrafficVisionAI/2.0 (Deepak BTech Project)'})
                    with urllib.request.urlopen(req, timeout=4) as response:
                        nom_data = json.loads(response.read().decode('utf-8'))
                        for item in nom_data:
                            parts = item.get('display_name', '').split(',')
                            results.append({
                                'name': parts[0].strip(),
                                'city': parts[1].strip() if len(parts) > 1 else 'Highway Zone',
                                'road': q,
                                'lat': float(item.get('lat')),
                                'lng': float(item.get('lon')),
                                'display_name': item.get('display_name')
                            })
                except Exception as e:
                    print(f"[API] Nominatim fallback failed: {e}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'results': results, 'query': q}).encode('utf-8'))
            return

        # Default static file handling
        return super().do_GET()

def launch():
    os.chdir(DIRECTORY)
    url = f"http://localhost:{PORT}/dashboard.html"
    print("=" * 65)
    print("[*] TRAFFICVISION AI - TACTICAL SURVEILLANCE & GEOCODING PROXY")
    print("=" * 65)
    print(f"[*] Serving Tactical UI from: {DIRECTORY}")
    print(f"[*] Access URL: {url}")
    print("=" * 65)
    
    with socketserver.TCPServer(("", PORT), TacticalHandler) as httpd:
        print("[*] Server active with /api/geocode endpoint enabled.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")

if __name__ == "__main__":
    launch()
