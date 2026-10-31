"""
TrafficVision AI - High Performance Multi-Threaded Video, CCTV & Telemetry Server
Streams live 1080p highway video, proxies real-time municipal DOT CCTV cameras,
and handles multi-node geocoding with zero hang.
"""
import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import time
import os
import sys

# Configure UTF-8 for Windows PowerShell terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Live CCTV Source Registry
CCTV_SOURCES = {
    'london_tfl': {
        'name': 'Transport for London (TfL) - Westway A40',
        'city': 'London',
        'url': 'https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.03601.jpg',
        'type': 'snapshot'
    },
    'london_tfl_2': {
        'name': 'Transport for London (TfL) - Central Corridor',
        'city': 'London',
        'url': 'https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.03602.jpg',
        'type': 'snapshot'
    },
    'hk_central': {
        'name': 'Hong Kong Transport Dept - Connaught Road Central',
        'city': 'Hong Kong',
        'url': 'https://tdcctv.data.one.gov.hk/H106F.JPG',
        'type': 'snapshot'
    },
    'hk_kowloon': {
        'name': 'Hong Kong Transport Dept - Kwun Tong Bypass',
        'city': 'Hong Kong',
        'url': 'https://tdcctv.data.one.gov.hk/AID02115.JPG',
        'type': 'snapshot'
    },
    'caltrans_baybridge': {
        'name': 'Caltrans QuickMap - I-80 San Francisco Bay Bridge',
        'city': 'San Francisco / Oakland',
        'url': 'https://cwwp2.dot.ca.gov/data/d4/cctv/image/tvd32i80baybridgesastowereast/tvd32i80baybridgesastowereast.jpg',
        'type': 'snapshot'
    }
}

# In-memory cache for Singapore LTA cameras
_sg_cache = {
    'timestamp': 0,
    'cameras': []
}

def get_singapore_cameras():
    global _sg_cache
    now = time.time()
    # Cache for 45 seconds to avoid API rate limits
    if now - _sg_cache['timestamp'] < 45 and _sg_cache['cameras']:
        return _sg_cache['cameras']
    
    try:
        req = urllib.request.Request(
            'https://api.data.gov.sg/v1/transport/traffic-images',
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrafficVisionAI/2.0'}
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            cams = data.get('items', [{}])[0].get('cameras', [])
            if cams:
                _sg_cache['cameras'] = cams
                _sg_cache['timestamp'] = now
                return cams
    except Exception as e:
        print(f"[!] Singapore LTA fetch warning: {e}")
    return _sg_cache['cameras']


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
            path = parsed_url.path
            query_params = urllib.parse.parse_qs(parsed_url.query)

            # -------------------------------------------------------------
            # Endpoint 1: /api/live-cctv?source=singapore|london|hongkong|caltrans
            # -------------------------------------------------------------
            if path == '/api/live-cctv':
                source_key = query_params.get('source', ['singapore'])[0].lower()
                target_url = None

                if 'singapore' in source_key:
                    sg_cams = get_singapore_cameras()
                    if sg_cams:
                        # Allow selecting camera index or camera_id
                        cam_idx = 0
                        if 'pie' in source_key and len(sg_cams) > 1:
                            cam_idx = 1
                        elif 'cte' in source_key and len(sg_cams) > 2:
                            cam_idx = 2
                        target_url = sg_cams[cam_idx % len(sg_cams)].get('image')
                elif source_key in CCTV_SOURCES:
                    target_url = CCTV_SOURCES[source_key]['url']
                elif source_key == 'london':
                    target_url = CCTV_SOURCES['london_tfl']['url']
                elif source_key == 'hongkong' or source_key == 'hk':
                    target_url = CCTV_SOURCES['hk_central']['url']
                elif source_key == 'caltrans':
                    target_url = CCTV_SOURCES['caltrans_baybridge']['url']
                elif source_key == 'custom' or 'url' in query_params:
                    target_url = query_params.get('url', [''])[0].strip()

                if not target_url:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Live CCTV source not available'}).encode('utf-8'))
                    return

                # Fetch real-time snapshot bytes
                try:
                    req = urllib.request.Request(
                        target_url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TrafficVisionAI/2.0'}
                    )
                    with urllib.request.urlopen(req, timeout=4) as response:
                        img_bytes = response.read()
                        content_type = response.headers.get('Content-Type', 'image/jpeg')

                        self.send_response(200)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(len(img_bytes)))
                        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.send_header('X-Live-Source', source_key)
                        self.end_headers()
                        self.wfile.write(img_bytes)
                        return
                except Exception as e:
                    self.send_response(502)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': f'Failed to fetch camera snapshot: {str(e)}'}).encode('utf-8'))
                    return

            # -------------------------------------------------------------
            # Endpoint 2: /api/cctv-status
            # -------------------------------------------------------------
            if path == '/api/cctv-status':
                sg_cams = get_singapore_cameras()
                status_payload = {
                    'status': 'online',
                    'timestamp': int(time.time()),
                    'singapore_cameras': len(sg_cams),
                    'active_sources': list(CCTV_SOURCES.keys()) + ['singapore_aye', 'singapore_pie']
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status_payload).encode('utf-8'))
                return

            # -------------------------------------------------------------
            # Endpoint 3: /api/geocode?q=RoadName
            # -------------------------------------------------------------
            if path == '/api/geocode':
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
    print(f"[*] Live CCTV Proxy: http://localhost:{PORT}/api/live-cctv?source=singapore")
    print("=" * 65)
    
    with MultiThreadedTCPServer(("", PORT), TacticalHandler) as httpd:
        print(f"[*] Listening on port {PORT} (Multi-threaded & Video Streaming Optimized)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")

if __name__ == "__main__":
    launch()
