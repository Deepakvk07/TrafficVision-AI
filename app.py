"""
TrafficVision AI - Real-time Tactical Dashboard Launcher
Serves the web command center with video streaming and live WebSocket/REST readiness.
"""
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def launch():
    os.chdir(DIRECTORY)
    url = f"http://localhost:{PORT}/dashboard.html"
    print("=" * 65)
    print("🚦 TRAFFICVISION AI - SMART HIGHWAY SURVEILLANCE DASHBOARD")
    print("=" * 65)
    print(f"[*] Serving Tactical UI from: {DIRECTORY}")
    print(f"[*] Access URL: {url}")
    print("=" * 65)
    print("[*] Opening default browser automatically...")
    
    webbrowser.open(url)
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("[*] Dashboard server running. Press Ctrl+C to terminate.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")

if __name__ == "__main__":
    launch()
