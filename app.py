"""
TrafficVision AI - Indian Intelligent Traffic Vision & Neural Testing Server
High-performance Multi-Threaded Ultralytics YOLOv8 inference engine for Indian
highway corridors, smart city surveillance, and custom video testing.
"""
import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import time
import os
import re
import sys
import base64
import threading
import numpy as np
import cv2
from ultralytics import YOLO

# UTF-8 terminal encoding configuration
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIRECTORY, "yolov8n.pt")

print("=" * 70)
print("[*] TRAFFICVISION AI - INDIAN HIGHWAY & URBAN VISION TESTING SERVER")
print("=" * 70)
print(f"[*] Loading Ultralytics YOLOv8 Neural Model from: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)
# --------------------------------------------------------------------------
# Spatial Anti-Flicker Vehicle Tracker with Accurate Single-Pass Gate Counting
# Prevents duplicate counting from ByteTrack ID flickering on sampled frames
# --------------------------------------------------------------------------
class SpatialVehicleTracker:
    def __init__(self):
        self.tracks = {}  # id -> track dict
        self.next_id = 1
        self.max_lost_time = 3.5  # Maintain track across occlusions up to 3.5s
        self.cumulative = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.counted_ids = set()

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.cumulative = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.counted_ids.clear()

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = boxAArea + boxBArea - interArea
        return interArea / denom if denom > 0 else 0

    def update(self, raw_detections, gate_y_px, gate_enabled=True):
        now = time.time()
        # Clean up tracks that vanished longer than max_lost_time
        stale_ids = [tid for tid, trk in self.tracks.items() if now - trk['last_seen'] > self.max_lost_time]
        for tid in stale_ids:
            del self.tracks[tid]

        matched_tracks = set()
        matched_dets = set()
        active_ids = list(self.tracks.keys())

        # Pass 1: Bounding Box IoU matching (robust to small shifts)
        for d_idx, det in enumerate(raw_detections):
            best_iou = 0.0
            best_tid = None
            for tid in active_ids:
                if tid in matched_tracks:
                    continue
                trk = self.tracks[tid]
                if trk['cls'] == det['cls'] or {trk['cls'], det['cls']}.issubset({'car', 'truck', 'bus', 'vehicle'}):
                    score = self.iou(det['box'], trk['box'])
                    if score > best_iou:
                        best_iou = score
                        best_tid = tid

            if best_iou > 0.18 and best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                trk = self.tracks[best_tid]
                trk['prev_cy'] = trk['cy']
                trk['prev_cx'] = trk['cx']
                trk['box'] = det['box']
                trk['cx'] = (det['box'][0] + det['box'][2]) / 2.0
                trk['cy'] = (det['box'][1] + det['box'][3]) / 2.0
                trk['last_seen'] = now
                trk['frames_seen'] += 1
                det['assigned_id'] = best_tid

        # Pass 2: Centroid Proximity matching (handles fast movement across 120ms gaps)
        for d_idx, det in enumerate(raw_detections):
            if d_idx in matched_dets:
                continue
            cx = (det['box'][0] + det['box'][2]) / 2.0
            cy = (det['box'][1] + det['box'][3]) / 2.0
            best_dist = 999999.0
            best_tid = None

            for tid in active_ids:
                if tid in matched_tracks:
                    continue
                trk = self.tracks[tid]
                if trk['cls'] == det['cls'] or {trk['cls'], det['cls']}.issubset({'car', 'truck', 'bus', 'vehicle'}):
                    dist = ((cx - trk['cx'])**2 + (cy - trk['cy'])**2) ** 0.5
                    bw = det['box'][2] - det['box'][0]
                    bh = det['box'][3] - det['box'][1]
                    max_allowed_dist = max(80.0, max(bw, bh) * 1.8)
                    if dist < max_allowed_dist and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                trk = self.tracks[best_tid]
                trk['prev_cy'] = trk['cy']
                trk['prev_cx'] = trk['cx']
                trk['box'] = det['box']
                trk['cx'] = cx
                trk['cy'] = cy
                trk['last_seen'] = now
                trk['frames_seen'] += 1
                det['assigned_id'] = best_tid

        # Pass 3: Truly new vehicle entering scene
        for d_idx, det in enumerate(raw_detections):
            if d_idx in matched_dets:
                continue
            new_id = self.next_id
            self.next_id += 1
            cx = (det['box'][0] + det['box'][2]) / 2.0
            cy = (det['box'][1] + det['box'][3]) / 2.0
            self.tracks[new_id] = {
                'id': new_id,
                'cls': det['cls'],
                'box': det['box'],
                'cx': cx,
                'cy': cy,
                'prev_cx': cx,
                'prev_cy': cy,
                'last_seen': now,
                'frames_seen': 1,
                'counted': False
            }
            det['assigned_id'] = new_id

        # Pass 4: Gate Crossing & Strict Single-Pass Counting
        for d_idx, det in enumerate(raw_detections):
            tid = det['assigned_id']
            trk = self.tracks[tid]
            cls_name = trk['cls']

            if not trk['counted']:
                should_count = False
                prev_y = trk['prev_cy']
                curr_y = trk['cy']

                if gate_enabled and gate_y_px > 0:
                    # Vehicle centroid crossed the virtual counting gate
                    if (prev_y < gate_y_px and curr_y >= gate_y_px) or (prev_y > gate_y_px and curr_y <= gate_y_px):
                        should_count = True
                    # Near gate line and confirmed for >= 2 frames
                    elif abs(curr_y - gate_y_px) <= 22 and trk['frames_seen'] >= 2:
                        should_count = True
                else:
                    # Gate disabled: count confirmed vehicle seen for >= 2 frames
                    if trk['frames_seen'] >= 2:
                        should_count = True

                if should_count and tid not in self.counted_ids:
                    trk['counted'] = True
                    self.counted_ids.add(tid)
                    if cls_name in self.cumulative:
                        self.cumulative[cls_name] += 1

        return raw_detections

spatial_tracker = SpatialVehicleTracker()

def reset_session_tracker():
    spatial_tracker.reset()


# --------------------------------------------------------------------------
# High-Performance Multi-Threaded Live Stream Manager (HLS / MJPEG)
# Proxies live YouTube webcams & RTSP/HLS feeds for zero-CORS canvas access
# --------------------------------------------------------------------------
class LiveStreamManager:
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()

    def get_or_create_stream(self, vid, stream_url=None):
        with self.lock:
            if vid in self.streams:
                s = self.streams[vid]
                s['last_access'] = time.time()
                return s

            if not stream_url:
                stream_url = self._extract_stream_url(vid)

            if not stream_url:
                return None

            s = {
                'vid': vid,
                'url': stream_url,
                'cap': None,
                'last_frame': None,
                'last_access': time.time(),
                'running': True,
                'lock': threading.Lock()
            }
            self.streams[vid] = s
            t = threading.Thread(target=self._worker, args=(vid,), daemon=True)
            s['thread'] = t
            t.start()
            return s

    def _extract_stream_url(self, vid):
        try:
            import yt_dlp
            ydl_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f'https://www.youtube.com/watch?v={vid}', download=False)
                formats = info.get('formats', [])
                for f in formats:
                    h = f.get('height') or 0
                    if f.get('vcodec') != 'none' and ('m3u8' in f.get('url', '') or f.get('ext') == 'mp4'):
                        if 360 <= h <= 720:
                            return f.get('url')
                for f in formats:
                    if f.get('vcodec') != 'none' and ('m3u8' in f.get('url', '') or f.get('ext') == 'mp4'):
                        return f.get('url')
        except Exception as e:
            print(f"[!] Error extracting live stream URL for {vid}: {e}")
        return None

    def _worker(self, vid):
        stream = self.streams.get(vid)
        if not stream:
            return
        url = stream['url']
        cap = cv2.VideoCapture(url)
        stream['cap'] = cap
        print(f"[*] Live stream capture thread started for video ID: {vid}")

        fail_count = 0
        while stream['running']:
            if time.time() - stream['last_access'] > 45:
                print(f"[*] Live stream {vid} idle timeout (worker shutting down).")
                break

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                fail_count += 1
                if fail_count > 30:
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(url)
                    fail_count = 0
                time.sleep(0.04)
                continue

            fail_count = 0
            h, w = frame.shape[:2]
            if w > 720:
                scale = 720.0 / w
                frame = cv2.resize(frame, (720, int(h * scale)))

            ret_enc, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ret_enc:
                with stream['lock']:
                    stream['last_frame'] = jpeg.tobytes()

            time.sleep(0.04)

        cap.release()
        with self.lock:
            if vid in self.streams and self.streams[vid] is stream:
                del self.streams[vid]

    def get_frame(self, vid):
        stream = self.streams.get(vid)
        if not stream:
            stream = self.get_or_create_stream(vid)
        if stream:
            stream['last_access'] = time.time()
            with stream['lock']:
                return stream.get('last_frame')
        return None

live_stream_manager = LiveStreamManager()


# Neural warm-up
_warmup_frame = np.zeros((360, 640, 3), dtype=np.uint8)
yolo_model.predict(_warmup_frame, classes=[2, 3, 5, 7], verbose=False)
print("[+] YOLOv8 Model Online: Tracking Classes [car, motorcycle, bus, truck]")

# --------------------------------------------------------------------------
# Verified Indian Highway Corridors & Smart City Surveillance Nodes
# (Indian Cities ONLY - Delhi NCR, Mumbai, Bengaluru, Hyderabad, etc.)
# --------------------------------------------------------------------------
INDIAN_CORRIDORS = [
    {
        "id": 1,
        "name": "Yamuna Expressway",
        "city": "Delhi NCR",
        "state": "Uttar Pradesh",
        "road": "Greater Noida-Agra Corridor (165 km)",
        "lat": 28.3240,
        "lng": 77.5450,
        "count": 34,
        "in": 21,
        "out": 13,
        "speed": 88,
        "lanes": 6,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 22, "motorcycle": 6, "bus": 3, "truck": 3},
        "description": "High-speed access-controlled 6-lane concrete corridor connecting Greater Noida to Agra."
    },
    {
        "id": 2,
        "name": "Delhi-Meerut Expressway",
        "city": "Delhi NCR",
        "state": "Delhi / UP",
        "road": "NE-3 / NH-9 Akshardham Corridor",
        "lat": 28.6280,
        "lng": 77.3000,
        "count": 68,
        "in": 42,
        "out": 26,
        "speed": 64,
        "lanes": 14,
        "congestion": "Heavy",
        "status": "online",
        "classes": {"car": 45, "motorcycle": 14, "bus": 5, "truck": 4},
        "description": "India's widest 14-lane expressway with dedicated multi-lane access transit."
    },
    {
        "id": 3,
        "name": "DND Flyway",
        "city": "Delhi NCR",
        "state": "Delhi / Noida",
        "road": "Yamuna River Toll Bridge",
        "lat": 28.5750,
        "lng": 77.2900,
        "count": 52,
        "in": 31,
        "out": 21,
        "speed": 72,
        "lanes": 8,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 38, "motorcycle": 9, "bus": 3, "truck": 2},
        "description": "Critical 8-lane corridor linking South Delhi with Noida across the Yamuna River."
    },
    {
        "id": 4,
        "name": "Western Express Highway",
        "city": "Mumbai",
        "state": "Maharashtra",
        "road": "NH-48 Santacruz-Dahisar Corridor",
        "lat": 19.1136,
        "lng": 72.8697,
        "count": 62,
        "in": 38,
        "out": 24,
        "speed": 32,
        "lanes": 8,
        "congestion": "Heavy",
        "status": "online",
        "classes": {"car": 41, "motorcycle": 15, "bus": 4, "truck": 2},
        "description": "Major arterial north-south highway of suburban Mumbai with elevated metro line."
    },
    {
        "id": 5,
        "name": "Bandra-Worli Sea Link",
        "city": "Mumbai",
        "state": "Maharashtra",
        "road": "Rajiv Gandhi Coastal Freeway",
        "lat": 19.0269,
        "lng": 72.8177,
        "count": 44,
        "in": 25,
        "out": 19,
        "speed": 76,
        "lanes": 8,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 38, "motorcycle": 0, "bus": 4, "truck": 2},
        "description": "Iconic 8-lane cable-stayed bridge spanning Mahim Bay connecting Bandra to Worli."
    },
    {
        "id": 6,
        "name": "Eastern Freeway",
        "city": "Mumbai",
        "state": "Maharashtra",
        "road": "P D'Mello Road to Chembur Corridor",
        "lat": 19.0050,
        "lng": 72.8600,
        "count": 38,
        "in": 22,
        "out": 16,
        "speed": 70,
        "lanes": 4,
        "congestion": "Free Flow",
        "status": "online",
        "classes": {"car": 32, "motorcycle": 0, "bus": 4, "truck": 2},
        "description": "Elevated 4-lane freeway drastically cutting travel time between South Mumbai and Eastern suburbs."
    },
    {
        "id": 7,
        "name": "Outer Ring Road (Marathahalli)",
        "city": "Bengaluru",
        "state": "Karnataka",
        "road": "ORR Silicon Tech Corridor",
        "lat": 12.9560,
        "lng": 77.7011,
        "count": 74,
        "in": 46,
        "out": 28,
        "speed": 22,
        "lanes": 6,
        "congestion": "Severe",
        "status": "warning",
        "classes": {"car": 42, "motorcycle": 24, "bus": 6, "truck": 2},
        "description": "High-density IT corridor connecting Whitefield, Bellandur, and Electronic City."
    },
    {
        "id": 8,
        "name": "Electronic City Elevated Flyover",
        "city": "Bengaluru",
        "state": "Karnataka",
        "road": "Hosur Road Expressway (NH-44)",
        "lat": 12.8452,
        "lng": 77.6602,
        "count": 42,
        "in": 24,
        "out": 18,
        "speed": 62,
        "lanes": 4,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 30, "motorcycle": 8, "bus": 3, "truck": 1},
        "description": "9.9 km elevated 4-lane expressway bridging Silk Board to Electronic City tech campus."
    },
    {
        "id": 9,
        "name": "PVNR Elevated Expressway",
        "city": "Hyderabad",
        "state": "Telangana",
        "road": "Airport Corridor (11.6 km)",
        "lat": 17.3600,
        "lng": 78.4400,
        "count": 36,
        "in": 20,
        "out": 16,
        "speed": 78,
        "lanes": 4,
        "congestion": "Free Flow",
        "status": "online",
        "classes": {"car": 31, "motorcycle": 0, "bus": 3, "truck": 2},
        "description": "India's premier elevated expressway linking Mehdipatnam directly to Rajiv Gandhi Airport."
    },
    {
        "id": 10,
        "name": "Nehru Outer Ring Road",
        "city": "Hyderabad",
        "state": "Telangana",
        "road": "Gachibowli Junction ORR",
        "lat": 17.4400,
        "lng": 78.3489,
        "count": 50,
        "in": 29,
        "out": 21,
        "speed": 92,
        "lanes": 8,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 36, "motorcycle": 7, "bus": 3, "truck": 4},
        "description": "8-lane ring expressway surrounding Hyderabad designed for 120 km/h transit."
    },
    {
        "id": 11,
        "name": "Rajiv Gandhi Salai (OMR)",
        "city": "Chennai",
        "state": "Tamil Nadu",
        "road": "Old Mahabalipuram IT Expressway",
        "lat": 12.9249,
        "lng": 80.2200,
        "count": 48,
        "in": 27,
        "out": 21,
        "speed": 45,
        "lanes": 6,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 28, "motorcycle": 15, "bus": 3, "truck": 2},
        "description": "Primary 6-lane IT corridor of Chennai hosting global technology parks and research hubs."
    },
    {
        "id": 12,
        "name": "Mumbai-Pune Expressway",
        "city": "Pune",
        "state": "Maharashtra",
        "road": "Yashwantrao Chavan Expy (Khandala)",
        "lat": 18.7500,
        "lng": 73.4000,
        "count": 46,
        "in": 25,
        "out": 21,
        "speed": 84,
        "lanes": 6,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 32, "motorcycle": 0, "bus": 5, "truck": 9},
        "description": "India's first 6-lane high-speed concrete tollway cutting through the Sahyadri Ghats."
    },
    {
        "id": 13,
        "name": "Vidyasagar Setu (Second Hooghly Bridge)",
        "city": "Kolkata",
        "state": "West Bengal",
        "road": "Howrah-Kolkata Cable-Stayed Bridge",
        "lat": 22.5583,
        "lng": 88.3289,
        "count": 55,
        "in": 32,
        "out": 23,
        "speed": 42,
        "lanes": 6,
        "congestion": "Moderate",
        "status": "online",
        "classes": {"car": 30, "motorcycle": 12, "bus": 8, "truck": 5},
        "description": "Toll bridge spanning the Hooghly River linking the twin cities of Kolkata and Howrah."
    },
    {
        "id": 14,
        "name": "National Expressway 1 (NE-1)",
        "city": "Ahmedabad",
        "state": "Gujarat",
        "road": "Ahmedabad-Vadodara Expressway",
        "lat": 22.8800,
        "lng": 72.8200,
        "count": 35,
        "in": 19,
        "out": 16,
        "speed": 94,
        "lanes": 4,
        "congestion": "Free Flow",
        "status": "online",
        "classes": {"car": 24, "motorcycle": 3, "bus": 3, "truck": 5},
        "description": "93 km high-velocity expressway between Gujarat's commercial capital and Vadodara."
    }
]


class MultiThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class TacticalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Range')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        try:
            parsed_url = urllib.parse.urlparse(self.path)

            # -------------------------------------------------------------
            # Endpoint: POST /api/detect
            # Executes real Ultralytics YOLOv8 inference on video frame
            # -------------------------------------------------------------
            if parsed_url.path == '/api/detect':
                content_len = int(self.headers.get('Content-Length', 0))
                if content_len == 0:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Empty payload'}).encode('utf-8'))
                    return

                body_bytes = self.rfile.read(content_len)
                try:
                    payload = json.loads(body_bytes.decode('utf-8'))
                except Exception:
                    payload = {}

                img_data = payload.get('image', '')
                if not img_data:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'No image provided'}).encode('utf-8'))
                    return

                if ',' in img_data:
                    img_data = img_data.split(',', 1)[1]

                try:
                    raw_bytes = base64.b64decode(img_data)
                    nparr = np.frombuffer(raw_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    frame = None

                if frame is None or frame.size == 0:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Failed to decode image'}).encode('utf-8'))
                    return

                h, w = frame.shape[:2]
                t0 = time.time()

                if payload.get('reset', False):
                    reset_session_tracker()

                conf_thresh = float(payload.get('conf', 0.15))
                gate_ratio = float(payload.get('gate_y', 0.45))
                gate_line_y = h * gate_ratio
                gate_enabled = bool(payload.get('gate_enabled', True))

                # Execute Ultralytics YOLOv8 detection
                res = yolo_model.predict(frame, classes=[2, 3, 5, 7], conf=conf_thresh, imgsz=640, verbose=False)
                dt = (time.time() - t0) * 1000

                raw_detections = []
                boxes = res[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names.get(cls_id, 'vehicle')
                    raw_detections.append({
                        'box': xyxy,
                        'conf': conf,
                        'cls': cls_name
                    })

                # Run Spatial Anti-Flicker Vehicle Tracker & Accurate Gate Counting
                tracked_detections = spatial_tracker.update(raw_detections, gate_line_y, gate_enabled=gate_enabled)

                class_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
                detections = []
                for det in tracked_detections:
                    cls_name = det['cls']
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                    detections.append({
                        'id': det['assigned_id'],
                        'box': [round(x, 1) for x in det['box']],
                        'conf': round(det['conf'], 2),
                        'cls': cls_name
                    })

                count = len(detections)
                if count <= 10:
                    congestion = "Free Flow"
                elif count <= 25:
                    congestion = "Moderate"
                elif count <= 45:
                    congestion = "Heavy"
                else:
                    congestion = "Severe Congestion"

                cum = dict(spatial_tracker.cumulative)
                total_cum = sum(cum.values())

                result_payload = {
                    'detections': detections,
                    'count': count,
                    'class_counts': class_counts,
                    'cumulative_counts': cum,
                    'total_cumulative': total_cum,
                    'congestion': congestion,
                    'frame_width': w,
                    'frame_height': h,
                    'inference_ms': round(dt, 1),
                    'fps': round(1000.0 / max(dt, 1.0), 1)
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result_payload).encode('utf-8'))
                return

            # -------------------------------------------------------------
            # Endpoint: POST /api/prepare-youtube
            # Streams/caches YouTube videos as same-origin MP4
            # enabling full HTML5 Canvas + YOLOv8 real-time AI bounding box detection!
            # -------------------------------------------------------------
            if parsed_url.path == '/api/reset-tracking':
                reset_session_tracker()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'reset',
                    'cumulative': dict(spatial_tracker.cumulative),
                    'total_cumulative': sum(spatial_tracker.cumulative.values())
                }).encode('utf-8'))
                return

            if parsed_url.path == '/api/prepare-youtube':
                content_len = int(self.headers.get('Content-Length', 0))
                body_bytes = self.rfile.read(content_len) if content_len > 0 else b'{}'
                try:
                    payload = json.loads(body_bytes.decode('utf-8'))
                except Exception:
                    payload = {}

                raw_url = payload.get('url', '').strip()
                if not raw_url:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'No URL provided'}).encode('utf-8'))
                    return

                # Extract 11-char video ID
                m = re.search(r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?|live|shorts)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})', raw_url)
                vid = m.group(1) if m else raw_url

                cache_dir = os.path.join(DIRECTORY, 'youtube_cache')
                os.makedirs(cache_dir, exist_ok=True)
                out_file = os.path.join(cache_dir, f'{vid}.mp4')

                # 1. First inspect the URL to determine if it is a live stream or static video
                try:
                    import yt_dlp
                    ydl_info_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True}
                    with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
                        info = ydl.extract_info(f'https://www.youtube.com/watch?v={vid}', download=False)
                        is_live = bool(info.get('is_live') or info.get('live_status') in ['is_live', 'is_upcoming'])
                        title = info.get('title', f'YouTube Stream ({vid})')
                except Exception as e:
                    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            'status': 'ready',
                            'is_live': False,
                            'video_url': f'/youtube_cache/{vid}.mp4',
                            'video_id': vid,
                            'title': f'YouTube Stream ({vid})'
                        }).encode('utf-8'))
                        return
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e), 'video_id': vid}).encode('utf-8'))
                    return

                # 2. If it is a LIVE STREAM -> Return live stream proxy feed URL!
                if is_live:
                    stream_url = None
                    formats = info.get('formats', [])
                    for f in formats:
                        h = f.get('height') or 0
                        if f.get('vcodec') != 'none' and ('m3u8' in f.get('url', '') or f.get('ext') == 'mp4'):
                            if 360 <= h <= 720:
                                stream_url = f.get('url')
                                break
                    if not stream_url:
                        for f in formats:
                            if f.get('vcodec') != 'none' and ('m3u8' in f.get('url', '') or f.get('ext') == 'mp4'):
                                stream_url = f.get('url')
                                break
                    live_stream_manager.get_or_create_stream(vid, stream_url=stream_url)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'ready',
                        'is_live': True,
                        'stream_type': 'mjpeg',
                        'video_url': f'/api/live-stream-feed?vid={vid}',
                        'video_id': vid,
                        'title': title
                    }).encode('utf-8'))
                    return

                # 3. If it is a REGULAR VIDEO -> Check cache or download MP4 (no ffmpeg required)
                if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'ready',
                        'is_live': False,
                        'video_url': f'/youtube_cache/{vid}.mp4',
                        'video_id': vid,
                        'title': title
                    }).encode('utf-8'))
                    return

                try:
                    ydl_dl_opts = {
                        'format': '136/135/134/133/160/bestvideo[ext=mp4]/best',
                        'outtmpl': out_file,
                        'quiet': True,
                        'no_warnings': True
                    }
                    with yt_dlp.YoutubeDL(ydl_dl_opts) as ydl:
                        ydl.download([f'https://www.youtube.com/watch?v={vid}'])

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'ready',
                        'is_live': False,
                        'video_url': f'/youtube_cache/{vid}.mp4',
                        'video_id': vid,
                        'title': title
                    }).encode('utf-8'))
                    return
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e), 'video_id': vid}).encode('utf-8'))
                    return

            self.send_response(404)
            self.end_headers()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as e:
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            except Exception:
                pass

    def do_GET(self):
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            query_params = urllib.parse.parse_qs(parsed_url.query)

            # -------------------------------------------------------------
            # Endpoints: /api/live-stream-feed & /api/live-frame
            # High-performance MJPEG live webcam streaming with full Canvas draw access
            # -------------------------------------------------------------
            if path == '/api/live-stream-feed':
                vid = query_params.get('vid', [''])[0].strip()
                if not vid:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Missing vid parameter"}')
                    return

                stream = live_stream_manager.get_or_create_stream(vid)
                if not stream:
                    self.send_response(502)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unable to connect to live stream"}')
                    return

                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                last_sent = None
                try:
                    while True:
                        frame_bytes = live_stream_manager.get_frame(vid)
                        if frame_bytes and frame_bytes != last_sent:
                            last_sent = frame_bytes
                            chunk = (
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                b'Content-Length: ' + str(len(frame_bytes)).encode('utf-8') + b'\r\n\r\n' +
                                frame_bytes + b'\r\n'
                            )
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        time.sleep(0.04)
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    pass
                return

            if path == '/api/live-frame':
                vid = query_params.get('vid', [''])[0].strip()
                if not vid:
                    self.send_response(400)
                    self.end_headers()
                    return

                frame_bytes = live_stream_manager.get_frame(vid)
                if not frame_bytes:
                    for _ in range(25):
                        time.sleep(0.1)
                        frame_bytes = live_stream_manager.get_frame(vid)
                        if frame_bytes:
                            break

                if frame_bytes:
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                else:
                    self.send_response(503)
                    self.end_headers()
                return

            # -------------------------------------------------------------
            # Endpoint: /api/indian-corridors
            # -------------------------------------------------------------
            if path == '/api/indian-corridors':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps(INDIAN_CORRIDORS).encode('utf-8'))
                return

            # -------------------------------------------------------------
            # Endpoint: /api/status
            # -------------------------------------------------------------
            if path == '/api/status':
                status_payload = {
                    'status': 'online',
                    'timestamp': int(time.time()),
                    'region': 'India',
                    'active_corridors': len(INDIAN_CORRIDORS),
                    'yolo_model': 'yolov8n.pt (Active, PyTorch)',
                    'supported_classes': ['car', 'motorcycle', 'bus', 'truck']
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status_payload).encode('utf-8'))
                return

            # -------------------------------------------------------------
            # Endpoint: /api/geocode?q=RoadName
            # Geocodes Indian roads / corridors
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
                try:
                    search_query = q if 'india' in q.lower() else f"{q}, India"
                    photon_url = f"https://photon.komoot.io/api/?q={urllib.parse.quote(search_query)}&limit=4"
                    req = urllib.request.Request(photon_url, headers={'User-Agent': 'TrafficVisionAI-India/2.0'})
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        for feat in data.get('features', []):
                            coords = feat.get('geometry', {}).get('coordinates', [0, 0])
                            props = feat.get('properties', {})
                            name = props.get('name') or props.get('street') or q
                            city = props.get('city') or props.get('state') or 'India'
                            results.append({
                                'name': name,
                                'city': city,
                                'country': props.get('country', 'India'),
                                'lat': coords[1],
                                'lng': coords[0]
                            })
                except Exception:
                    pass

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'query': q, 'results': results}).encode('utf-8'))
                return

            # -------------------------------------------------------------
            # Video Streaming with HTTP 206 Partial Content Support
            # -------------------------------------------------------------
            clean_path = path.lstrip('/') or 'dashboard.html'
            file_path = os.path.join(DIRECTORY, clean_path)

            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                range_header = self.headers.get('Range')

                # Content type guessing
                content_type = 'application/octet-stream'
                if file_path.endswith('.html'):
                    content_type = 'text/html; charset=utf-8'
                elif file_path.endswith('.css'):
                    content_type = 'text/css'
                elif file_path.endswith('.js'):
                    content_type = 'application/javascript'
                elif file_path.endswith('.mp4'):
                    content_type = 'video/mp4'
                elif file_path.endswith('.webm'):
                    content_type = 'video/webm'
                elif file_path.endswith('.json'):
                    content_type = 'application/json'
                elif file_path.endswith('.png'):
                    content_type = 'image/png'
                elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                    content_type = 'image/jpeg'

                if range_header and range_header.startswith('bytes='):
                    ranges = range_header[6:].split('-')
                    start = int(ranges[0]) if ranges[0] else 0
                    end = int(ranges[1]) if len(ranges) > 1 and ranges[1] else file_size - 1
                    end = min(end, file_size - 1)
                    length = end - start + 1

                    self.send_response(206)
                    self.send_header('Content-Type', content_type)
                    self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                    self.send_header('Content-Length', str(length))
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()

                    with open(file_path, 'rb') as f:
                        f.seek(start)
                        remaining = length
                        chunk_size = 64 * 1024
                        while remaining > 0:
                            read_len = min(remaining, chunk_size)
                            chunk = f.read(read_len)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return

                # Standard 200 OK
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(file_path, 'rb') as f:
                    chunk_size = 64 * 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return

            # Default fallback to parent SimpleHTTPRequestHandler
            return super().do_GET()

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as e:
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Server Internal Error: {str(e)}".encode('utf-8'))
            except Exception:
                pass


def launch():
    url = f"http://localhost:{PORT}/dashboard.html"
    print("=" * 70)
    print(f"[*] Serving Directory: {DIRECTORY}")
    print(f"[*] Access URL: {url}")
    print(f"[*] Neural Inference Endpoint: http://localhost:{PORT}/api/detect")
    print(f"[*] Indian Corridors Registry: http://localhost:{PORT}/api/indian-corridors")
    print("=" * 70)

    with MultiThreadedTCPServer(("", PORT), TacticalHandler) as httpd:
        print(f"[*] Listening on port {PORT} (Multi-threaded & HTTP 206 Partial Streaming Ready)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")


if __name__ == "__main__":
    launch()
