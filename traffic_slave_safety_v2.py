
#!/usr/bin/env python3
"""
traffic_slave.py - Board B Slave Controller
===========================================
Runs on: 192.168.20.168 (hkcrc2)

This is the slave traffic light client. It:
  1. Detects vehicles on Road B via NPU (YOLOv8-RKNN)
  2. Sends Road B vehicle counts to Board A via LoRa
  3. Receives traffic light state commands from Board A via LoRa
  4. Displays Road B camera feed with traffic light overlay

LoRa Protocol:
  B → A:  DET:car:N,motorcycle:N,bus:N,truck:N
  A → B:  LIGHT:<STATE>,<REMAINING_SECONDS>
"""

import os
import sys
import threading
import time
import serial
import json
import queue
from collections import deque

os.environ["GLOG_minloglevel"] = "3"
os.environ["RKNN_LOG_LEVEL"] = "0"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
os.environ["OMP_NUM_THREADS"] = "2"

# Auto-detect DISPLAY from available X sockets
def _find_display():
    if os.environ.get("DISPLAY"):
        return
    import glob
    sockets = glob.glob("/tmp/.X11-unix/X*")
    for s in sorted(sockets):
        n = int(s.rsplit("X", 1)[-1])
        if n < 100:  # skip GDM sockets (X1024 etc)
            os.environ["DISPLAY"] = f":{n}"
            return
    os.environ["DISPLAY"] = ":0"  # fallback
_find_display()

import cv2
import numpy as np
from rknnlite.api import RKNNLite
from relay_controller import RS485RelayController

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

# CAMERA_SOURCE = 0

CAMERA_SOURCE = "rtsp://admin:HKcrc3130@192.168.1.64:554/h264/ch1/main/av_stream"

MODEL_PATH = "/home/hkcrc2/Sentinel-Traffic-Control_B/yolov8n.rknn"
SERIAL_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG0419BN-if00-port0"
BAUDRATE = 115200

# RS485 relay board for real B-side traffic light output
# Use by-id path so USB replug / re-enumeration cannot swap LoRa and relay.
RELAY_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BH002IVA-if00-port0"
RELAY_BAUDRATE = 9600

CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# Vehicle classes (COCO): car, motorcycle, bus, truck
TARGET_CLASS_IDS = [2, 3, 5, 7]
TARGET_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

# How often to send detection data to Board A (seconds)
DET_SEND_INTERVAL = 0.5

# Maximum age of Board A light command before fallback
LIGHT_CMD_TIMEOUT = 1.5

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Light state constants (mirrored from TrafficState enum)
LIGHT_STATES = {
    "GREEN_A":  (0, 255, 0),    # Road A green
    "YELLOW_A": (0, 255, 255),  # Road A yellow
    "ALL_RED_A": (0, 0, 255),   #  (A→B)
    "GREEN_B":  (0, 255, 0),    # Road B green
    "YELLOW_B": (0, 255, 255),  # Road B yellow
    "ALL_RED_B": (0, 0, 255),   # All red (B→A)
}


# ──────────────────────────────────────────────
#  YOLOv8 Fast Post-Processor
# ──────────────────────────────────────────────

class YOLOv8_Fast_PostProcess:
    """Fast RKNN post-processor for YOLOv8 detection heads."""

    def __init__(self, conf_threshold=0.35, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.strides = [8, 16, 32]
        self.reg_max = 16
        self.classes = COCO_CLASSES

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def process(self, outputs, ori_w, ori_h):
        all_boxes, all_confs, all_class_ids = [], [], []
        for i in range(3):
            stride = self.strides[i]
            box_head = np.squeeze(outputs[i * 3])
            cls_head = np.squeeze(outputs[i * 3 + 1])
            score_head = np.squeeze(outputs[i * 3 + 2])
            _, h, w = box_head.shape
            box_head = box_head.reshape(64, -1).T
            cls_head = cls_head.reshape(80, -1).T
            score_head = score_head.reshape(-1, 1)
            scores = cls_head * score_head
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            keep_idx = max_scores > self.conf_threshold
            if not np.any(keep_idx):
                continue
            filtered_boxes_raw = box_head[keep_idx]
            filtered_scores = max_scores[keep_idx]
            filtered_class_ids = class_ids[keep_idx]
            num_kept = filtered_boxes_raw.shape[0]
            filtered_boxes_raw = filtered_boxes_raw.reshape(num_kept, 4, self.reg_max)
            filtered_boxes_raw = self._softmax(filtered_boxes_raw, axis=-1)
            acc_matrix = np.arange(self.reg_max, dtype=np.float32)
            box_decoded = np.sum(filtered_boxes_raw * acc_matrix, axis=-1) * stride
            grid_y, grid_x = np.indices((h, w), dtype=np.float32)
            grid_x = grid_x.flatten()[keep_idx]
            grid_y = grid_y.flatten()[keep_idx]
            anchor_x = (grid_x + 0.5) * stride
            anchor_y = (grid_y + 0.5) * stride
            x1 = anchor_x - box_decoded[:, 0]
            y1 = anchor_y - box_decoded[:, 1]
            x2 = anchor_x + box_decoded[:, 2]
            y2 = anchor_y + box_decoded[:, 3]
            rx1 = (x1 * (ori_w / 640.0)).astype(np.int32)
            ry1 = (y1 * (ori_h / 640.0)).astype(np.int32)
            rw = ((x2 - x1) * (ori_w / 640.0)).astype(np.int32)
            rh = ((y2 - y1) * (ori_h / 640.0)).astype(np.int32)
            all_boxes.extend(np.stack([rx1, ry1, rw, rh], axis=1).tolist())
            all_confs.extend(filtered_scores.astype(float).tolist())
            all_class_ids.extend(filtered_class_ids.tolist())
        indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, self.conf_threshold, self.iou_threshold)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': all_boxes[i],
                    'conf': all_confs[i],
                    'class': self.classes[all_class_ids[i]]
                })
        return results


# ──────────────────────────────────────────────
#  RTSP Stream Reader
# ──────────────────────────────────────────────

class RTSPStreamReader:
    """
    Thread-safe video capture with automatic reconnection.

    - A background thread reads frames and pushes the latest one
      into a Queue(maxsize=1) so the main loop never sees a torn
      (ret, frame) pair.
    - When the stream drops (cap.read() returns False), the reader
      automatically releases the old capture, sleeps, and re-opens
      the RTSP source so the pipeline never permanently stalls.
    """

    RECONNECT_DELAY = 2.0   # seconds to wait before reconnection attempt

    def __init__(self, source, width=640, height=480):
        """
        Args:
            source: camera index (int) or RTSP URL (str).
            width / height: desired capture resolution.
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.stopped = False
        # maxsize=1 → only the single latest frame is kept.
        # Queue itself provides the thread-safety guarantee.
        self.frame_queue = queue.Queue(maxsize=1)

    # ── public API ──────────────────────────────

    def start(self):
        """Open the first capture and launch the background reader."""
        self._open_capture()
        self.stopped = False
        threading.Thread(target=self._update, args=(), daemon=True).start()
        return self

    def read_latest(self):
        """
        Return the most recent (ret, frame) tuple without blocking.
        If no frame is available yet, returns (False, None).
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return (False, None)

    def stop(self):
        """Signal the background thread to exit."""
        self.stopped = True

    # ── internals ───────────────────────────────

    def _open_capture(self):
        """Create and configure a VideoCapture for *source*."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        return False

    def _update(self):
        """Background loop: read frames → push into queue; reconnect on failure."""
        while not self.stopped:
            # ── ensure capture handle is valid ──
            if self.cap is None or not self.cap.isOpened():
                print("[CAM] Stream lost — attempting reconnect ...")
                if not self._open_capture():
                    time.sleep(self.RECONNECT_DELAY)
                    continue
                print("[CAM] Reconnect succeeded")

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Frame read failed — tear down and retry after a delay
                print("[CAM] Frame read failed, reconnecting ...")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.RECONNECT_DELAY)
                continue

            # ── thread-safe publish: discard old, keep only latest ──
            try:
                self.frame_queue.get_nowait()   # drop stale frame
            except queue.Empty:
                pass
            self.frame_queue.put((True, frame))


# ──────────────────────────────────────────────
#  LoRa Communication (send DET, receive LIGHT)
# ──────────────────────────────────────────────

class LoRaSlave:
    """
    Slave LoRa communication.

    - Sends DET messages to Board A periodically
    - Receives LIGHT commands from Board A (background thread)
    """

    def __init__(self, port, baudrate, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self._reconnecting = False

        # Latest light command from Board A
        self.lock = threading.Lock()
        self.send_lock = threading.Lock()   # protects serial writes from concurrent threads
        self.light_state_name = "GREEN_A"   # default
        self.light_remaining = 30.0
        self.light_last_update = 0.0
        self.fb_start_time = 0.0           # fallback start timestamp
        self.light_color = (0, 255, 0)      # default green for A (Road B sees red)
        self.is_green = False               # Road B's own green status
        self.is_yellow = False

    def _reconnect(self):
        """Reconnect serial after USB unplug/replug (I/O error)."""
        if self._reconnecting:
            return
        self._reconnecting = True
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        delay = 2.0
        while self.running:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
                print(f"[LoRa] Reconnected to {self.port}")
                break
            except Exception as e:
                print(f"[LoRa] Reconnect failed, retry in {delay:.0f}s: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 10.0)
        self._reconnecting = False

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            print(f"[LoRa] Serial {self.port} opened (baudrate={self.baudrate})")
            return True
        except Exception as e:
            print(f"[LoRa] Error: cannot open serial {self.port}: {e}")
            return False

    def start_receiver(self):
        """Start background thread to listen for Board A commands."""
        self.running = True
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()
        print("[LoRa] Receiver thread started")

    def _receive_loop(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw = self.ser.readline()
                    if raw:
                        msg = raw.decode('utf-8', errors='ignore').strip()
                        self._handle_message(msg)
            except Exception as e:
                if 'Input/output error' in str(e) or 'Errno 5' in str(e):
                    print(f"  [LoRa] I/O error, reconnecting...")
                    self._reconnect()
                else:
                    print(f"  [LoRa] receive error: {e}")
            time.sleep(0.05)

    def _handle_message(self, msg):
        """Parse incoming messages from Board A."""
        if msg.startswith("LIGHT:"):
            # Board A is sending traffic light state
            # Format: LIGHT:STATE_NAME,REMAINING
            # Example: LIGHT:GREEN_A,25
            data = msg[6:]
            parts = data.split(",")
            if len(parts) >= 1:
                state_name = parts[0].strip()
                if len(parts) >= 2:
                    try:
                        remaining = float(parts[1].strip())
                    except ValueError:
                        remaining = 0.0
                else:
                    remaining = 0.0
                with self.lock:
                    self.light_state_name = state_name
                    self.light_remaining = remaining
                    self.light_last_update = time.time()

                    # Determine Road B's light color
                    if state_name in ("GREEN_B",):
                        self.light_color = (0, 255, 0)    # Green for B
                        self.is_green = True
                        self.is_yellow = False
                    elif state_name in ("YELLOW_B",):
                        self.light_color = (0, 255, 255)  # Yellow for B
                        self.is_green = False
                        self.is_yellow = True
                    elif state_name in ("GREEN_A", "YELLOW_A", "ALL_RED_A", "ALL_RED_B"):
                        self.light_color = (0, 0, 255)    # Red for B
                        self.is_green = False
                        self.is_yellow = False
                    else:
                        self.light_color = (0, 0, 255)    # Default red
                        self.is_green = False
                        self.is_yellow = False

        elif msg.startswith("DET:"):
            # Echo from Board A (should not normally happen)
            pass
        elif msg == "ACK":
            pass
        elif msg.startswith("PING"):
            self._send_raw("PONG")
        else:
            pass

    def send_detection(self, class_counts):
        """
        Send Road B detection results to Board A.
        Format: DET:car:3,motorcycle:1,bus:0,truck:2
        """
        parts = [f"{name}:{class_counts.get(name, 0)}"
                 for name in TARGET_CLASS_NAMES]
        msg = "DET:" + ",".join(parts)
        self._send_raw(msg)

    def _send_raw(self, message):
        """Send a raw string over LoRa serial (thread-safe)."""
        if self.ser:
            try:
                data = (message + "\n").encode('utf-8')
                with self.send_lock:
                    self.ser.write(data)
                return True
            except Exception as e:
                if 'Input/output error' in str(e) or 'Errno 5' in str(e):
                    print(f"  [LoRa] I/O error on send, reconnecting...")
                    self._reconnect()
                else:
                    print(f"  [LoRa] send failed: {e}")
                return False
        return False

    def get_light_state(self):
        """Thread-safe read of current light state."""
        with self.lock:
            return (self.light_state_name, self.light_remaining,
                    self.light_color, self.is_green, self.is_yellow,
                    self.light_last_update)

    def is_light_data_fresh(self):
        """Check if light command from A is recent."""
        with self.lock:
            if self.light_last_update == 0.0:
                return False
            return (time.time() - self.light_last_update) < LIGHT_CMD_TIMEOUT

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()
            print("[LoRa] serial closed")




# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def draw_traffic_light(frame, x, y, color, radius=35):
    """Draw a simulated traffic light circle."""
    cv2.circle(frame, (x, y), radius, color, -1)
    cv2.circle(frame, (x, y), radius, (255, 255, 255), 3)


def draw_slave_display(frame, detections, class_counts, total_count,
                        light_state, light_remaining, light_color,
                        is_green, light_fresh, fps):
    """
    Draw the slave display overlay on the frame.

    Layout:
      - Left panel: detection summary + light state
      - Right side: traffic light indicator
    """
    h, w = frame.shape[:2]

    # ── Left panel background ──
    panel_w = 240
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Title ──
    cv2.putText(frame, "SLAVE (Board B)", (10, y), font, 0.7, (0, 255, 255), 2)
    y += 30

    # ── Road B detection ──
    cv2.putText(frame, "--- Road B (Local) ---", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    for cls_name in TARGET_CLASS_NAMES:
        cnt = class_counts.get(cls_name, 0)
        cv2.putText(frame, f"  {cls_name}: {cnt}", (10, y), font, 0.5, (255, 255, 255), 1)
        y += 20
    cv2.putText(frame, f"  TOTAL: {total_count}", (10, y), font, 0.6, (0, 255, 0), 1)
    y += 28

    # ── LoRa TX status ──
    cv2.putText(frame, "LoRa TX: DET -> A", (10, y), font, 0.4, (0, 200, 0), 1)
    y += 20

    # ── Light state from Master ──
    cv2.putText(frame, "--- Light (from Master) ---", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 22
    if light_fresh:
        cv2.putText(frame, f"  State: {light_state}", (10, y), font, 0.6, (0, 255, 255), 2)
        y += 22
        cv2.putText(frame, f"  Remaining: {light_remaining:.1f}s", (10, y), font, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "  NO SIGNAL", (10, y), font, 0.5, (0, 0, 255), 1)
        y += 22
    y += 28

    # ── FPS ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15), font, 0.5, (0, 255, 0), 1)

    # ── Right side: Road B traffic light ──
    light_x = w - 60
    light_y = 80

    # Status text
    if is_green:
        status = "GO"
        status_color = (0, 255, 0)
    elif light_state in ("YELLOW_B",):
        status = "YIELD"
        status_color = (0, 255, 255)
    else:
        status = "STOP"
        status_color = (0, 0, 255)

    # Label left of circle, status below circle
    cv2.putText(frame, "B", (light_x - 35, light_y + 8), font, 0.8, (255, 255, 255), 2)
    draw_traffic_light(frame, light_x, light_y, light_color, 30)
    cv2.putText(frame, status, (light_x - 23, light_y + 52), font, 0.5, status_color, 2)

    return frame


# ──────────────────────────────────────────────
#  Fallback Cycle (pure timer, no smart detection)
# ──────────────────────────────────────────────
# When the master LIGHT command is lost, the slave continues cycling
# from the last known state with fixed durations.  Starting from the
# same state+remaining as the master means the two sides stay roughly
# in phase; the 2 s ALL_RED transitions absorb small timing skew.

def _fb_advance_from(start_state, elapsed):
    """Walk cycle from start_state, consuming elapsed seconds."""
    idx = _fb_find_index(start_state)
    remaining = _FB_CYCLE[idx][1] - elapsed
    while remaining <= 0:
        idx = (idx + 1) % len(_FB_CYCLE)
        remaining += _FB_CYCLE[idx][1]
    return _FB_CYCLE[idx][0], remaining, _FB_CYCLE[idx][2]

_FB_CYCLE = [
    # (state_name,  duration_s,  b_relay)
    ("GREEN_A",     15.0,        "RED"),
    ("YELLOW_A",     3.0,        "RED"),
    ("ALL_RED_A",    15.0,        "RED"),
    ("RED_YELLOW_B", 2.0,        "RED_YELLOW"),
    ("GREEN_B",     15.0,        "GREEN"),
    ("YELLOW_B",     3.0,        "YELLOW"),
    ("ALL_RED_B",    15.0,        "RED"),
    ("RED_YELLOW_A", 2.0,        "RED"),
]


def _fb_find_index(state_name):
    """Return index of *state_name* in _FB_CYCLE (0 = GREEN_A if unknown)."""
    for i, (name, _, _) in enumerate(_FB_CYCLE):
        if name == state_name:
            return i
    return 0


def _fb_advance(start_idx, overflow_s):
    """Walk the cycle by *overflow_s* seconds past *start_idx*.

    Returns (state_name, remaining_s, b_relay).
    """
    idx = start_idx
    remaining = overflow_s
    while remaining >= _FB_CYCLE[idx][1]:
        remaining -= _FB_CYCLE[idx][1]
        idx = (idx + 1) % len(_FB_CYCLE)
    name, dur, relay = _FB_CYCLE[idx]
    return name, dur - remaining, relay


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Traffic Slave Controller (Board B)")
    print("  Road B: local NPU detection")
    print("  Light: remote command from Board A via LoRa")
    print("=" * 60)

    # ── 1. Init NPU ──
    print("[NPU] Loading RKNN model...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"[NPU] model load failed, ret={ret}")
        sys.exit(ret)
    rknn.init_runtime()
    print("[NPU] NPU ready")

    post_processor = YOLOv8_Fast_PostProcess(
        conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)

    # ── 2. Init Camera ──
    print(f"[CAM] Opening camera: {CAMERA_SOURCE}")
    stream_reader = RTSPStreamReader(CAMERA_SOURCE).start()
    time.sleep(1.0)
    if stream_reader.cap is None or not stream_reader.cap.isOpened():
        print(f"[CAM] WARNING: camera not reachable now ({CAMERA_SOURCE})")
        print("[CAM]         will keep retrying in background; light control continues")
        # Do NOT exit: the background reader thread keeps reconnecting,
        # and the main loop keeps driving the traffic lights via fallback.
    else:
        print("[CAM] Camera ready")

    # ── 3. Init LoRa ──
    print("[LoRa] Initializing serial...")
    lora = LoRaSlave(SERIAL_PORT, BAUDRATE)
    if not lora.open():
        print("[LoRa] ERROR: Check LoRa module connection")
        stream_reader.stop()
        rknn.release()
        sys.exit(1)
    lora.start_receiver()
    print("[LoRa] Slave communication ready")

    # ── 3.5 Init RS485 Relay Output ──
    relay = RS485RelayController(RELAY_PORT, RELAY_BAUDRATE)
    if not relay.open():
        print("[Relay] ERROR: Check RS485 relay board connection")
        lora.close()
        stream_reader.stop()
        rknn.release()
        sys.exit(1)
    print("[Relay] B-side traffic light output ready")

    # ── 4. Main Loop ──
    frame_count = 0
    last_det_send = 0
    last_status_write = 0
    fps = 0.0
    fps_smooth = deque(maxlen=30)

    print("\n[MAIN] Starting slave loop (Ctrl+C to exit)")
    print("-" * 60)

    try:
        while True:
            frame_start = time.time()

            # ── Read camera ──
            ret, frame = stream_reader.read_latest()

            # ── Light control: ALWAYS run, independent of camera ──
            now = time.time()

            light_state, light_remaining, light_color, is_green, is_yellow, light_last = \
                lora.get_light_state()
            light_fresh = lora.is_light_data_fresh()

            # ── P0 Safety Fallback: timed cycle on master signal loss ──
            # When the LIGHT command from Board A is lost, the slave
            # continues cycling with fixed 30-s durations, seeded from
            # the last received state + remaining time.  Because both
            # master and slave start from the same point in the cycle,
            # the 2-s ALL_RED transitions absorb minor timing skew.
            # This matches the original traffic_master.py / traffic_slave.py
            # behaviour: one allocator runs, and when data stops updating
            # it naturally falls back to MAX_GREEN_TIME switching.
            if not light_fresh:
                # ---- FALLBACK ----
                if light_last == 0.0:
                    if lora.fb_start_time == 0.0:
                        lora.fb_start_time = now
                    fb_state, fb_remaining, _ = _fb_advance_from(
                        "GREEN_B", now - lora.fb_start_time)
                else:
                    if lora.fb_start_time == 0.0:
                        phase_dur = _FB_CYCLE[_fb_find_index(light_state)][1]
                        lora.fb_start_time = light_last - (phase_dur - light_remaining)
                    fb_state, fb_remaining, _ = _fb_advance_from(
                        light_state, now - lora.fb_start_time)

                b_relay_map = {s[0]: s[2] for s in _FB_CYCLE}
                b_relay = b_relay_map.get(fb_state, "RED")
                light_state = fb_state
                light_remaining = fb_remaining
                if b_relay == "GREEN":
                    light_color = (0, 255, 0); is_green = True; is_yellow = False
                    relay.set_light("GREEN")
                elif b_relay == "YELLOW":
                    light_color = (0, 255, 255); is_green = False; is_yellow = True
                    relay.set_light("YELLOW")
                elif b_relay == "RED_YELLOW":
                    light_color = (0, 165, 255); is_green = False; is_yellow = True
                    relay.set_light("RED_YELLOW")
                else:
                    light_color = (0, 0, 255); is_green = False; is_yellow = False
                    relay.set_light("RED")
            else:
                # ---- LIGHT MODE ----
                lora.fb_start_time = 0.0
                if light_state == "GREEN_B":
                    relay.set_light("GREEN")
                elif light_state == "YELLOW_B":
                    relay.set_light("YELLOW")
                elif light_state == "RED_YELLOW_B":
                    relay.set_light("RED_YELLOW")
                else:
                    relay.set_light("RED")

            # ── Periodically read actual relay status (0.5s) → file ──
            if now - last_status_write > 0.5:
                last_status_write = now
                actual = relay.read_status() if relay is not None else None
                try:
                    with open("/tmp/relay_actual.json", "w") as f:
                        f.write(json.dumps({
                            "actual": actual,
                            "state": light_state,
                            "remaining": light_remaining,
                            "t": now,
                        }))
                except Exception:
                    pass

            # ── No frame yet / camera down: keep driving lights, skip vision ──
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            frame_count += 1
            ori_h, ori_w = frame.shape[:2]

            # ── NPU inference ──
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            outputs = rknn.inference(inputs=[img])
            detections = post_processor.process(outputs, ori_w, ori_h)

            # ── Count Road B vehicles ──
            class_counts = {name: 0 for name in TARGET_CLASS_NAMES}
            for det in detections:
                cls_name = det['class']
                conf_score = det['conf']
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                    # Draw bounding box
                    x, y_b, w_box, h_box = det['box']
                    label = f"{cls_name} {conf_score:.2f}"
                    cv2.rectangle(frame, (x, y_b), (x + w_box, y_b + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(y_b - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            total_count = sum(class_counts.values())

            # ── Send detection to Board A (prime interval to avoid LoRa collision)
            if now - last_det_send > 0.737:
                lora.send_detection(class_counts)
                last_det_send = now

            draw_fresh = True
            frame = draw_slave_display(
                frame, detections, class_counts, total_count,
                light_state, light_remaining, light_color,
                is_green, draw_fresh, fps)

            # ── Show ──
            show_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Traffic Slave (Board B)", show_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ── FPS ──
            elapsed = time.time() - frame_start
            if elapsed > 0:
                fps_smooth.append(1.0 / elapsed)
            fps = np.mean(fps_smooth) if fps_smooth else 0

            # ── Status print (every 30 frames) ──
            if frame_count % 30 == 0:
                if light_fresh:
                    l_info = f"Light={light_state}"
                elif light_state == "NO_SIGNAL":
                    l_info = "Light=NO_SIGNAL"
                else:
                    l_info = f"FALLBACK={light_state}"
                print(f"  B_count={total_count} {l_info} "
                      f"remaining={light_remaining:.1f}s FPS={fps:.1f}")

    except KeyboardInterrupt:
        print("\n[MAIN] User interrupt")
    finally:
        relay.close()
        lora.close()
        stream_reader.stop()
        rknn.release()
        cv2.destroyAllWindows()
        print("[MAIN] Resources released, exiting")


if __name__ == '__main__':
    main()
