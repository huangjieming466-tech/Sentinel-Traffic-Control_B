#!/usr/bin/env python3
"""
LoRa Detect Send — Server A (发送端)
NPU YOLO 检测 + LoRa 串口发送检测结果给 Server B

复用:
  - npu_predict_usb_fast.py 的 YOLOv8_Fast_PostProcess + RKNN 管线
  - lora_pingtest.py 的串口通信模式

协议:
  A -> B:  DET:car:3,motorcycle:1,bus:0,truck:2
  B -> A:  ACK
"""

import os
import sys
import threading
import time
import serial

# =========================================================
#  关闭底层 C++ 硬件/图形库的冗余警告
# =========================================================
os.environ["GLOG_minloglevel"] = "3"
os.environ["RKNN_LOG_LEVEL"] = "0"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["DISPLAY"] = ":0"

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ==================== 可配置参数 ====================

# 摄像头源: 0=USB摄像头, 或 RTSP 地址字符串
CAMERA_SOURCE = 0
# CAMERA_SOURCE = "rtsp://admin:HKcrc3130@192.168.1.64:554/h264/ch1/main/av_stream" 

# 模型路径
MODEL_PATH = "/home/hkcrc2/Sentinel-Traffic-Control_B/yolov8n.rknn"

# LoRa 串口
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

# 发送间隔 (秒)
SEND_INTERVAL = 2.0

# 检测置信度阈值
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# 目标类别 (COCO 索引)

# TARGET_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
# TARGET_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

TARGET_CLASS_IDS = [0]  # car, motorcycle, bus, truck
TARGET_CLASS_NAMES = ["person"]  # test 


# =========================================================
#  YOLOv8 后处理 (复用自 npu_predict_usb_fast.py)
# =========================================================
# 标准 COCO 80 类名
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


class YOLOv8_Fast_PostProcess:
    """向量化后处理 (复用自 npu_predict_usb_fast.py)"""

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
            box_head = np.squeeze(outputs[i * 3])       # (64, H, W)
            cls_head = np.squeeze(outputs[i * 3 + 1])   # (80, H, W)
            score_head = np.squeeze(outputs[i * 3 + 2]) # (1, H, W)

            _, h, w = box_head.shape

            box_head = box_head.reshape(64, -1).T       # (H*W, 64)
            cls_head = cls_head.reshape(80, -1).T       # (H*W, 80)
            score_head = score_head.reshape(-1, 1)      # (H*W, 1)

            scores = cls_head * score_head               # (H*W, 80)
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

        indices = cv2.dnn.NMSBoxes(all_boxes, all_confs,
                                   self.conf_threshold, self.iou_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': all_boxes[i],
                    'conf': all_confs[i],
                    'class': self.classes[all_class_ids[i]]
                })
        return results


# =========================================================
#  异步帧读取器 (复用自 npu_predict_usb_fast.py)
# =========================================================
class RTSPStreamReader:
    def __init__(self, capture):
        self.cap = capture
        self.ret = False
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                self.stopped = True

    def read_latest(self):
        return self.ret, self.frame


# =========================================================
#  LoRa 通信模块
# =========================================================
class LoRaComm:
    """LoRa 串口通信，带接收线程"""

    def __init__(self, port, baudrate, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.last_ack_time = 0
        self.ack_received = False
        self.running = False

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            print(f"[LoRa] 串口 {self.port} 打开成功 (baudrate={self.baudrate})")
            return True
        except Exception as e:
            print(f"[LoRa] 错误: 无法打开串口 {self.port}: {e}")
            return False

    def start_receiver(self):
        self.running = True
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()

    def _receive_loop(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw = self.ser.readline()
                    if raw:
                        msg = raw.decode('utf-8', errors='ignore').strip()
                        if msg == "ACK":
                            self.last_ack_time = time.time()
                            self.ack_received = True
                            print(f"  [LoRa] <- 收到 ACK (确认)")
                        else:
                            print(f"  [LoRa] <- 未知消息: {msg}")
            except Exception as e:
                print(f"  [LoRa] 接收异常: {e}")
            time.sleep(0.05)

    def send(self, message):
        """发送一行文本"""
        if self.ser:
            try:
                data = (message + "\n").encode('utf-8')
                self.ser.write(data)
                return True
            except Exception as e:
                print(f"  [LoRa] 发送失败: {e}")
                return False
        return False

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()
            print("[LoRa] 串口已关闭")


# =========================================================
#  主程序
# =========================================================
def main():
    print("=" * 60)
    print("  LoRa 检测发送端 (Server A)")
    print("=" * 60)

    # ---- 1. 初始化 NPU ----
    print("[NPU] 加载 RKNN 模型...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"[NPU] 模型加载失败, ret={ret}")
        sys.exit(ret)
    rknn.init_runtime()
    print("[NPU] 硬件推理引擎就绪")

    post_processor = YOLOv8_Fast_PostProcess(
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )

    # ---- 2. 初始化摄像头 ----
    print(f"[CAM] 打开摄像头: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[CAM] 错误: 无法打开摄像头 {CAMERA_SOURCE}")
        rknn.release()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    stream_reader = RTSPStreamReader(cap).start()
    time.sleep(1.0)  # 等待摄像头稳定
    print("[CAM] 摄像头就绪")

    # ---- 3. 初始化 LoRa ----
    print("[LoRa] 初始化串口...")
    lora = LoRaComm(SERIAL_PORT, BAUDRATE)
    if not lora.open():
        cap.release()
        rknn.release()
        sys.exit(1)
    lora.start_receiver()
    print("[LoRa] 接收线程已启动")

    # ---- 4. 主循环 ----
    last_send_time = 0
    frame_count = 0

    print("\n[MAIN] 开始检测 + LoRa 传输循环 (按 Ctrl+C 退出)")
    print("-" * 60)

    try:
        while True:
            frame_start = time.time()

            # 异步抓取最新帧
            ret, frame = stream_reader.read_latest()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            ori_h, ori_w = frame.shape[:2]

            # NPU 推理
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)

            outputs = rknn.inference(inputs=[img])
            detections = post_processor.process(outputs, ori_w, ori_h)

            # 统计目标类别数量 + 画框
            class_counts = {}
            for det in detections:
                cls_name = det['class']
                conf_score = det['conf']
                if cls_name in TARGET_CLASS_NAMES:
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    # 只画目标类别的检测框
                    x, y, w, h = det['box']
                    label = f"{cls_name} {conf_score:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 在画面上叠加检测统计
            y_offset = 30
            cv2.putText(frame, f"Detection Summary:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            for cls_name in TARGET_CLASS_NAMES:
                cnt = class_counts.get(cls_name, 0)
                y_offset += 25
                cv2.putText(frame, f"  {cls_name}: {cnt}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # LoRa 状态指示 (每帧都发送)
            cv2.putText(frame, "LoRa TX: LIVE", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # FPS
            fps = 1.0 / (time.time() - frame_start) if frame_count > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示画面
            show_frame = cv2.resize(frame, (1270, 720))
            cv2.imshow("LoRa Detect Send (Server A)", show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 终端日志 (每 1 帧打印一次)
            if frame_count % 1 == 0:
                if class_counts:
                    summary = ", ".join([f"{k}:{v}" for k, v in class_counts.items()])
                    print(f"  [DET] 检测到 -> {summary}")
                else:
                    print(f"  [DET] 当前画面无目标车辆")

            # 每帧发送检测结果
            parts = [f"{name}:{class_counts.get(name, 0)}"
                     for name in TARGET_CLASS_NAMES]
            det_msg = "DET:" + ",".join(parts)
            lora.send(det_msg)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[MAIN] 用户中断")

    finally:
        lora.close()
        cap.release()
        rknn.release()
        print("[MAIN] 资源已释放，程序退出")


if __name__ == '__main__':
    main()
