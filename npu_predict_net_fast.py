import os
import sys
import threading
import time

# =========================================================
# 🤐 彻底关闭所有底层硬件和图形库的话痨警告
# =========================================================
os.environ["GLOG_minloglevel"] = "3"
os.environ["RKNN_LOG_LEVEL"] = "0" 
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["DISPLAY"] = ":1" 

import cv2
import numpy as np
from rknnlite.api import RKNNLite 

# =========================================================
# ⚡ YOLOv8 矩阵向量化超高速后处理（干掉 for 循环）
# =========================================================
class YOLOv8_Fast_PostProcess:
    def __init__(self, conf_threshold=0.35, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.strides = [8, 16, 32]
        self.reg_max = 16
        
        # 标准 COCO 80 类
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                        'cell phone', 'microwave', 'overn', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def process(self, outputs, ori_w, ori_h):
        all_boxes, all_confs, all_class_ids = [], [], []
        
        for i in range(3):
            stride = self.strides[i]
            # 拿到当前尺度的 3 个输出分支
            box_head = np.squeeze(outputs[i * 3])          # (64, H, W)
            cls_head = np.squeeze(outputs[i * 3 + 1])      # (80, H, W)
            score_head = np.squeeze(outputs[i * 3 + 2])    # (1, H, W)
            
            _, h, w = box_head.shape
            
            # 1. 矩阵变形与对齐
            box_head = box_head.reshape(64, -1).T          # (H*W, 64)
            cls_head = cls_head.reshape(80, -1).T          # (H*W, 80)
            score_head = score_head.reshape(-1, 1)         # (H*W, 1)
            
            # 2. 向量化计算当前尺度下所有格子的类别概率
            scores = cls_head * score_head                  # (H*W, 80)
            max_scores = np.max(scores, axis=1)            # 每个网格最强物体的得分
            class_ids = np.argmax(scores, axis=1)          # 每个网格对应的类别 ID
            
            # 3. 过滤掉低于置信度阈值的多余网格（瞬间过滤掉 99% 的无效背景）
            keep_idx = max_scores > self.conf_threshold
            if not np.any(keep_idx):
                continue
                
            filtered_boxes_raw = box_head[keep_idx]
            filtered_scores = max_scores[keep_idx]
            filtered_class_ids = class_ids[keep_idx]
            
            # 4. 向量化 DFL (Distance Focal Loss) 解码
            num_kept = filtered_boxes_raw.shape[0]
            filtered_boxes_raw = filtered_boxes_raw.reshape(num_kept, 4, self.reg_max)
            filtered_boxes_raw = self._softmax(filtered_boxes_raw, axis=-1)
            acc_matrix = np.arange(self.reg_max, dtype=np.float32)
            box_decoded = np.sum(filtered_boxes_raw * acc_matrix, axis=-1) * stride # (num_kept, 4)
            
            # 5. 计算当前过滤网格对应的网格物理坐标阵
            grid_y, grid_x = np.indices((h, w), dtype=np.float32)
            grid_x = grid_x.flatten()[keep_idx]
            grid_y = grid_y.flatten()[keep_idx]
            
            anchor_x = (grid_x + 0.5) * stride
            anchor_y = (grid_y + 0.5) * stride
            
            # 6. 一次性算出左上角和宽高
            x1 = anchor_x - box_decoded[:, 0]
            y1 = anchor_y - box_decoded[:, 1]
            x2 = anchor_x + box_decoded[:, 2]
            y2 = anchor_y + box_decoded[:, 3]
            
            # 映射回图像的物理分辨率
            rx1 = (x1 * (ori_w / 640.0)).astype(np.int32)
            ry1 = (y1 * (ori_h / 640.0)).astype(np.int32)
            rw = ((x2 - x1) * (ori_w / 640.0)).astype(np.int32)
            rh = ((y2 - y1) * (ori_h / 640.0)).astype(np.int32)
            
            all_boxes.extend(np.stack([rx1, ry1, rw, rh], axis=1).tolist())
            all_confs.extend(filtered_scores.astype(float).tolist())
            all_class_ids.extend(filtered_class_ids.tolist())
            
        # 7. 全局非极大值抑制（NMS）
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

# =========================================================
# 🧵 无延迟后台网络流刷新线程
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
# 🚀 启动硬件加速流
# =========================================================
rknn = RKNNLite()
print("🔄 正在加载超高速 YOLOv8 RKNN 模型...")
ret = rknn.load_rknn('/home/hkcrc2/Sentinel-Traffic-Control_B/yolov8n.rknn') 
if ret != 0:
    print("❌ 模型加载失败")
    exit(ret)

rknn.init_runtime()
print("🟢 硬件与多线程向量化管道就绪！")

post_processor = YOLOv8_Fast_PostProcess(conf_threshold=0.35, iou_threshold=0.45)

# 🔔 填写你的网络摄像头网络流地址（如果是USB摄像头则写 0）
rtsp_url = "rtsp://admin:HKcrc3130@192.168.1.64:554/h264/ch1/main/av_stream" 
# rtsp_url = 0 
cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)

# 强行限缩缓冲区并开启多线程抓包
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
stream_reader = RTSPStreamReader(cap).start()
time.sleep(1.0) # 等待视频流握手稳定

while True:
    start_time = time.time()
    
    # 异步抢占：永远只拿最新鲜的、刚捕获到的那一帧
    ret, frame = stream_reader.read_latest()
    if not ret or frame is None:
        continue
    
    ori_h, ori_w = frame.shape[:2]
        
    # 输入预处理
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) 
    
    # NPU 全速前向推理
    outputs = rknn.inference(inputs=[img])
    
    # ⚡ 调用毫秒级向量化矩阵后处理
    detections = post_processor.process(outputs, ori_w, ori_h)
    
    # 画框及分类计数
    class_counts = {}
    for det in detections:
        obj_class = det['class']
        conf_score = det['conf']
        class_counts[obj_class] = class_counts.get(obj_class, 0) + 1
        
        x, y, w, h = det['box']
        label = f"{obj_class}_{class_counts[obj_class]} {conf_score:.2f}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if class_counts:
        summary = ", ".join([f"{k}: {v}个" for k, v in class_counts.items()])
        print(f"当前画面检测到 {summary}")
        
    # 帧率计算与绘制
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    show_fame=cv2.resize(frame,(1280,720))
    cv2.imshow("OrangePi NPU Run", show_fame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
rknn.release()