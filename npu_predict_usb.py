import os
# 强行屏蔽底层所有的黄字警告和图形界面警告
os.environ["GLOG_minloglevel"] = "3"
os.environ["RKNN_LOG_LEVEL"] = "0" 
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["DISPLAY"] = ":0" 

import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite 

# =========================================================
# 📦 YOLOv8 RKNN 多输出头（9个输出层）专用后处理工具类
# =========================================================
class YOLOv8_MultiHead_PostProcess:
    def __init__(self, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.strides = [8, 16, 32]  # 特征图对应的下采样步长
        self.reg_max = 16          # 边界框分布式预测最大值
        
        # 标准 80 类
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

    def _dfl(self, position):
        # Distance Focal Loss 解码：将 64 维转换为 4 维边界框偏移量
        position = position.reshape(-1, 4, self.reg_max)
        position = self._softmax(position, axis=-1)
        acc_matrix = np.arange(self.reg_max, dtype=np.float32)
        position = np.sum(position * acc_matrix, axis=-1)
        return position

    def process(self, outputs, ori_w, ori_h):
        boxes, confs, class_ids = [], [], []
        
        # 你的 outputs 有 9 个元素，每 3 个组成一个尺度层
        for i in range(3):
            # 获取当前特征图尺度的 3 个输出头数据
            box_head = np.squeeze(outputs[i * 3])          # (64, H, W)
            cls_head = np.squeeze(outputs[i * 3 + 1])      # (80, H, W)
            # 第三位视你的模型导出而定，一般为置信度或辅助锚点 score_head
            score_head = np.squeeze(outputs[i * 3 + 2])    # (1, H, W)
            
            stride = self.strides[i]
            grid_h, grid_w = box_head.shape[1], box_head.shape[2]
            
            # 转置为方便按像素位置排布的格式
            box_head = box_head.reshape(64, -1).T          # (H*W, 64)
            cls_head = cls_head.reshape(80, -1).T          # (H*W, 80)
            score_head = score_head.reshape(1, -1).T       # (H*W, 1)
            
            # 将 DFL 算子结果转换回常规宽高偏移量
            box_decoded = self._dfl(box_head) * stride      # (H*W, 4)
            
            # 遍历当前特征图上的每一个网络格子点（Grid Cell）
            for idx in range(grid_h * grid_w):
                row = idx // grid_w
                col = idx % grid_w
                
                # 结合 score_head 与 cls_head 算真正的类别得分
                scores = cls_head[idx] * score_head[idx][0]
                class_id = np.argmax(scores)
                score = scores[class_id]
                
                if score > self.conf_threshold:
                    # 计算当前锚点在 640x640 下的核心中心点
                    anchor_x = (col + 0.5) * stride
                    anchor_y = (row + 0.5) * stride
                    
                    # 解算边界框左上角和右下角坐标
                    x1 = anchor_x - box_decoded[idx][0]
                    y1 = anchor_y - box_decoded[idx][1]
                    x2 = anchor_x + box_decoded[idx][2]
                    y2 = anchor_y + box_decoded[idx][3]
                    
                    # 映射回摄像头的原图物理分辨率 (ori_w, ori_h)
                    rx1 = int(x1 * (ori_w / 640.0))
                    ry1 = int(y1 * (ori_h / 640.0))
                    rw = int((x2 - x1) * (ori_w / 640.0))
                    rh = int((y2 - y1) * (ori_h / 640.0))
                    
                    boxes.append([rx1, ry1, rw, rh])
                    confs.append(float(score))
                    class_ids.append(class_id)
                    
        # 运行非极大值抑制（NMS），清除同目标重叠的多余检测框
        indices = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.iou_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'conf': confs[i],
                    'class': self.classes[class_ids[i]]
                })
        return results

# =========================================================
# 🚀 主运行流程
# =========================================================
rknn = RKNNLite()
print("🔄 正在加载多头输出型 YOLOv8 专属 RKNN 模型...")
ret = rknn.load_rknn('/home/hkcrc2/Sentinel-Traffic-Control_B/yolov8n.rknn') 
if ret != 0:
    print("❌ 模型加载失败")
    exit(ret)

rknn.init_runtime()
print("🟢 NPU 硬件加速解包模块已就位！")

# 实例化多头专用后处理类
post_processor = YOLOv8_MultiHead_PostProcess(conf_threshold=0.35, iou_threshold=0.45)

# 启动 USB 摄像头
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    ori_h, ori_w = frame.shape[:2]
        
    # 输入预处理
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) 
    
    # NPU 高速推理获取 9 个独立矩阵
    outputs = rknn.inference(inputs=[img])
    
    # 🟢 核心修复：调用多头输出解析算法
    detections = post_processor.process(outputs, ori_w, ori_h)
    
    
    
    # 画框及打标签
    for det in detections:
        x, y, w, h = det['box']
        label = f"{det['class']} {det['conf']:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'检测到{label}',f'数量{len(detections)}')
    
    
    
    
    # 计算并显示帧率
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    
    cv2.imshow("OrangePi NPU Run", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
rknn.release()