import cv2

from ultralytics import YOLO
# 配置摄像头地址
url = "rtsp://admin:HKcrc3130@192.168.1.64:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(url)

print("🟢 正在建立连接，画面将显示在香橙派本地屏幕上...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 读取画面失败")
        break
        
    # === 这里以后可以放你的 YOLO 核心代码 ===
    # frame = model(frame) 
    
    # model=YOLO("yolov8n.pt")  # 加载预训练的 YOLOv8n 模型
    # results=model(
    
    # source=frame,  # 输入当前帧
    # stream=True,  # 启用流式推理
    # )
    
    
    # 对当前帧进行检测
    
    screen_height=720
    screen_width=1280
    show_frame=cv2.resize(frame, (screen_width, screen_height))
    
    # 在香橙派本地屏幕弹窗显示实时画面
    cv2.imshow("OrangePi Cam", show_frame)
    
    # 每帧等 1 毫秒，按键盘上的 'q' 键可以退出播放
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("退出")