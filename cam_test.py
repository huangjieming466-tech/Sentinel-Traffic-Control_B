import cv2
import sys
from ultralytics import YOLO
import time

def main():
    cap = None
    opened_index = -1

    # 尝试打开 index 0 到 5
    print("Searching for camera (index 0-5)...")
    for i in range(6):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            # 尝试读取一帧以确认摄像头真正可用
            ret, _ = temp_cap.read()
            if ret:
                print(f"✅ Camera opened at index {i}")
                cap = temp_cap
                opened_index = i
                break
            else:
                temp_cap.release()
        
    if cap is None:
        print("❌ Could not open any camera (index 0-5)")
        return

    window_name = "Sentinel Eye"
    
    print("Starting video feed. Press 'q' to exit.")
    
    model = YOLO("yolov8n.pt")  # 加载预训练的 YOLOv8n 模型
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
     
        results = model(
            source=frame,  # 输入当前帧
            stream=False,  # 启用流式推理
        )
        if results is not None and len(results) > 0:
            
            cv2.imshow(window_name, results[0].plot())  # 显示带检测框的画面
            
        else:
            cv2.imshow(window_name, frame)  # 显示原始画面

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
