import os
import sys
# ================= 核心配置区 (新增) =================
# 1. 告诉 Qt 使用 xcb (解决 wayland 报错)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 2. 如果系统没有指定显示器(比如SSH环境)，强制指定到 HDMI 屏幕 (:0)
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
# ===================================================
import cv2
import time
from ultralytics import YOLO

def main():
    # Load the YOLOv8n model
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n_ncnn_model')

    # Open the USB camera (usually index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Force MJPG video encoding
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Target classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    # COCO dataset class indices
    target_classes = [2, 3, 5, 7]

    print("Starting inference... Press 'q' to exit.")

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Start timer for inference (optional, but FPS calculation below covers total loop time)
        
        # Run inference on the frame
        # stream=True is efficient for video sources
        # classes argument filters the detections
        results = model.predict(frame, classes=target_classes, verbose=False)

        # Visualize the results on the frame
        # plot() draws bounding boxes and labels
        annotated_frame = results[0].plot()

        # --- Traffic Light Logic ---
        # ROI Definition: Bottom Middle 300x200
        # Frame size: 640x480
        roi_x1 = 170  # 320 - 150
        roi_y1 = 280  # 480 - 200
        roi_x2 = 470  # 320 + 150
        roi_y2 = 480
        
        vehicle_in_roi = False
        
        # Check if any vehicle center is in ROI
        if results[0].boxes:
            for box in results[0].boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center point
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Check if center is within ROI
                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    vehicle_in_roi = True
                    break
        
        # Draw ROI and Status
        if vehicle_in_roi:
            color = (0, 0, 255) # Red (BGR)
            status_text = "STATUS: BUSY (RED LIGHT)"
        else:
            color = (0, 255, 0) # Green (BGR)
            status_text = "STATUS: CLEAR (GREEN LIGHT)"
            
        # Draw ROI rectangle
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), color, 2)
        # Draw Status text
        cv2.putText(annotated_frame, status_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # ---------------------------

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Display FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('YOLOv8 Vehicle Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
