import cv2
import sys

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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow(window_name, frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
