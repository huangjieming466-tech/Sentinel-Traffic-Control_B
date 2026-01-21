import cv2
import numpy as np
import time
from ultralytics import YOLO

def main():
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')

    # Create a 640x640 black image (3 channels)
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    print("Starting benchmark (50 iterations)...")
    
    # Warmup (optional, but recommended to stabilize the model first run)
    # model.predict(img, verbose=False) 

    total_time = 0
    num_runs = 50

    for i in range(num_runs):
        start_time = time.time()
        
        # Run inference
        model.predict(img, verbose=False)
        
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        
        # Optional: Print progress every 10 runs
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_runs} runs.")

    avg_time = total_time / num_runs
    avg_fps = 1 / avg_time if avg_time > 0 else 0

    print(f"\nBenchmark Results:")
    print(f"Total time for {num_runs} runs: {total_time:.4f} seconds")
    print(f"Average time per inference: {avg_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

if __name__ == '__main__':
    main()
