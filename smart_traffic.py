import cv2
import time
import numpy as np
from ultralytics import YOLO
from enum import Enum

# 定义交通灯状态
class TrafficState(Enum):
    GREEN_A = 1   # A路绿灯，B路红灯
    YELLOW_A = 2  # A路黄灯，B路红灯
    ALL_RED_A = 5 # A黄变A红后的全红
    GREEN_B = 3   # B路绿灯，A路红灯
    YELLOW_B = 4  # B路黄灯，A路红灯
    ALL_RED_B = 6 # B黄变B红后的全红

def init_camera(index_candidates):
    """
    尝试打开摄像头，支持传入候选索引列表
    """
    if isinstance(index_candidates, int):
        index_candidates = [index_candidates]
        
    for idx in index_candidates:
        print(f"尝试打开摄像头 index={idx} ...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # 强制设置为 640x480 和 MJPG
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print(f"成功打开摄像头 index={idx}")
            return cap
        
    print(f"错误: 无法打开任何摄像头: {index_candidates}")
    return None

def draw_traffic_info(frame, label, count, score, state_color, remaining_time, is_green_active):
    """
    在画面上绘制交通灯状态和车辆统计
    state_color: (B, G, R) 元组
    """
    # 绘制半透明背景
    # 提取左上角 ROI 区域 (用于显示文字)
    h_bg, w_bg = 90, 200
    roi = frame[0:h_bg, 0:w_bg]
    
    # 创建黑色遮罩并混合
    # 目标效果: 原图变暗 (原图 * 0.6)
    # 使用 zeros_like 创建全黑图，然后 addWeighted
    black_rect = np.zeros_like(roi)
    res = cv2.addWeighted(roi, 0.6, black_rect, 0.4, 0)
    
    # 将处理后的半透明区域放回原图
    frame[0:h_bg, 0:w_bg] = res
    
    # 显示路口名称和车辆数
    cv2.putText(frame, f"{label} Count: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # 显示分数
    cv2.putText(frame, f"Score: {score:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 绘制模拟交通灯 (实心圆)
    # 位置：画面右上角
    h, w = frame.shape[:2]
    light_pos = (w - 50, 50)
    radius = 30
    
    # 绘制灯
    cv2.circle(frame, light_pos, radius, state_color, -1)
    # 绘制灯的边框
    cv2.circle(frame, light_pos, radius, (255, 255, 255), 2)
    
    # 显示倒计时 (在灯的下方)
    time_text = f"{int(remaining_time)}s"
    cv2.putText(frame, time_text, (w - 70, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 如果是绿灯，显示 GO，红灯显示 STOP
    status_text = "GO" if is_green_active else "STOP"
    text_color = (0, 255, 0) if is_green_active else (0, 0, 255)
    cv2.putText(frame, status_text, (w - 80, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

def main():
    # 1. 加载模型 (NCNN模式)
    # 注意：请确保当前目录下有 yolov8n_ncnn_model 文件夹
    # 如果没有，可以使用 model.export(format='ncnn') 导出，或者暂时使用 yolov8n.pt
    print("正在加载 YOLOv8n NCNN 模型...")
    try:
        model = YOLO('yolov8n_ncnn_model')
    except Exception as e:
        print(f"加载 NCNN 模型失败，尝试加载 yolov8n.pt: {e}")
        model = YOLO('yolov8n.pt')

    # 2. 初始化双摄
    # A路: 尝试 index 0
    cap_a = init_camera([0])
    # B路: 尝试 index 2, 如果失败尝试 1
    cap_b = init_camera([2, 1])

    if cap_a is None or cap_b is None:
        print("错误: 无法打开两个摄像头。请检查连接。")
        if cap_a: cap_a.release()
        if cap_b: cap_b.release()
        return

    # 目标类别 (COCO)
    target_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck

    # 状态机参数
    MIN_GREEN_TIME = 5.0
    MAX_GREEN_TIME = 30.0
    YELLOW_TIME = 3.0
    ALL_RED_TIME = 2.0 # 全红清空时间
    
    # 初始状态
    current_state = TrafficState.GREEN_A
    state_start_time = time.time()
    
    # 去抖动计时器
    # 记录拥堵条件首次满足的时间
    condition_met_start_time = 0
    DEBOUNCE_TIME = 3.0 # 连续满足3秒才切换
    
    print("开始双路智能交通灯控制。按 'q' 退出。")

    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()

        if not ret_a or not ret_b:
            print("错误: 读取视频帧失败。")
            break

        # 3. 双路推理
        # 使用 stream=True 提高性能
        results_a = model.predict(frame_a, classes=target_classes, verbose=False, stream=True)
        results_b = model.predict(frame_b, classes=target_classes, verbose=False, stream=True)
        
        # 解析结果 A
        count_a = 0
        annotated_a = frame_a.copy()
        for r in results_a:
            annotated_a = r.plot()
            count_a += len(r.boxes)
            
        # 解析结果 B
        count_b = 0
        annotated_b = frame_b.copy()
        for r in results_b:
            annotated_b = r.plot()
            count_b += len(r.boxes)

        # 4. 状态机逻辑
        current_time = time.time()
        elapsed_time = current_time - state_start_time

        # 默认颜色
        color_a = (0, 0, 255) # Red
        color_b = (0, 0, 255) # Red
        is_green_a = False
        is_green_b = False
        remaining = 0
        
        # 计算分数 (Patience Score)
        # Score = Count + (Wait_Time / 10)
        # 如果是绿灯，Wait_Time = 0
        score_a = count_a + (elapsed_time / 10.0 if current_state in [TrafficState.GREEN_B, TrafficState.YELLOW_B] else 0)
        score_b = count_b + (elapsed_time / 10.0 if current_state in [TrafficState.GREEN_A, TrafficState.YELLOW_A] else 0)
        
        if current_state == TrafficState.GREEN_A:
            color_a = (0, 255, 0) # Green
            is_green_a = True
            
            # 逻辑判断
            should_switch = False
            
            # 正常逻辑
            # 必须满足最小绿灯时间
            if elapsed_time > MIN_GREEN_TIME:
                # 条件1: 达到最大绿灯时间 (无条件切换)
                if elapsed_time > MAX_GREEN_TIME:
                    should_switch = True
                    condition_met_start_time = 0 # 重置去抖动
                else:
                    # 条件2: 权重分数判定 (Weighted Scoring)
                    # B路分数 > A路分数 * 1.5
                    # 且 B路有车 (>3 阈值)
                    congestion_condition = (count_b > 3) and (score_b > score_a * 1.5)
                    
                    if congestion_condition:
                        if condition_met_start_time == 0:
                            condition_met_start_time = current_time
                        elif current_time - condition_met_start_time > DEBOUNCE_TIME:
                            # 连续满足去抖动时间，允许切换
                            should_switch = True
                            condition_met_start_time = 0 # 重置
                    else:
                        # 条件不满足，重置计时器
                        condition_met_start_time = 0
            
            remaining = MAX_GREEN_TIME - elapsed_time

            if should_switch:
                current_state = TrafficState.YELLOW_A
                state_start_time = current_time
                elapsed_time = 0
                condition_met_start_time = 0 # 确保重置
            
            if remaining < 0: remaining = 0

        elif current_state == TrafficState.YELLOW_A:
            color_a = (0, 255, 255) # Yellow
            
            # 黄灯不受紧急锁定控制，必须走完流程
            if elapsed_time > YELLOW_TIME:
                current_state = TrafficState.ALL_RED_A
                state_start_time = current_time
                elapsed_time = 0
                condition_met_start_time = 0 # 重置去抖动
            
            remaining = YELLOW_TIME - elapsed_time

        elif current_state == TrafficState.ALL_RED_A:
            # 全红状态：两边都是红灯
            color_a = (0, 0, 255) # Red
            color_b = (0, 0, 255) # Red
            
            if elapsed_time > ALL_RED_TIME:
                current_state = TrafficState.GREEN_B
                state_start_time = current_time
                elapsed_time = 0
            
            remaining = ALL_RED_TIME - elapsed_time

        elif current_state == TrafficState.GREEN_B:
            color_b = (0, 255, 0) # Green
            is_green_b = True
            
            should_switch = False
            
            if elapsed_time > MIN_GREEN_TIME:
                if elapsed_time > MAX_GREEN_TIME:
                    should_switch = True
                    condition_met_start_time = 0
                else:
                    # 权重分数判定
                    # A路分数 > B路分数 * 1.5
                    congestion_condition = (count_a > 3) and (score_a > score_b * 1.5)
                    
                    if congestion_condition:
                        if condition_met_start_time == 0:
                            condition_met_start_time = current_time
                        elif current_time - condition_met_start_time > DEBOUNCE_TIME:
                            should_switch = True
                            condition_met_start_time = 0
                    else:
                        condition_met_start_time = 0
            
            remaining = MAX_GREEN_TIME - elapsed_time
            
            if should_switch:
                current_state = TrafficState.YELLOW_B
                state_start_time = current_time
                elapsed_time = 0
                condition_met_start_time = 0
                
            if remaining < 0: remaining = 0

        elif current_state == TrafficState.YELLOW_B:
            color_b = (0, 255, 255) # Yellow
            
            if elapsed_time > YELLOW_TIME:
                current_state = TrafficState.ALL_RED_B
                state_start_time = current_time
                elapsed_time = 0
                condition_met_start_time = 0
            
            remaining = YELLOW_TIME - elapsed_time

        elif current_state == TrafficState.ALL_RED_B:
            # 全红状态：两边都是红灯
            color_a = (0, 0, 255) # Red
            color_b = (0, 0, 255) # Red
            
            if elapsed_time > ALL_RED_TIME:
                current_state = TrafficState.GREEN_A
                state_start_time = current_time
                elapsed_time = 0
            
            remaining = ALL_RED_TIME - elapsed_time

        # 5. UI 绘制
        # 绘制 A 路信息
        draw_traffic_info(annotated_a, "Road A", count_a, score_a, color_a, remaining if (current_state in [TrafficState.GREEN_A, TrafficState.YELLOW_A]) else 0, is_green_a)
        
        # 绘制 B 路信息
        draw_traffic_info(annotated_b, "Road B", count_b, score_b, color_b, remaining if (current_state in [TrafficState.GREEN_B, TrafficState.YELLOW_B]) else 0, is_green_b)

        # 6. 拼接显示
        # 横向拼接
        combined_frame = np.hstack((annotated_a, annotated_b))
        
        cv2.imshow('Smart Traffic Control System', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_a.release()
    cap_b.release()
    cv2.destroyAllWindows()
    print("系统已退出。")

if __name__ == '__main__':
    main()
