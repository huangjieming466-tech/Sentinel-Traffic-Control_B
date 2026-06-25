import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import cv2
import numpy as np
import ncnn  # 导入底层的 ncnn 库

# ==========================================
# 核心优化 1：开机时只加载一次 NCNN 网络（参考你的 model_ncnn.py）
# ==========================================
print("🔄 正在加载硬核 NCNN 底层模型...")
net = ncnn.Net()
# 读入你左侧文件树里那两个优化过的核心参数文件
net.load_param("yolov8n_ncnn_model/model.ncnn.param")
net.load_model("yolov8n_ncnn_model/model.ncnn.bin")
print("💾 NCNN 模型在香橙派上加载成功！")

# 摄像头配置
url = "rtsp://admin:HKcrc3130@192.168.1.64:554/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(url)

print("🟢 正在建立连接，画面将显示在香橙派本地屏幕上...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 读取画面失败")
        break
        
    # ==========================================
    # 核心优化 2：把摄像头当前帧（frame）喂给 NCNN
    # ==========================================
    # 1. 先把画面缩放到 YOLO 规定的 640x640 标准尺寸
    img_resized = cv2.resize(frame, (640, 640))
    
    # 2. 将 OpenCV 的 BGR 格式转换为 NCNN 需要的 Mat 矩阵格式
    # （这里的 2 代表 BGR2RGB 转换，根据你的模型训练情况决定是否开启）
    mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 640, 640)
    
    # 3. 填入你的 NCNN 减去均值、除以方差的归一化系数 (根据 YOLOv8 标准配置)
    # 如果你的模型训练时有特殊的归一化，可以在这里调整数字
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1/255.0, 1/255.0, 1/255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)
    
    # 4. 执行不卡顿的底层推理
    with net.create_extractor() as ex:
        # "in0" 和 "out0" 是你在 model_ncnn.py 第 15 和 17 行定义的输入输出节点名
        ex.input("in0", mat_in) 
        r, mat_out = ex.extract("out0")
        
        # 5. 获取推理出来的原始结果矩阵
        out_array = np.array(mat_out)
        
    # ==========================================
    # 📝 提示：拿到 out_array（检测框结果）后
    # 你可以在这里写后处理画框逻辑。为了测试流畅度，我们先直接看原生画面：
    # ==========================================
    
    # 缩放画面以适配你的物理屏幕
    screen_height = 720
    screen_width = 1280
    show_frame = cv2.resize(frame, (screen_width, screen_height))
    
    # 弹出实时画面窗口
    cv2.imshow("OrangePi Cam (NCNN Accelerating)", show_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("退出")