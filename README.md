# Sentinel Traffic Control

双板协同智能交通灯控制系统。基于 Rockchip NPU + YOLOv8 + LoRa 通信。

## 架构

```
┌─────────────────────────┐       LoRa        ┌─────────────────────────┐
│  板 A — Master          │◄─────────────────►│  板 B — Slave           │
│                         │                    │                         │
│                         │  B → A: DET:car:.. │                         │
│  USB摄像头 → NPU 检测    │  A → B: LIGHT:...  │  USB摄像头 → NPU 检测    │
│  统计 Road A 车辆数      │                    │  统计 Road B 车辆数      │
│           ↓             │                    │           ↓             │
│  ┌──────────────────┐   │                    │  上报检测结果给 A         │
│  │ TrafficLightAllocator│                    │  接收红绿灯指令并显示     │
│  │ 中位数滤波 + 状态机  │                    │                         │
│  │ → 决定哪条路绿灯    │                    │                         │
│  └──────────────────┘   │                    │                         │
│           ↓             │                    │           ↓             │
│  屏幕显示 + 发送指令给B   │                    │  屏幕显示红绿灯状态       │
└─────────────────────────┘                    └─────────────────────────┘
```

## 文件说明

| 文件 | 部署位置 | 说明 |
|------|---------|------|
| `traffic_master.py` | 板 A | 主控程序：NPU 检测 + 接收 B 数据 + 红绿灯决策 + 发送指令 |
| `traffic_slave.py` | 板 B | 从属程序：NPU 检测 + 上报数据 + 接收指令 + 显示灯态 |
| `traffic_light_allocator.py` | 两板共用 | 核心分配器：Patience Score 算法 + 中位数滤波 + 状态机 |

## 运行

**板 A：**
```bash
cd Sentinel-Traffic-Control_A
python3 traffic_master.py
```

**板 B：**
```bash
cd Sentinel-Traffic-Control_B
python3 traffic_slave.py
```

按 `q` 退出，按 `r` 重置状态机。

## 红绿灯决策逻辑

```
GREEN_A → YELLOW_A → ALL_RED → GREEN_B → YELLOW_B → ALL_RED → GREEN_A
```

- **Patience Score**：`车辆数 + 等待时间/10`，等得越久分数越高
- **切换条件**：对方路拥堵（score > 本方×1.5 且持续 3 秒）
- **强制切换**：单次绿灯最长 30 秒
- **中位数滤波**：5 帧窗口过滤 YOLO 偶发漏检

## LoRa 通信协议

| 方向 | 格式 | 说明 |
|------|------|------|
| B → A | `DET:car:2,motorcycle:0,bus:1,truck:1` | B 板检测数据 |
| A → B | `LIGHT:GREEN_A,25` | 红绿灯指令 + 剩余秒数 |

## 硬件

- Rockchip RK3588 开发板 ×2
- LoRa 透传模块 (FT232, /dev/ttyUSB0, 115200)
- YOLOv8n RKNN 模型 (NPU)
- USB 摄像头 / 海康网络摄像头 (RTSP)
