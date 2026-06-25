#!/usr/bin/env python3
"""
LoRa Detect Recv — Server B (接收端)
从 LoRa 串口接收 Server A 的检测结果，解析并显示，自动回复 ACK

复用:
  - lora_pingtest.py 的串口通信模式

协议:
  A -> B:  DET:car:3,motorcycle:1,bus:0,truck:2
  B -> A:  ACK
"""

import serial
import threading
import time
import sys

# ==================== 可配置参数 ====================

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200


# =========================================================
#  LoRa 接收端
# =========================================================
class LoRaReceiver:
    """接收端: 监听 DET 消息，解析并打印，自动回复 ACK"""

    def __init__(self, port, baudrate, timeout=2):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            print(f"[LoRa] 串口 {self.port} 打开成功 (baudrate={self.baudrate})")
            return True
        except Exception as e:
            print(f"[LoRa] 错误: 无法打开串口 {self.port}: {e}")
            return False

    def start(self):
        """启动接收线程"""
        self.running = True
        t = threading.Thread(target=self._receive_loop, daemon=True)
        t.start()

    def _receive_loop(self):
        while self.running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw = self.ser.readline()
                    if raw:
                        msg = raw.decode('utf-8', errors='ignore').strip()
                        self._handle_message(msg)
            except Exception as e:
                print(f"[LoRa] 接收异常: {e}")
            time.sleep(0.05)

    def _handle_message(self, msg):
        """解析并处理收到的消息"""
        timestamp = time.strftime('%H:%M:%S')

        if msg.startswith("DET:"):
            # 解析 DET:car:3,motorcycle:1,bus:0,truck:2
            data = msg[4:]  # 去掉 "DET:" 前缀
            pairs = data.split(",")

            print(f"\n{'=' * 50}")
            print(f"[{timestamp}] <<< 收到 Server A 检测数据 <<<")
            print(f"{'=' * 50}")

            total = 0
            detections = {}
            for pair in pairs:
                if ":" in pair:
                    cls_name, count_str = pair.split(":", 1)
                    try:
                        count = int(count_str)
                    except ValueError:
                        count = 0
                    detections[cls_name] = count
                    total += count
                    
                    print(f"  {cls_name:>12s}: {count}")

            print(f"  {'─' * 30}")
            print(f"  {'TOTAL':>12s}: {total} ")
            print(f"{'=' * 50}")

            # 自动回复 ACK
            self._send_ack()

        else:
            print(f"[{timestamp}] <- 未知消息: {msg}")

    def _send_ack(self):
        """发送 ACK 确认"""
        if self.ser:
            try:
                self.ser.write(b"ACK\n")
                ts = time.strftime('%H:%M:%S')
                print(f"[{ts}] >>> 已回复 ACK >>>")
            except Exception as e:
                print(f"[LoRa] 发送 ACK 失败: {e}")

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()
            print("[LoRa] 串口已关闭")


# =========================================================
#  主程序
# =========================================================
def main():
    print("=" * 60)
    print("  LoRa 检测接收端 (Server B)")
    print("=" * 60)

    receiver = LoRaReceiver(SERIAL_PORT, BAUDRATE)

    if not receiver.open():
        print("请检查:")
        print("  1. LoRa 模块是否已插入 USB")
        print("  2. 串口设备是否为 /dev/ttyUSB0")
        print("  3. 是否需要 sudo 权限")
        sys.exit(1)

    receiver.start()

    print("\n[MAIN] 正在监听 LoRa 检测数据... (按 Ctrl+C 退出)")
    print("[MAIN] 支持消息:")
    print("        DET:...  — 检测数据 (自动回复 ACK)")
    print("-" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] 用户中断")
    finally:
        receiver.close()
        print("[MAIN] 程序退出")


if __name__ == '__main__':
    main()
