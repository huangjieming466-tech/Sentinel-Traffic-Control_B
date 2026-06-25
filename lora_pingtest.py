import serial
import threading
import time
import sys

# 配置串口参数 (根据你的配置软件，波特率必须是 115200)
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2)
except Exception as e:
    print(f"错误: 无法打开串口 {SERIAL_PORT}。请检查是否被占用或缺少 sudo 权限: {e}")
    sys.exit(1)

# 接收线程：收到任何数据都打印出来，如果收到 PING 就自动回一个 PONG
def receive_thread():
    while True:
        try:
            if ser.in_waiting > 0:
                # 读取一行数据
                raw_data = ser.readline()
                if raw_data:
                    message = raw_data.decode('utf-8', errors='ignore').strip()
                    
                    # 收到对方的 PING，立刻自动回应 PONG
                    if message == "PING":
                        print(f"[{time.strftime('%H:%M:%S')}] 收到 -> PING (自动回应 PONG)")
                        ser.write(b"PONG\n")
                    # 收到对方响应的 PONG
                    elif message == "PONG":
                        print(f"[{time.strftime('%H:%M:%S')}] 收到来自远端的响应 <- PONG")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] 收到未知数据: {message}")
        except Exception as e:
            print(f"接收出错: {e}")
            break
        time.sleep(0.05)

def main():
    # 启动接收监听
    t = threading.Thread(target=receive_thread)
    t.daemon = True
    t.start()
    
    print("==================================================")
    print(" LoRa 无线 Ping 测试工具已就绪 ")
    print(" 指令说明: 输入 'p' 并回车手动发送单次 PING")
    print("          输入 'q' 并回车退出程序")
    print("==================================================")
    
    try:
        while True:
            user_input = input().strip().lower()
            
            if user_input == 'p':
                print(f"[{time.strftime('%H:%M:%S')}] 正在发送 -> PING...")
                ser.write(b"PING\n")
            elif user_input == 'q':
                break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("串口已关闭，程序退出。")

if __name__ == '__main__':
    main()