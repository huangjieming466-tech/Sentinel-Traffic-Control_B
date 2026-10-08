import paho.mqtt.client as mqtt
import time
import json
import signal
import sys
import os

# ======================================================================
# 智能红绿灯 · B 板上报脚本（真实灯态版）
# 从主程序写出的 /tmp/relay_actual.json 读取真实继电器灯态，MQTT 发布到服务器
# 不碰串口，不影响主程序
# ======================================================================

MQTT_SERVER = "129.226.138.87"
MQTT_PORT   = 1883
LIGHT_ID    = "TL-NEXT-002"          # B 板设备 ID
TOPIC       = f"traffic/light/{LIGHT_ID}/state"

STATUS_FILE = "/tmp/relay_actual.json"   # 主程序每秒写入的真实灯态

client = mqtt.Client(client_id=LIGHT_ID, clean_session=False)


def on_connect(_c, _userdata, _flags, rc):
    code = {
        0: "连接成功",
        1: "协议版本错误",
        2: "Client ID 非法",
        3: "服务器不可用",
        4: "用户名/密码错误",
        5: "未授权",
    }.get(rc, f"未知错误 rc={rc}")
    print(f"[MQTT] {code} (rc={rc})")


def on_disconnect(_c, _userdata, rc):
    if rc != 0:
        print(f"[MQTT] 异常断开 rc={rc}，自动重连中...")


client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.connect_async(MQTT_SERVER, MQTT_PORT, keepalive=15)
client.loop_start()


def sigint_handler(_signum, _frame):
    print("\n[EXIT] 正在断开 MQTT ...")
    client.loop_stop()
    client.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

print(f"[启动] 设备 ID = {LIGHT_ID}")
print(f"[启动] Topic   = {TOPIC}")
print(f"[启动] Broker  = {MQTT_SERVER}:{MQTT_PORT}")
print(f"[启动] 读取    = {STATUS_FILE}\n")

# 灯态映射：继电器实际灯态 -> 上报 color
COLOR_MAP = {
    "GREEN":      "green",
    "YELLOW":     "yellow",
    "RED_YELLOW": "red_yellow",
    "RED":        "red",
    "OFF":        "off",
}

last_color = None

while True:
    color = None
    actual = None
    remaining = 0
    total = 0
    state = "UNKNOWN"
    online = False

    try:
        with open(STATUS_FILE) as f:
            data = json.load(f)
        actual = data.get("actual")
        color = COLOR_MAP.get(actual)
        remaining = int(data.get("remaining", 0))
        total = int(data.get("total", 0))
        state = data.get("state", "UNKNOWN")
        online = True
    except Exception:
        pass  # 文件还没生成 / 读取失败，用默认值（red）

    payload = {
        "id":               LIGHT_ID,
        "color":            color,
        "actual":           actual,
        "feedbackAvailable": color is not None,
        "remainingSeconds": remaining,
        "totalSeconds":     total,
        "state":            state,
        "isOnline":         online,
    }

    try:
        info = client.publish(
            TOPIC,
            json.dumps(payload, ensure_ascii=False),
            qos=1,
            retain=True,
        )
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            if color != last_color:
                print(f"🚦 变灯 → {(color or 'UNKNOWN').upper():<6s}  倒计时 {remaining}/{total}s  "
                      f"state={state}  t={time.strftime('%H:%M:%S')}")
            else:
                print(f"   · heartbeat  {(color or 'unknown'):<6s}  {remaining}/{total}s  "
                      f"t={time.strftime('%H:%M:%S')}")
        else:
            print(f"[ERR] 发布失败 rc={info.rc}")
    except Exception as e:
        print(f"[ERR] 发布异常: {e}")

    last_color = color
    time.sleep(1)
