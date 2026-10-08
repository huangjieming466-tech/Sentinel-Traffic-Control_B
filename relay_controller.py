#!/usr/bin/env python3

import serial
import time


class RS485RelayController:
    """
    4-channel RS485 Modbus relay controller.

    Relay mapping:
        relay 0 = red
        relay 1 = yellow
        relay 2 = green
        relay 3 = spare

    Design:
        1. Only update last_state after commands succeed.
        2. Failed writes do NOT update last_state.
        3. Failed writes are retried.
        4. Before changing color, all relays are turned off first.
    """

    RELAY_COMMANDS = {
        0: {
            True:  bytes.fromhex("01 05 00 00 FF 00 8C 3A"),
            False: bytes.fromhex("01 05 00 00 00 00 CD CA"),
        },
        1: {
            True:  bytes.fromhex("01 05 00 01 FF 00 DD FA"),
            False: bytes.fromhex("01 05 00 01 00 00 9C 0A"),
        },
        2: {
            True:  bytes.fromhex("01 05 00 02 FF 00 2D FA"),
            False: bytes.fromhex("01 05 00 02 00 00 6C 0A"),
        },
        3: {
            True:  bytes.fromhex("01 05 00 03 FF 00 7C 3A"),
            False: bytes.fromhex("01 05 00 03 00 00 3D CA"),
        },
    }

    def __init__(
        self,
        port="/dev/ttyUSB1",
        baudrate=9600,
        timeout=0.5,
        retry_count=3
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retry_count = retry_count

        self.ser = None

        # IMPORTANT:
        # This means "the last state that software believes
        # was successfully sent to the relay".
        self.last_state = None


    # ============================================================
    # Serial open / close
    # ============================================================

    def open(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=1,
                timeout=self.timeout,
                write_timeout=self.timeout
            )

            print(
                f"[Relay] RS485 relay opened on "
                f"{self.port}, baudrate={self.baudrate}"
            )

            # Clear all relays on startup.
            if not self.all_off():
                print("[Relay] WARNING: failed to turn all relays off at startup")
                self.last_state = None

            return True

        except Exception as e:
            print(f"[Relay] Failed to open RS485 relay: {e}")
            self.ser = None
            return False


    def close(self):
        try:
            # Try to leave hardware in a safe state.
            self.all_off()

            if self.ser:
                self.ser.close()
                self.ser = None

            self.last_state = None

            print("[Relay] RS485 relay closed")

        except Exception as e:
            print(f"[Relay] Close error: {e}")


    # ============================================================
    # Low-level RS485 communication
    # ============================================================

    def _write_cmd_once(self, cmd):
        """
        Send one Modbus command once.

        Returns:
            True  -> serial write completed without exception
            False -> serial write failed

        NOTE:
            This only confirms that the command was written to
            the serial port. It does NOT yet confirm that the
            physical relay really changed state.
        """

        if self.ser is None:
            print("[Relay] Serial port is not open")
            return False

        try:
            self.ser.write(cmd)
            self.ser.flush()

            # Give the RS485 relay a short time to execute.
            time.sleep(0.03)

            return True

        except Exception as e:
            print(f"[Relay] Write failed: {e}")
            return False


    def _write_cmd(self, cmd):
        """
        Send one command with retry.

        Returns:
            True if one attempt succeeds.
            False if all attempts fail.
        """

        for attempt in range(1, self.retry_count + 1):

            if self._write_cmd_once(cmd):
                return True

            print(
                f"[Relay] Retry "
                f"{attempt}/{self.retry_count}"
            )

            time.sleep(0.05)

        print("[Relay] Command failed after all retries")
        return False


    # ============================================================
    # Relay control
    # ============================================================

    def set_relay(self, relay_id, on):
        """
        Control one relay.

        Args:
            relay_id: 0 / 1 / 2 / 3
            on: True / False

        Returns:
            True on serial write success.
            False on failure.
        """

        if relay_id not in self.RELAY_COMMANDS:
            print(f"[Relay] Invalid relay id: {relay_id}")
            return False

        cmd = self.RELAY_COMMANDS[relay_id][on]

        return self._write_cmd(cmd)


    def all_off(self):
        """
        Turn off all relays.

        Returns True only if all four commands succeed.
        """

        success = True

        for relay_id in range(4):

            ok = self.set_relay(relay_id, False)

            if not ok:
                print(
                    f"[Relay] Failed to turn OFF "
                    f"relay {relay_id}"
                )
                success = False

        return success


    def read_status(self):
        """
        读继电器线圈状态，返回实际灯态字符串。
        只发功能码 01（读线圈），纯查询，不改变继电器状态。

        Returns:
            "RED" / "YELLOW" / "GREEN" / "RED_YELLOW" / "OFF"，
            失败或超时返回 None。
        """
        if self.ser is None:
            return None
        try:
            # 清空旧缓冲，稍等再清一次（清掉之前写命令的延迟回显）
            self.ser.reset_input_buffer()
            time.sleep(0.08)
            self.ser.reset_input_buffer()
            # 读线圈 0x00-0x07（8 路）
            self.ser.write(bytes.fromhex("01 01 00 00 00 08 3D CC"))
            time.sleep(0.3)                 # 等继电器响应（部分继电器较慢）
            self.ser.timeout = 0.3
            resp = self.ser.read(16)        # 读足够字节（可能含前导回显）
            self.ser.timeout = 0.5          # 恢复默认
            idx = resp.find(b'\x01\x01')    # 定位读线圈响应头
            if idx < 0 or idx + 6 > len(resp):
                return None
            coil = resp[idx + 3]            # 8 路线圈状态字节
            red = bool(coil & 0x01)      # 继电器 0 = 红
            yellow = bool(coil & 0x02)   # 继电器 1 = 黄
            green = bool(coil & 0x04)    # 继电器 2 = 绿
            if red and yellow:
                return "RED_YELLOW"
            if green:
                return "GREEN"
            if yellow:
                return "YELLOW"
            if red:
                return "RED"
            return "OFF"
        except Exception:
            return None


    # ============================================================
    # Traffic light high-level API
    # ============================================================

    def set_light(self, color):
        """
        Set traffic light color.

        Supported:
            RED
            YELLOW
            GREEN
            OFF

        Returns:
            True  -> command sequence succeeded
            False -> command sequence failed
        """

        color = color.upper()

        # --------------------------------------------------------
        # Same state:
        # skip repeated command ONLY because the previous command
        # was considered successful.
        # --------------------------------------------------------

        if color == self.last_state:
            return True


        # --------------------------------------------------------
        # Validate color
        # --------------------------------------------------------

        if color not in ("RED", "YELLOW", "GREEN", "RED_YELLOW", "OFF"):

            print(
                f"[Relay] Unknown light color: {color}, "
                f"falling back to RED"
            )

            color = "RED"


        # --------------------------------------------------------
        # RED_YELLOW: red stays on, just add yellow (NO all_off).
        # --------------------------------------------------------

        if color == "RED_YELLOW":

            ok_red = self.set_relay(0, True)     # ensure red is on (idempotent)
            ok_yellow = self.set_relay(1, True)  # add yellow

            if ok_red and ok_yellow:
                self.last_state = color
                print("[Relay] Light -> RED_YELLOW")
                return True

            self.last_state = None
            return False


        # --------------------------------------------------------
        # Step 1:
        # Turn everything off first.
        # --------------------------------------------------------

        if not self.all_off():

            print(
                f"[Relay] ERROR: failed to clear relays "
                f"before switching to {color}"
            )

            # Very important:
            # do NOT remember this color.
            #
            # Next main-loop iteration will retry.
            self.last_state = None

            return False


        # --------------------------------------------------------
        # OFF requires no additional relay.
        # --------------------------------------------------------

        if color == "OFF":

            self.last_state = "OFF"

            print("[Relay] Light -> OFF")

            return True


        # --------------------------------------------------------
        # Step 2:
        # Enable target relay.
        # --------------------------------------------------------

        if color == "RED":
            target_relay = 0

        elif color == "YELLOW":
            target_relay = 1

        elif color == "GREEN":
            target_relay = 2

        else:
            # Defensive fallback.
            target_relay = 0
            color = "RED"


        success = self.set_relay(
            target_relay,
            True
        )


        # --------------------------------------------------------
        # Step 3:
        # Update last_state ONLY after successful write.
        # --------------------------------------------------------

        if success:

            self.last_state = color

            print(f"[Relay] Light -> {color}")

            return True


        # --------------------------------------------------------
        # Failed:
        # don't cache the requested state.
        #
        # This guarantees that next call retries it.
        # --------------------------------------------------------

        print(
            f"[Relay] ERROR: failed to switch "
            f"light to {color}"
        )

        self.last_state = None

        return False


# ================================================================
# Standalone test
# ================================================================

if __name__ == "__main__":

    relay = RS485RelayController(
        port="/dev/ttyUSB1",
        baudrate=9600,
        timeout=0.5,
        retry_count=3
    )

    if not relay.open():
        print("[TEST] Cannot open relay")
        raise SystemExit(1)

    try:

        print("[TEST] RED")
        relay.set_light("RED")
        time.sleep(3)

        print("[TEST] GREEN")
        relay.set_light("GREEN")
        time.sleep(3)

        print("[TEST] YELLOW")
        relay.set_light("YELLOW")
        time.sleep(3)

        print("[TEST] RED")
        relay.set_light("RED")
        time.sleep(3)

        print("[TEST] OFF")
        relay.set_light("OFF")

    except KeyboardInterrupt:
        pass

    finally:
        relay.close()