#!/usr/bin/env python3
"""
Traffic Light Allocator - Core decision engine for smart traffic control.

Extracted from smart_traffic.py, this module encapsulates the state machine
and patience-score-based allocation algorithm. It runs on Board A as the
master controller, combining vehicle counts from both roads to decide
which direction gets the green light and for how long.

State Machine:
    GREEN_A → YELLOW_A → ALL_RED_A → GREEN_B → YELLOW_B → ALL_RED_B → GREEN_A

Allocation Strategy (Patience Score):
    - Score = vehicle_count + (wait_time / 10.0)
    - The road that has been waiting longer gets a gradually increasing score
    - Switch when: opposing_road_score > current_road_score * 1.5
    - Must wait MIN_GREEN_TIME before considering a switch
    - Force switch after MAX_GREEN_TIME
    - Debounce congestion detection for DEBOUNCE_TIME seconds
"""

import time
from collections import deque
from enum import Enum


class MedianFilter:
    """Sliding-window median filter for smoothing noisy detection counts.

    A median filter discards isolated outliers (e.g. a single frame where
    YOLO misses all vehicles) while faithfully tracking real trends.
    """

    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        return self.median()

    def median(self):
        if not self.window:
            return 0
        sorted_vals = sorted(self.window)
        n = len(sorted_vals)
        return sorted_vals[n // 2]

    def reset(self):
        self.window.clear()


class TrafficState(Enum):
    """Traffic light phases for a two-road intersection."""
    GREEN_A = 1    # Road A green, Road B red
    YELLOW_A = 2   # Road A yellow, Road B red
    ALL_RED_A = 5  # All red (clearing after A→B transition)
    GREEN_B = 3    # Road B green, Road A red
    YELLOW_B = 4   # Road B yellow, Road A red
    ALL_RED_B = 6  # All red (clearing after B→A transition)


class TrafficLightAllocator:
    """
    Smart traffic light allocation engine.

    Input: vehicle counts from Road A and Road B (updated each frame)
    Output: current light state, remaining time, and RGB colors for display

    Usage:
        allocator = TrafficLightAllocator()
        while True:
            count_a = get_road_a_vehicle_count()
            count_b = get_road_b_vehicle_count()
            state, remaining, color_a, color_b, is_green_a, is_green_b = allocator.update(count_a, count_b)
            # Drive hardware outputs or display based on these values
    """

    # ── Timing parameters (seconds) ──
    MIN_GREEN_TIME = 5.0    # Minimum green light duration
    MAX_GREEN_TIME = 30.0   # Maximum green light duration (force switch)
    YELLOW_TIME = 3.0       # Yellow light duration
    ALL_RED_TIME = 2.0      # All-red clearance interval
    DEBOUNCE_TIME = 2.0     # Congestion must persist this long before switching

    # ── Congestion threshold ──
    MIN_VEHICLES_FOR_SWITCH = 2  # Opposing road needs at least this many vehicles

    # ── Score weight ──
    WAIT_WEIGHT = 10.0  # Divisor for wait_time in patience score

    def __init__(self, median_window=5):
        """Initialize the allocator in GREEN_A state."""
        self.current_state = TrafficState.GREEN_A
        self.state_start_time = time.time()
        self.condition_met_start_time = 0.0  # Debounce timer for congestion

        # Median filters for YOLO count stabilisation
        self.filter_a = MedianFilter(window_size=median_window)
        self.filter_b = MedianFilter(window_size=median_window)

        # Output state
        self.color_a = (0, 255, 0)   # (B, G, R) - starts green
        self.color_b = (0, 0, 255)   # starts red
        self.is_green_a = True
        self.is_green_b = False
        self.remaining_time = self.MAX_GREEN_TIME

        # Stats
        self.total_switches = 0
        self.last_switch_reason = "init"

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    def update(self, count_a, count_b):
        """
        Run one frame of the allocation state machine.

        Args:
            count_a (int): Number of vehicles detected on Road A (raw)
            count_b (int): Number of vehicles detected on Road B (raw)

        Returns:
            tuple: (TrafficState, remaining_seconds, color_a, color_b,
                    is_green_a, is_green_b, score_a, score_b)
        """
        # ── Median filter to suppress YOLO flicker ──
        count_a_smooth = self.filter_a.update(count_a)
        count_b_smooth = self.filter_b.update(count_b)

        current_time = time.time()
        elapsed_time = current_time - self.state_start_time

        # Reset to defaults each frame
        self.color_a = (0, 0, 255)  # Red
        self.color_b = (0, 0, 255)  # Red
        self.is_green_a = False
        self.is_green_b = False
        remaining = 0.0

        # Calculate patience scores (use smoothed counts)
        score_a = self._patience_score(count_a_smooth, elapsed_time, is_road_a=True)
        score_b = self._patience_score(count_b_smooth, elapsed_time, is_road_a=False)

        # ── Dispatch to state handler ──
        if self.current_state == TrafficState.GREEN_A:
            remaining, score_a, score_b = self._handle_green_a(
                current_time, elapsed_time,
                count_a_smooth, count_b_smooth, score_a, score_b)
        elif self.current_state == TrafficState.YELLOW_A:
            remaining = self._handle_yellow(current_time, elapsed_time, TrafficState.ALL_RED_A)
        elif self.current_state == TrafficState.ALL_RED_A:
            remaining = self._handle_all_red(current_time, elapsed_time, TrafficState.GREEN_B)
        elif self.current_state == TrafficState.GREEN_B:
            remaining, score_a, score_b = self._handle_green_b(
                current_time, elapsed_time,
                count_a_smooth, count_b_smooth, score_a, score_b)
        elif self.current_state == TrafficState.YELLOW_B:
            remaining = self._handle_yellow(current_time, elapsed_time, TrafficState.ALL_RED_B)
        elif self.current_state == TrafficState.ALL_RED_B:
            remaining = self._handle_all_red(current_time, elapsed_time, TrafficState.GREEN_A)

        self.remaining_time = max(0.0, remaining)

        return (self.current_state, self.remaining_time,
                self.color_a, self.color_b,
                self.is_green_a, self.is_green_b,
                score_a, score_b)

    def get_gpio_state(self):
        """
        Return GPIO-friendly output for hardware traffic lights.

        Returns:
            dict: {'red_a': bool, 'yellow_a': bool, 'green_a': bool,
                   'red_b': bool, 'yellow_b': bool, 'green_b': bool}
        """
        state = self.current_state
        return {
            'red_a': state not in (TrafficState.GREEN_A, TrafficState.YELLOW_A),
            'yellow_a': state == TrafficState.YELLOW_A,
            'green_a': state == TrafficState.GREEN_A,
            'red_b': state not in (TrafficState.GREEN_B, TrafficState.YELLOW_B),
            'yellow_b': state == TrafficState.YELLOW_B,
            'green_b': state == TrafficState.GREEN_B,
        }

    def reset(self):
        """Reset allocator to initial state."""
        self.current_state = TrafficState.GREEN_A
        self.state_start_time = time.time()
        self.condition_met_start_time = 0.0
        self.color_a = (0, 255, 0)
        self.color_b = (0, 0, 255)
        self.is_green_a = True
        self.is_green_b = False
        self.remaining_time = self.MAX_GREEN_TIME
        self.filter_a.reset()
        self.filter_b.reset()

    def get_state_summary(self):
        """Return a human-readable summary string."""
        return (f"[{self.current_state.name}] "
                f"A: vehicles → {'GO' if self.is_green_a else 'STOP'} | "
                f"B: vehicles → {'GO' if self.is_green_b else 'STOP'} | "
                f"remaining={self.remaining_time:.1f}s | "
                f"switches={self.total_switches}")

    # ──────────────────────────────────────────────
    #  Patience Score
    # ──────────────────────────────────────────────

    def _patience_score(self, count, elapsed_time, is_road_a):
        """
        Calculate patience score for a road.

        Score = vehicle_count + (wait_time / WAIT_WEIGHT)
        The road accumulates waiting time only when ITS light is red.

        Args:
            count (int): Current vehicle count on this road
            elapsed_time (float): Time elapsed in current state
            is_road_a (bool): Whether this is Road A

        Returns:
            float: Patience score
        """
        road_is_waiting = False
        if is_road_a:
            # Road A waits whenever it is NOT green or yellow
            road_is_waiting = self.current_state in (
                TrafficState.GREEN_B, TrafficState.YELLOW_B, TrafficState.ALL_RED_B)
        else:
            # Road B waits whenever it is NOT green or yellow
            road_is_waiting = self.current_state in (
                TrafficState.GREEN_A, TrafficState.YELLOW_A, TrafficState.ALL_RED_A)

        wait_time = elapsed_time if road_is_waiting else 0.0
        return count + (wait_time / self.WAIT_WEIGHT)

    # ──────────────────────────────────────────────
    #  State Handlers
    # ──────────────────────────────────────────────

    def _handle_green_a(self, current_time, elapsed_time, count_a, count_b, score_a, score_b):
        """Handle GREEN_A state: check if should switch to YELLOW_A."""
        self.color_a = (0, 255, 0)  # Green (BGR)
        self.is_green_a = True

        should_switch = False

        if elapsed_time > self.MIN_GREEN_TIME:
            if elapsed_time > self.MAX_GREEN_TIME:
                # Force switch: max green time reached
                should_switch = True
                self.condition_met_start_time = 0.0
                self.last_switch_reason = "max_time"
            else:
                # Weighted scoring: B more congested than A?
                congestion = (count_b >= self.MIN_VEHICLES_FOR_SWITCH and
                              score_b > score_a * 1.5)

                if congestion:
                    if self.condition_met_start_time == 0.0:
                        self.condition_met_start_time = current_time
                    elif current_time - self.condition_met_start_time > self.DEBOUNCE_TIME:
                        should_switch = True
                        self.condition_met_start_time = 0.0
                        self.last_switch_reason = "congestion_B"
                else:
                    self.condition_met_start_time = 0.0

        remaining = self.MAX_GREEN_TIME - elapsed_time

        if should_switch:
            self._transition_to(TrafficState.YELLOW_A, current_time)

        return remaining, score_a, score_b

    def _handle_green_b(self, current_time, elapsed_time, count_a, count_b, score_a, score_b):
        """Handle GREEN_B state: check if should switch to YELLOW_B."""
        self.color_b = (0, 255, 0)  # Green (BGR)
        self.is_green_b = True

        should_switch = False

        if elapsed_time > self.MIN_GREEN_TIME:
            if elapsed_time > self.MAX_GREEN_TIME:
                should_switch = True
                self.condition_met_start_time = 0.0
                self.last_switch_reason = "max_time"
            else:
                congestion = (count_a >= self.MIN_VEHICLES_FOR_SWITCH and
                              score_a > score_b * 1.5)

                if congestion:
                    if self.condition_met_start_time == 0.0:
                        self.condition_met_start_time = current_time
                    elif current_time - self.condition_met_start_time > self.DEBOUNCE_TIME:
                        should_switch = True
                        self.condition_met_start_time = 0.0
                        self.last_switch_reason = "congestion_A"
                else:
                    self.condition_met_start_time = 0.0

        remaining = self.MAX_GREEN_TIME - elapsed_time

        if should_switch:
            self._transition_to(TrafficState.YELLOW_B, current_time)

        return remaining, score_a, score_b

    def _handle_yellow(self, current_time, elapsed_time, next_state):
        """Handle YELLOW_A or YELLOW_B state."""
        if self.current_state == TrafficState.YELLOW_A:
            self.color_a = (0, 255, 255)  # Yellow (BGR)
        else:
            self.color_b = (0, 255, 255)  # Yellow (BGR)

        if elapsed_time > self.YELLOW_TIME:
            self._transition_to(next_state, current_time)

        return self.YELLOW_TIME - elapsed_time

    def _handle_all_red(self, current_time, elapsed_time, next_state):
        """Handle ALL_RED_A or ALL_RED_B state (both roads red)."""
        self.color_a = (0, 0, 255)  # Red
        self.color_b = (0, 0, 255)  # Red

        if elapsed_time > self.ALL_RED_TIME:
            self._transition_to(next_state, current_time)

        return self.ALL_RED_TIME - elapsed_time

    def _transition_to(self, new_state, current_time):
        """Perform a state transition."""
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = current_time
        self.condition_met_start_time = 0.0
        self.total_switches += 1
        # Debug logging (can be removed in production)
        # print(f"  [Allocator] {old_state.name} → {new_state.name}  "
        #       f"(reason: {self.last_switch_reason})")


# ──────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("TrafficLightAllocator - Standalone Test (with median filter)")
    print("=" * 60)

    allocator = TrafficLightAllocator()
    import random
    random.seed(42)

    # Simulate YOLO flicker: realistic scene with occasional 0 detections
    flicker_a = [3,0,3,4,0,3,0,4,3,0, 3,3,0,3,4,0,3,3,0,4,
                 8,8,0,8,8,0,8,0,8,8]
    flicker_b = [0,1,0,0,1,0,0,0,1,0, 0,0,1,0,0,1,0,0,0,1,
                 0,0,0,0,0,0,0,0,0,0]

    for frame, (raw_a, raw_b) in enumerate(zip(flicker_a, flicker_b)):
        state, remaining, color_a, color_b, is_green_a, is_green_b, score_a, score_b = \
            allocator.update(raw_a, raw_b)

        smooth_a = allocator.filter_a.median()
        smooth_b = allocator.filter_b.median()

        a_status = "GO " if is_green_a else "STOP"
        b_status = "GO " if is_green_b else "STOP"
        print(f"F{frame:2d} raw A:{raw_a} B:{raw_b} -> "
              f"smooth A:{smooth_a} B:{smooth_b} | {state.name:10s} | "
              f"{a_status}/{b_status} | {remaining:5.1f}s")
        time.sleep(0.03)

    print(f"\nFinal: {allocator.get_state_summary()}")
