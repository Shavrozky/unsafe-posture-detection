from collections import deque, Counter
import time


class TemporalSmoothing:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.labels = deque(maxlen=window_size)
        self.current_label = "unknown"
        self.current_start_time = time.time()

    def update(self, label):
        self.labels.append(label)

        if len(self.labels) == 0:
            return "unknown", 0.0

        most_common_label = Counter(self.labels).most_common(1)[0][0]

        now = time.time()

        if most_common_label != self.current_label:
            self.current_label = most_common_label
            self.current_start_time = now

        duration = now - self.current_start_time

        return self.current_label, duration