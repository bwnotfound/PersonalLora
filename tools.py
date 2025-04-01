import time


class TimeCounter:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        print(f"{self.name}: {(self.end_time - self.start_time) * 1000:.2f} ms")


class AvgRecorder:
    def __init__(self, max_count_num=10):
        self.max_count_num = max_count_num
        self.record_value = []

    def __call__(self, value):
        if not isinstance(value, (int, float)):
            try:
                value = value.item()
            except:
                try:
                    value = float(value)
                except:
                    raise ValueError(f"Cannot convert {value} to float")
        self.record_value.append(value)
        if self.max_count_num != -1 and len(self.record_value) > self.max_count_num:
            self.record_value.pop(0)

    def __str__(self):
        if len(self.record_value) == 0:
            return "0.0"
        return f"{sum(self.record_value) / len(self.record_value):.4f}"
