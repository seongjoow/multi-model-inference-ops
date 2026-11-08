"""GPU 지표를 Prometheus 형식으로 내놓는 작은 exporter (NVML 사용).

Prometheus가 긁어 갈 때마다(collect) NVML에서 값을 새로 읽는다. nvidia-smi가 보여주는 값과 같은 출처다.
pid: host 로 실행하면 GPU 메모리를 쓰는 프로세스의 이름(vllm, cv-worker)까지 붙여, VRAM을 누가 얼마나
쓰는지 나눠 볼 수 있다.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Iterator

import pynvml
from prometheus_client import start_http_server
from prometheus_client.core import REGISTRY, GaugeMetricFamily
from prometheus_client.registry import Collector

PORT = int(os.getenv("GPU_EXPORTER_PORT", "9400"))
log = logging.getLogger("gpu-exporter")


def process_label(pid: int) -> str:
    """/proc/<pid>/cmdline 으로 프로세스를 알아본다. pid: host 가 아니면 이름을 알 수 없다."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return f"pid-{pid}"
    low = cmd.lower()
    if "workers.cv.worker" in low:
        return "cv-worker"
    if "vllm" in low:
        return "vllm"
    return os.path.basename(cmd.split(" ")[0]) or f"pid-{pid}"


def _safe(fn, *args):
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return None  # 이 GPU에서 지원하지 않는 항목


class GpuCollector(Collector):
    def __init__(self) -> None:
        self.driver = pynvml.nvmlSystemGetDriverVersion()
        self.count = pynvml.nvmlDeviceGetCount()

    def collect(self) -> Iterator[GaugeMetricFamily]:
        up = GaugeMetricFamily("nvidia_gpu_up", "1 if NVML answered this scrape")
        info = GaugeMetricFamily("nvidia_gpu_info", "GPU model and driver", labels=["gpu", "name", "driver"])
        util = GaugeMetricFamily("nvidia_gpu_utilization_percent", "Time a kernel was running, last sample period", labels=["gpu"])
        mem_util = GaugeMetricFamily("nvidia_gpu_memory_utilization_percent", "Time memory was being read or written", labels=["gpu"])
        used = GaugeMetricFamily("nvidia_gpu_memory_used_bytes", "GPU memory used", labels=["gpu"])
        total = GaugeMetricFamily("nvidia_gpu_memory_total_bytes", "GPU memory total", labels=["gpu"])
        power = GaugeMetricFamily("nvidia_gpu_power_watts", "Power draw", labels=["gpu"])
        temp = GaugeMetricFamily("nvidia_gpu_temperature_celsius", "GPU temperature", labels=["gpu"])
        clock = GaugeMetricFamily("nvidia_gpu_sm_clock_mhz", "SM clock", labels=["gpu"])
        proc = GaugeMetricFamily(
            "nvidia_gpu_process_memory_used_bytes", "GPU memory per process", labels=["gpu", "pid", "process"]
        )
        try:
            for i in range(self.count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                g = str(i)
                name = pynvml.nvmlDeviceGetName(h)
                info.add_metric([g, name if isinstance(name, str) else name.decode(), self.driver], 1)
                rates = _safe(pynvml.nvmlDeviceGetUtilizationRates, h)
                if rates is not None:
                    util.add_metric([g], rates.gpu)
                    mem_util.add_metric([g], rates.memory)
                mem = _safe(pynvml.nvmlDeviceGetMemoryInfo, h)
                if mem is not None:
                    used.add_metric([g], mem.used)
                    total.add_metric([g], mem.total)
                mw = _safe(pynvml.nvmlDeviceGetPowerUsage, h)
                if mw is not None:
                    power.add_metric([g], mw / 1000)
                t = _safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)
                if t is not None:
                    temp.add_metric([g], t)
                c = _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM)
                if c is not None:
                    clock.add_metric([g], c)
                for p in _safe(pynvml.nvmlDeviceGetComputeRunningProcesses, h) or []:
                    if p.usedGpuMemory is not None:
                        proc.add_metric([g, str(p.pid), process_label(p.pid)], p.usedGpuMemory)
            up.add_metric([], 1)
        except pynvml.NVMLError as exc:
            log.warning("nvml read failed: %s", exc)
            up.add_metric([], 0)
        yield from (up, info, util, mem_util, used, total, power, temp, clock, proc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        sys.exit(f"NVML을 초기화할 수 없습니다 (GPU 예약이 빠졌거나 드라이버가 없음): {exc}")
    collector = GpuCollector()
    REGISTRY.register(collector)
    start_http_server(PORT)
    log.info("gpu-exporter listening on :%d, gpus=%d, driver=%s", PORT, collector.count, collector.driver)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
