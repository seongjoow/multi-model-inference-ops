from __future__ import annotations

import socket
from dataclasses import dataclass

from common.env import env_bool, env_float, env_int, env_str


@dataclass(frozen=True)
class Settings:
    redis_url: str
    worker_id: str
    queue_mode: str  # simple: BLPOP (죽으면 처리 중 작업 유실) / reliable: BLMOVE + 처리중 목록 + 재처리
    batch_size: int  # 한 번에 묶는 최대 이미지 수
    max_wait_ms: float  # 첫 작업을 꺼낸 뒤 배치를 채우려고 더 기다리는 최대 시간
    max_attempts: int  # reliable 모드에서 이 횟수를 넘으면 DLQ로
    backend: str  # yolo | dummy
    model: str
    model_dir: str
    device: str  # cpu | 0 (첫 번째 GPU)
    half: bool  # FP16 추론 (GPU에서만 의미 있음)
    imgsz: int
    conf: float
    max_det: int
    torch_threads: int  # 0이면 torch 기본값
    dummy_base_ms: float
    dummy_per_item_ms: float
    metrics_port: int
    heartbeat_ttl_s: int
    heartbeat_file: str
    result_ttl_s: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=env_str("REDIS_URL", "redis://localhost:6379/0"),
            worker_id=env_str("CV_WORKER_ID", socket.gethostname()),
            queue_mode=env_str("CV_QUEUE_MODE", "simple").lower(),
            batch_size=env_int("CV_BATCH_SIZE", 8),
            max_wait_ms=env_float("CV_MAX_WAIT_MS", 10.0),
            max_attempts=env_int("CV_MAX_ATTEMPTS", 3),
            backend=env_str("CV_BACKEND", "yolo").lower(),
            model=env_str("CV_MODEL", "yolo11n.pt"),
            model_dir=env_str("CV_MODEL_DIR", "models"),
            device=env_str("CV_DEVICE", "cpu"),
            half=env_bool("CV_HALF", False),
            imgsz=env_int("CV_IMGSZ", 640),
            conf=env_float("CV_CONF", 0.25),
            max_det=env_int("CV_MAX_DET", 100),
            torch_threads=env_int("CV_TORCH_THREADS", 0),
            dummy_base_ms=env_float("CV_DUMMY_BASE_MS", 20.0),
            dummy_per_item_ms=env_float("CV_DUMMY_PER_ITEM_MS", 5.0),
            metrics_port=env_int("CV_METRICS_PORT", 9101),
            heartbeat_ttl_s=env_int("CV_HEARTBEAT_TTL_S", 15),
            heartbeat_file=env_str("CV_HEARTBEAT_FILE", "/tmp/cv-worker-heartbeat"),
            result_ttl_s=env_int("CV_RESULT_TTL_S", 600),
            log_level=env_str("LOG_LEVEL", "INFO"),
        )
