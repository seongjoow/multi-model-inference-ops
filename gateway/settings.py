from __future__ import annotations

from dataclasses import dataclass

from common.env import env_float, env_int, env_str


@dataclass(frozen=True)
class Settings:
    redis_url: str
    redis_max_connections: int
    llm_base_url: str
    llm_model: str
    llm_max_inflight: int  # 게이트웨이가 LLM 서버로 동시에 보내는 요청 수 상한 (admission control)
    llm_queue_timeout_s: float  # 상한에 걸려 이 시간 넘게 기다리면 503
    llm_request_timeout_s: float
    cv_max_queue: int  # CV 큐가 이 길이 이상이면 429 (backpressure)
    cv_wait_timeout_s: float  # wait=true 요청이 결과를 기다리는 최대 시간
    cv_job_ttl_s: int
    stats_interval_s: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=env_str("REDIS_URL", "redis://localhost:6379/0"),
            redis_max_connections=env_int("REDIS_MAX_CONNECTIONS", 512),
            llm_base_url=env_str("LLM_BASE_URL", "http://localhost:8080").rstrip("/"),
            llm_model=env_str("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            llm_max_inflight=env_int("LLM_MAX_INFLIGHT", 64),
            llm_queue_timeout_s=env_float("LLM_QUEUE_TIMEOUT_S", 30.0),
            llm_request_timeout_s=env_float("LLM_REQUEST_TIMEOUT_S", 120.0),
            cv_max_queue=env_int("CV_MAX_QUEUE", 10_000),
            cv_wait_timeout_s=env_float("CV_WAIT_TIMEOUT_S", 60.0),
            cv_job_ttl_s=env_int("CV_JOB_TTL_S", 3600),
            stats_interval_s=env_float("STATS_INTERVAL_S", 1.0),
            log_level=env_str("LOG_LEVEL", "INFO"),
        )
