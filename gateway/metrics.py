"""게이트웨이 Prometheus 지표. LLM은 대기/첫 토큰/전체를 나눠 재야 병목 위치를 구분할 수 있다."""

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 120)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)

REQUESTS = Counter("gateway_requests_total", "Requests handled by the gateway", ["workload", "status"])

# LLM
LLM_QUEUE_WAIT = Histogram(
    "gateway_llm_queue_wait_seconds", "Time waiting for an LLM concurrency slot", buckets=LATENCY_BUCKETS
)
LLM_TTFT = Histogram(
    "gateway_llm_ttft_seconds", "Gateway receive to first output token (includes queue wait)", buckets=LATENCY_BUCKETS
)
LLM_TPOT = Histogram("gateway_llm_tpot_seconds", "Time per output token after the first", buckets=TPOT_BUCKETS)
LLM_E2E = Histogram("gateway_llm_e2e_seconds", "Gateway receive to last token", buckets=LATENCY_BUCKETS)
LLM_OUTPUT_TOKENS = Counter("gateway_llm_output_tokens_total", "Output tokens streamed back")
LLM_INFLIGHT = Gauge("gateway_llm_inflight", "LLM requests currently sent upstream")
LLM_WAITING = Gauge("gateway_llm_waiting", "LLM requests waiting for a concurrency slot")

# CV
CV_E2E = Histogram("gateway_cv_e2e_seconds", "Gateway receive to CV result (wait=true only)", buckets=LATENCY_BUCKETS)
CV_QUEUE_DEPTH = Gauge("cv_queue_depth", "Jobs waiting in the CV queue")
CV_PROCESSING = Gauge("cv_processing_jobs", "Jobs taken by workers but not acknowledged (reliable mode)")
CV_DLQ_DEPTH = Gauge("cv_dlq_depth", "Jobs moved to the dead-letter queue")
CV_WORKERS_ALIVE = Gauge("cv_workers_alive", "Workers with a fresh heartbeat")
