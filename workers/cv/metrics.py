"""CV 워커 Prometheus 지표. 지연을 큐 대기 / 배치 채우기 대기 / 디코딩 / 추론으로 나눠 잰다."""

from prometheus_client import Counter, Gauge, Histogram

WAIT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
STAGE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)

JOBS = Counter("cv_worker_jobs_total", "Jobs finished by the worker", ["outcome"])
QUEUE_WAIT = Histogram("cv_worker_queue_wait_seconds", "Enqueue to pop", buckets=WAIT_BUCKETS)
BATCH_WAIT = Histogram("cv_worker_batch_wait_seconds", "Pop to batch start (waiting for the batch to fill)", buckets=WAIT_BUCKETS)
DECODE = Histogram("cv_worker_decode_seconds", "Image decoding time per batch (CPU)", buckets=STAGE_BUCKETS)
INFERENCE = Histogram("cv_worker_inference_seconds", "Model call time per batch", buckets=STAGE_BUCKETS)
BATCH_SIZE = Histogram("cv_worker_batch_size", "Images per batch", buckets=BATCH_BUCKETS)
RECOVERED = Counter("cv_worker_recovered_jobs_total", "Unacknowledged jobs requeued at startup")
LAST_HEARTBEAT = Gauge("cv_worker_last_heartbeat_timestamp_seconds", "Unix time of the last heartbeat")
