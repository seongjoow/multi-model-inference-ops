"""게이트웨이와 CV 워커가 함께 쓰는 Redis 키 이름."""

QUEUE = "cv:queue"  # 대기 중인 job_id 목록 (RPUSH로 넣고 왼쪽에서 꺼냄)
DLQ = "cv:dlq"  # 재시도 상한을 넘은 job_id
PROCESSING_PATTERN = "cv:processing:*"
HEARTBEAT_PATTERN = "cv:heartbeat:*"


def job(job_id: str) -> str:
    """작업 해시: image, status, enqueued_at, attempts, result ..."""
    return f"cv:job:{job_id}"


def result(job_id: str) -> str:
    """결과 전달용 리스트. 게이트웨이가 BLPOP으로 기다린다."""
    return f"cv:result:{job_id}"


def processing(worker_id: str) -> str:
    """reliable 모드에서 워커가 처리 중인 job_id 목록."""
    return f"cv:processing:{worker_id}"


def heartbeat(worker_id: str) -> str:
    return f"cv:heartbeat:{worker_id}"
