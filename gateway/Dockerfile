# 빌드 컨텍스트는 레포 루트 (common/ 을 같이 쓰기 위해)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY gateway/requirements.txt gateway/requirements.txt
RUN pip install --no-cache-dir -r gateway/requirements.txt

COPY common/ common/
COPY gateway/ gateway/

EXPOSE 8000
# 프로세스 1개: LLM 동시성 상한(세마포어)이 프로세스 단위라 여러 워커로 띄우면 상한이 깨진다
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
