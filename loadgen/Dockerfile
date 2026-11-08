# 빌드 컨텍스트는 레포 루트
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY loadgen/requirements.txt loadgen/requirements.txt
RUN pip install --no-cache-dir -r loadgen/requirements.txt

COPY loadgen/ loadgen/

ENTRYPOINT ["python", "-m", "loadgen"]
