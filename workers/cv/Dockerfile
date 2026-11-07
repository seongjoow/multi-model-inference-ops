# 빌드 컨텍스트는 레포 루트 (common/ 을 같이 쓰기 위해)
# 같은 Dockerfile로 Mac(arm64, CPU)과 EC2(x86_64, CUDA)를 모두 빌드한다. torch 휠은 pip가 아키텍처에 맞게 고른다.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# opencv-python(ultralytics 의존성)이 필요로 하는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY workers/cv/requirements.txt workers/cv/requirements.txt
RUN pip install --no-cache-dir -r workers/cv/requirements.txt

COPY common/ common/
COPY workers/ workers/

EXPOSE 9101
CMD ["python", "-m", "workers.cv.worker"]
