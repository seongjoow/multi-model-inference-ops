# 빌드 컨텍스트는 레포 루트
# GPU 드라이버 라이브러리(libnvidia-ml)는 NVIDIA Container Toolkit이 실행 시 넣어 준다 (compose의 gpu 예약)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY monitoring/gpu-exporter/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/gpu-exporter/exporter.py exporter.py

EXPOSE 9400
CMD ["python", "exporter.py"]
