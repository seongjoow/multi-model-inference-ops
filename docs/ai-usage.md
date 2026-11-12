# 사용 기술 및 구현 범위

## 1. 사용 모델·데이터·오픈소스

모든 모델은 EC2 인스턴스 안에서 직접 실행했으며, 외부 AI API는 사용하지 않았습니다.

| 구분 | 이름 | 버전 | 라이선스 |
| --- | --- | --- | --- |
| LLM | Qwen2.5-1.5B-Instruct (Hugging Face `Qwen/Qwen2.5-1.5B-Instruct`) | 최신 리비전 (해시 미기록) | Apache-2.0 |
| CV | YOLO11n, YOLO26n (Ultralytics 배포 가중치) | Ultralytics 8.4.203 | AGPL-3.0 |
| 데이터 | COCO 2017 val 이미지 | 2017 | 주석 CC BY 4.0, 이미지는 Flickr 원 조건 |
| LLM 서빙 | vLLM (`vllm/vllm-openai`) | 0.33.0 | Apache-2.0 |
| 게이트웨이 | FastAPI, uvicorn, httpx, redis-py | 0.162.0, 0.54.0, 0.28.1, 8.1.0 | MIT, BSD-3, BSD-3, MIT |
| CV 추론 | Ultralytics, OpenCV, PyTorch | 8.4.203, 5.0.0.93, 2.15.0 | AGPL-3.0, Apache-2.0, BSD-3 |
| 큐 | Redis | 7.2-alpine | BSD-3-Clause (7.4부터 라이선스 변경) |
| 모니터링 | Prometheus, node-exporter, Grafana | v3.5.0, v1.9.1, 12.4.13 | Apache-2.0, Apache-2.0, AGPL-3.0 |
| 인프라 | AWS EC2 g4dn.xlarge (NVIDIA T4), Docker Compose | - | - |

## 2. AI 도구 사용 범위

| 도구 | 사용한 곳 |
| --- | --- |
| Claude (Anthropic) | 코드 초안 작성(게이트웨이, CV 워커, 부하 생성기, 실험·분석 스크립트), 오류 원인 분석 보조, 보고서 초안과 문장 다듬기 |

## 3. 직접 수행한 범위

| 영역 | 내용 |
| --- | --- |
| 문제 정의와 설계 결정 | 워크로드·모델 선택, 처리 구조, SLO 기준, 실험 범위 결정 |
| 실험 수행 | AWS EC2 환경 구성, 전체 실험 실행, 측정 조건 문제 발견 후 전체 재실험 |
| 분석과 검증 | 결과 검토, 가설 판정 확인, 장애 대응 시나리오 구성 |
| 코드 검토와 수정 | AI가 작성한 코드의 리뷰, 실행 오류 수정, 설정값 조정 |
| 문서화 | Technical Report, Executive Summary, System Architecture의 구성과 최종 편집 |
