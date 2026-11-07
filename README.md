# multi-model-inference-ops

LLM과 Computer Vision 두 워크로드를 한 장비에서 함께 서빙하고, 자원 경합·병목·장애를 측정과 로그로 분석하는 추론 운영 환경입니다.

- **LLM**: 게이트웨이가 OpenAI 호환 서버(Mac: llama.cpp, EC2: vLLM)로 스트리밍 프록시. 동시 요청 수를 게이트웨이에서 제한하고 대기 시간·TTFT·토큰당 시간을 나눠 측정
- **CV**: 게이트웨이가 이미지를 Redis 큐에 넣고, 워커가 마이크로배치로 YOLO 추론. 지연을 큐 대기 / 배치 대기 / 디코딩 / 추론으로 나눠 측정
- **관측**: 모든 서비스가 Prometheus 지표(`/metrics`)와 JSON 로그(request_id 포함)를 남김

## 구조

```
gateway/            FastAPI 게이트웨이 (LLM 프록시, CV 작업 등록·조회)
workers/cv/         CV 워커 (Redis 큐, 마이크로배치, YOLO / dummy 백엔드)
common/             공용 코드 (설정, JSON 로그, Redis 키)
scripts/            llm_mac.sh (Mac LLM 서버), smoke_test.sh
docker-compose.yml      기본 구성 (Mac, CPU)
docker-compose.gpu.yml  EC2 GPU 구성 (vLLM 추가, CV 워커 GPU 사용)
.env.example        모든 설정값과 설명
```

## Mac에서 실행 (개발용)

```bash
make env                      # .env 생성
make llm-mac                  # 터미널 1: llama.cpp LLM 서버 (켜 둔 채로)
make up                       # 터미널 2: Redis + 게이트웨이 + CV 워커 (첫 빌드는 수 분)
unzip -q val2017.zip -d data/ # COCO val2017 이미지 (처음 한 번)
make smoke                    # 헬스체크, LLM, CV 요청이 모두 성공하면 SMOKE TEST PASSED
```

YOLO 가중치(`yolo11n.pt`)는 첫 실행 때 `models/`로 자동으로 받습니다. Mac 수치는 기능 확인용이고, 보고서용 측정은 EC2에서 합니다.

## EC2에서 실행 (GPU 실험용)

```bash
make gpu-up      # vLLM + 게이트웨이 + Redis + CV 워커(GPU)
make gpu-ps      # vllm이 healthy가 될 때까지 대기 (모델 다운로드 포함 수 분)
make smoke
```

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | `/v1/llm/chat` | OpenAI chat completions 형식. `"stream": true`면 SSE, `[DONE]` 직전에 `: gateway {...}` 주석으로 측정값 전달 |
| POST | `/v1/cv/detect` | multipart `image`. 기본은 202 + `job_id`, `?wait=true`면 결과까지 대기 |
| GET | `/v1/cv/jobs/{job_id}` | 작업 상태와 결과 |
| GET | `/healthz`, `/readyz` | 생존 / Redis·LLM 연결 확인 |
| GET | `/metrics` | Prometheus 지표 (CV 워커는 `:9101/metrics`) |

## 실험에서 바꾸는 주요 설정

설정은 모두 `.env`로 바꿉니다. 전체 목록과 설명은 `.env.example`에 있습니다.

| 설정 | 의미 |
| --- | --- |
| `LLM_MAX_INFLIGHT` | 게이트웨이가 LLM 서버로 동시에 보내는 요청 수 상한 |
| `CV_BATCH_SIZE`, `CV_MAX_WAIT_MS` | CV 배치 최대 크기, 배치를 채우려고 기다리는 최대 시간 |
| `CV_QUEUE_MODE` | `simple`(워커가 죽으면 처리 중 작업 유실) / `reliable`(재시작 시 재처리, 재시도 상한 후 DLQ) |
| `CV_MODEL` | `yolo11n.pt` / `yolo26n.pt` (버전 교체 실험) |
| `VLLM_GPU_MEM_UTIL` | vLLM이 미리 잡는 GPU 메모리 비율 (CV 워커와 VRAM 공유) |
| `CV_BACKEND=dummy` | 모델 없이 고정 시간만 쓰는 백엔드. 큐·배치 동작만 따로 검증할 때 |

## 사용 내역 및 구현 범위

사용한 모델·오픈소스·라이선스와 AI 도구 사용 범위는 `docs/ai-usage.md`에 정리합니다.
