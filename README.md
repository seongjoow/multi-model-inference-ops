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
loadgen/            부하 생성기 (요청별 CSV, 실험 조건 기록, 요약)
monitoring/         Prometheus 수집 설정, Grafana 대시보드, GPU exporter
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

`make up`은 GPU가 있으면(EC2) GPU 구성을, 없으면(Mac) 기본 구성을 자동으로 고릅니다.

## 대시보드 (Prometheus + Grafana)

`make up`이 Prometheus와 Grafana도 함께 띄웁니다. 모든 서비스의 `/metrics`를 2초마다 수집합니다.

- Grafana: http://localhost:3000 → "Inference Ops" 대시보드 (LLM, CV, 자원 3구역)
- Prometheus: http://localhost:9090 → Status > Target health 에서 수집 대상이 모두 UP인지 확인

EC2에서는 포트를 인터넷에 열지 않고 SSH 터널로 봅니다.

```bash
ssh -i ~/.ssh/mminfops.pem -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8000:localhost:8000 ubuntu@<IP>
```

GPU 지표는 `monitoring/gpu-exporter`(NVML, 프로세스별 VRAM 포함), 호스트 CPU·메모리는 node-exporter가 수집하며 EC2 구성에만 포함됩니다.

## 부하 실험

```bash
make load-llm RATE=2 DURATION=60 NAME=baseline-llm-r2   # LLM에 초당 2건, 60초
make load-cv  RATE=10 DURATION=60 NAME=baseline-cv-r10  # CV에 초당 10건, 60초
make summary RUNS="results/baseline-llm-r2 results/baseline-cv-r10"
```

- 기본은 고정 도착률(open loop, 포아송 간격)이고, `ARGS="--concurrency 8"`로 고정 동시성(closed loop)도 가능
- 앞 10초는 워밍업으로 요약에서 제외 (`--warmup`)
- LLM 프롬프트는 단어 수(`--input-words`)로 입력 길이를, `--max-tokens`와 `ignore_eos`로 출력 길이를 고정
- 결과 폴더 `results/<NAME>/`: `requests.csv`(요청별 시간), `summary.json`, `meta.json`(옵션·커밋), `env.txt`(당시 설정)
- 끝나면 그 실험 구간만 보여주는 Grafana 링크를 출력
- 모든 요청에 `X-Request-ID: <NAME>-<번호>`가 붙어, 서비스 로그에서 그대로 추적 가능

`python -m loadgen --help`로 전체 옵션을 볼 수 있습니다.

## 실험 실행과 분석

여러 조건을 한 번에 돌리는 실행기(`experiments/run.py`, 표준 라이브러리만 사용)와 분석 스크립트(`analysis/report.py`)가 있습니다.

| 시나리오 | 내용 |
| --- | --- |
| `llm-only` | LLM만, 도착률을 단계적으로 올림 (`--llm-rates`) |
| `cv-only` | CV만, 도착률을 단계적으로 올림 (`--cv-rates`) |
| `mixed-cv-sweep` | LLM을 `--llm-fixed`로 고정하고 CV를 올림 → CV가 LLM에 주는 영향 |
| `mixed-llm-sweep` | CV를 `--cv-fixed`로 고정하고 LLM을 올림 → LLM이 CV에 주는 영향 |

```bash
# EC2 (SSH가 끊겨도 계속 돌도록 tmux 안에서)
make exp-pilot EXP=d3            # 시험 실행 약 12분 → results/d3-pilot/
make exp EXP=d3                  # 본 실험: 3회 반복, 약 80분 → results/d3/
make exp EXP=d3 ARGS="--llm-rates 1,2,4,8 --llm-fixed 2 --repeat 2"   # 조건 변경

# Mac (결과를 가져온 뒤)
uv pip install -r analysis/requirements.txt
make report EXP=d3               # results/d3/report/ 에 표·그래프·report.md
```

- 단계마다 부하 생성기 결과와 함께, 측정 구간의 GPU 사용률·호스트 CPU·서버 내부 대기 등을 Prometheus에서 읽어 `resources.json`으로 저장
- 실행 조건(`plan.json`: 옵션, 커밋, GPU)과 당시 `.env`를 결과 폴더에 함께 저장
- 처리량은 "측정 구간 안에 끝난 요청 수"로 계산 (과부하에서 줄이 길어지는 것을 숨기지 않도록)
- `report.md`가 처리량이 떨어지거나 지연 p95가 3배 넘게 뛰는 첫 도착률을 자동으로 표시

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
