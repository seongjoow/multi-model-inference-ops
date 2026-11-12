# multi-model-inference-ops

GPU 한 장(NVIDIA T4)에서 LLM(Qwen2.5-1.5B-Instruct, vLLM)과 CV(YOLO11n) 추론을 함께 서빙하고, 자원 경합·병목·장애를 지표와 로그로 분석한 프로젝트입니다.

- **LLM**: 게이트웨이가 vLLM으로 동기 스트리밍(SSE). 동시 처리 수를 게이트웨이에서 제한
- **CV**: 게이트웨이가 작업을 Redis 큐에 넣고, 워커가 마이크로 배치로 디코딩(CPU)과 추론(GPU) 수행
- **관측**: Prometheus 지표, Grafana 대시보드, 경보 규칙 7개, 요청 ID가 포함된 JSON 로그

## 제출물

| 제출물 | 위치 |
| --- | --- |
| Executive Summary | [docs/executive-summary.pdf](docs/executive-summary.pdf) |
| Technical Report | [docs/technical-report.pdf](docs/technical-report.pdf) |
| System Architecture | [docs/architecture.md](docs/system-architecture.md) |
| 사용 기술 및 구현 범위 | [docs/ai-usage.md](docs/ai-usage.md) |

## 디렉터리 구조

```
gateway/                FastAPI 게이트웨이 (LLM 프록시, CV 작업 등록·결과 대기)
workers/cv/             CV 워커 (Redis 큐, 마이크로 배칭, YOLO 추론)
common/                 공용 코드 (설정, JSON 로그, Redis 키)
loadgen/                부하 생성기 (개방형 포아송 부하, 요청별 기록)
experiments/            실험 실행기, 실험별 설정(conf/*.env), 전체 실행 목록(run_all.sh)
analysis/               결과 분석 (표, 그래프, 전후 비교, 장애 분석)
monitoring/             Prometheus 수집·경보 규칙, Grafana 대시보드, GPU exporter
scripts/                llm_mac.sh (Mac LLM 서버), smoke_test.sh
docker-compose.yml      기본 구성 (Mac, CPU)
docker-compose.gpu.yml  EC2 GPU 구성 (vLLM, GPU 워커, exporter)
.env.example            모든 설정값과 설명
```

## 실행

### EC2 (g4dn.xlarge, 측정용)

```bash
make env                          # .env 생성
unzip -q val2017.zip -d data/     # COCO val2017 이미지 (처음 한 번)
make up                           # GPU가 있으면 vLLM·GPU 구성으로 실행
make ps                           # vllm이 healthy가 될 때까지 대기 (모델 다운로드 포함 수 분)
make smoke                        # LLM, CV 요청이 모두 성공하면 SMOKE TEST PASSED
```

### Mac (개발·기능 확인용)

```bash
make env
make llm-mac                      # 터미널 1: llama.cpp LLM 서버 (켜 둔 채로)
make up                           # 터미널 2: Redis + 게이트웨이 + CV 워커(CPU)
make smoke
```

YOLO 가중치는 첫 실행 때 `models/`로 자동으로 받습니다. Mac 수치는 기능 확인용이며, 보고서의 측정은 모두 EC2에서 했습니다.

## 실험 재현

보고서의 모든 결과는 한 세션(같은 GPU, 같은 커밋)에서 아래 명령 하나로 만들었습니다. 실행 목록과 순서는 `experiments/run_all.sh`에 있습니다.

```bash
# EC2 (SSH가 끊겨도 계속 돌도록 tmux 안에서, 약 4시간)
make exp-all                      # 결과: results/<실험 이름>/
```

결과 폴더를 Mac으로 가져와 분석합니다.

```bash
uv pip install -r analysis/requirements.txt
make report EXP=base && make report EXP=base-h2              # Baseline, 부하 증가, 경합
make compare EXPS="base e1-seqs64"                           # E1: vLLM 슬롯 32 → 64
make compare EXPS="base e2a-batch16 e2b-workers2"            # E2: 경합 중 CV 배치·워커
make fault-report EXPS="e5-simple e5-reliable"               # 장애 대응: CV 워커 중단
```

| 실험 | 바꾸는 설정 | 목적 |
| --- | --- | --- |
| `base` | 기본 설정 | Baseline, 부하 증가, 경합 ①(LLM 4 req/s 고정 + CV 증가) |
| `base-h2` | 기본 설정 | 경합 ②(CV 25 req/s 고정 + LLM 증가), 가설 H2-A |
| `e1-seqs64` | `VLLM_MAX_NUM_SEQS` 32 → 64 | 가설 H1, LLM 최적화 |
| `e2a-batch16` | CV 배치 16, 배치 대기 30 ms | 가설 H4 |
| `e2b-workers2` | `CV_WORKERS` 1 → 2 | 가설 H2-B, CV 최적화 |
| `e5-simple`, `e5-reliable` | `CV_QUEUE_MODE` simple / reliable | 장애 재현과 재발 방지 검증 |

실험별 설정은 `experiments/conf/*.env`에 있습니다. 결과 폴더마다 실행 옵션·커밋·GPU(`plan.json`)와 당시의 `.env`(`env.txt`)가 함께 저장됩니다.

## 대시보드

- Grafana: http://localhost:3000 → "Inference Ops" 대시보드
- Prometheus: http://localhost:9090 → Alerts 탭에서 경보 상태 확인

모든 포트는 `127.0.0.1`에만 열려 있으므로, EC2에서는 SSH 터널로 접속합니다.

```bash
ssh -i <키 파일> -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<IP>
```

## API

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | `/v1/llm/chat` | OpenAI chat completions 형식. `"stream": true`면 SSE |
| POST | `/v1/cv/detect` | multipart `image`. 기본은 202 + `job_id`, `?wait=true`면 결과까지 대기 |
| GET | `/v1/cv/jobs/{job_id}` | 작업 상태와 결과 |
| GET | `/healthz`, `/readyz` | 생존 / Redis·LLM 연결 확인 |
| GET | `/metrics` | Prometheus 지표 (CV 워커는 `:9101/metrics`) |

## 주요 설정

모든 설정은 `.env`로 바꿉니다. 전체 목록과 설명은 `.env.example`에 있습니다.

| 설정 | 의미 | 기본값 |
| --- | --- | --- |
| `LLM_MAX_INFLIGHT` | 게이트웨이의 LLM 동시 처리 상한 | 64 |
| `VLLM_MAX_NUM_SEQS` | vLLM 슬롯 수 | 32 |
| `VLLM_GPU_MEM_UTIL` | vLLM이 선점하는 GPU 메모리 비율 | 0.5 |
| `CV_BATCH_SIZE`, `CV_MAX_WAIT_MS` | CV 배치 최대 크기, 배치 대기 시간 | 8, 10 ms |
| `CV_WORKERS` | CV 워커 수 | 1 |
| `CV_QUEUE_MODE` | `simple`(워커 중단 시 처리 중 작업 유실) / `reliable`(재시작 시 재처리, 재시도 초과 시 DLQ) | simple |
| `CV_MAX_QUEUE` | CV 큐 길이 상한. 넘으면 429 | 10000 |
| `CV_WAIT_TIMEOUT_S` | CV 결과 대기 한도. 넘으면 504 | 60 |
