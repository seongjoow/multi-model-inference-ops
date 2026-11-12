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
| `contention-cv-sweep` | LLM을 `--llm-fixed`로 고정하고 CV를 올림 → CV가 LLM에 주는 영향 |
| `contention-llm-sweep` | CV를 `--cv-fixed`로 고정하고 LLM을 올림 → LLM이 CV에 주는 영향 |

D3까지는 경합 시나리오를 `mixed-cv-sweep`, `mixed-llm-sweep`이라 불렀습니다. 실행기는 옛 이름도 받아서 새 이름으로 바꿔 실행하고, 분석 스크립트는 옛 결과를 새 이름으로 읽습니다.

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
- 처리량은 "측정 구간 안에 끝난 요청 수"로, goodput은 그중 SLO까지 지킨 요청 수로 계산 (과부하에서 줄이 길어지는 것을 숨기지 않도록)
- `report.md`가 처리량이 떨어지거나 지연 p95가 3배 넘게 뛰는 첫 도착률을 자동으로 표시
- 부하 생성기가 요청을 다 보내지 못한(dropped) 단계나 200 ms 넘게 늦게 보낸 단계는 † 로 표시하고 따로 모아 보여 줌
- "게이트웨이 밖에서 쓴 시간"(클라이언트 지연 − 게이트웨이 지연)과 게이트웨이 CPU를 나란히 보여 줘, 게이트웨이 자체가 병목인지 확인
- CV 추론·디코딩 시간은 배치당과 이미지당(요청마다 배치 시간 ÷ 배치 크기)을 함께 보여 줌. 배치 크기를 바꾸는 실험은 이미지당으로 비교
- LLM은 출력 토큰 처리량(tok/s, 측정 구간 안에 끝난 요청 기준)을 함께 보여 줌

### 설정을 바꿔 가며 비교하기 (최적화 전후)

실험 조건은 `experiments/conf/<이름>.env` 파일로 남깁니다. `make use`는 실험용 값을 기준(`base.env`, D3 본 실험과 같은 값)으로 되돌린 뒤 그 파일의 값만 덮어쓰고, 컨테이너를 다시 띄우고, 준비될 때까지 기다립니다. 적용한 이름은 `.env`의 `EXP_CONF`로 남아 결과의 `env.txt`에도 기록됩니다.

```bash
make confs                                   # 설정 목록과 각 설정의 가설
A="--scenarios contention-cv-sweep --llm-fixed 4 --cv-rates 25,40,50,75,100 --duration 45 --yes"
make use CONF=base         && make exp EXP=e2-base     ARGS="$A"
make use CONF=e2a-batch16  && make exp EXP=e2a-batch16 ARGS="$A"

# Mac
make compare EXPS="e2-base e2a-batch16"      # → results/compare/<이름들>/compare.md, 그래프
```

`compare.md`에는 실험마다 달랐던 `.env` 값, 실험별 SLO 용량·최대 goodput, 같은 도착률끼리 나란히 놓은 표가 들어갑니다. 첫 번째 실험이 기준입니다.

| 설정 | 바꾸는 것 | 확인하려는 것 |
| --- | --- | --- |
| `e1-seqs64` | `VLLM_MAX_NUM_SEQS` 32 → 64 | LLM 한계가 동시 처리 슬롯 수인가, TPOT를 얼마나 내주는가 |
| `e2a-batch16` | CV 배치 16, 배치 대기 30 ms | GPU 시분할 대기를 큰 배치가 나눠 내는가 |
| `e2b-workers2` | CV 워커 2개 (`CV_WORKERS`) | 해독 병렬화와 GPU 차례 확보로 경합 중 용량이 오르는가 |
| `e3-shed8` | `CV_MAX_QUEUE` 10000 → 8 | 넘치는 요청을 일찍 거절(429)해 받아들인 요청의 SLO를 지키는가 (상한은 지연 예산으로 계산) |
| `e3-shed20` | `CV_MAX_QUEUE` 10000 → 20 | 대조군: 상한이 지연 예산보다 크면 어떻게 되는가 |
| `e4-yolo26` | YOLO11n → YOLO26n | 모델 버전 교체가 추론 시간·용량에 주는 영향 |
| `e5-reliable` | `CV_QUEUE_MODE=reliable` | 워커 장애 때 작업이 유실되지 않는가 |

### 과제 실험 전체를 한 번에 (재현용)

보고서의 모든 결과는 한 세션(같은 GPU, 같은 커밋)에서 아래 한 줄로 만든 것입니다. 실행 목록과 순서는 `experiments/run_all.sh`에 있습니다.

```bash
# EC2 (tmux 안에서, 약 4시간)
make exp-all                       # → results/base, base-h2, e1-seqs64, e2a-batch16, e2b-workers2, e3-shed8, e3-shed20, e4-yolo26, e5-simple, e5-reliable

# Mac (results/ 를 가져온 뒤)
make report EXP=base && make report EXP=base-h2
make compare EXPS="base e1-seqs64"                                  # E1: vLLM 슬롯
make compare EXPS="base e2a-batch16 e2b-workers2 e3-shed8 e3-shed20" # E2, E3: 경합 중 CV
make compare EXPS="base e4-yolo26"                                  # E4: 모델 버전
make fault-report EXPS="e5-simple e5-reliable"                      # E5: 워커 장애
```

### 장애 재현 (워커 장애)

```bash
make use CONF=base        && make fault-worker NAME=e5-simple      # 60초에 워커 종료, 30초 뒤 수동 재시작
make use CONF=e5-reliable && make fault-worker NAME=e5-reliable
make fault-worker NAME=e5-crash MODE=crash                          # 프로세스만 죽여 Docker가 자동 재시작 (sudo)

# Mac
make fault-report EXPS="e5-simple e5-reliable"   # 유실 의심 요청, 되살린 작업, 적체·회복 시간, 경보 시점 + 시간축 그림
```

결과 폴더에는 요청별 기록, 사건 시각(`timeline.csv`), 장애 전후의 큐 길이·워커 수·경보(`prometheus.json`), 그 구간의 워커·게이트웨이 로그가 남습니다.

### 경보 규칙

`monitoring/prometheus/alerts.yml`에 워커 장애(`CVWorkerDown`, `CVWorkerRestarted`), 큐 적체(`CVQueueBacklog`), DLQ, SLO 위반(`CVLatencySLO`, `LLMTTFTSLO`), CV 실패율 경보가 있습니다. Prometheus 화면의 Alerts 탭에서 상태를 봅니다. 알림을 보내는 Alertmanager는 붙이지 않았습니다.

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
| `CV_WORKERS` | CV 워커 컨테이너 수 |
| `CV_MAX_QUEUE` | CV 큐 길이 상한. 넘으면 429로 거절 |

## 제출물

| 제출물 | 위치 |
| --- | --- |
| System Architecture | [docs/architecture.md](docs/system-architecture.md) |
| Technical Report | [docs/technical-report.pdf](docs/technical-report.pdf) |
| Executive Summary | [docs/executive-summary.pdf](docs/executive-summary.pdf) |
| 사용 내역 및 구현 범위 | [docs/ai-usage.md](docs/ai-usage.md) |
