# 자주 쓰는 명령 모음. `make help` 로 목록 확인
# GPU(nvidia-smi)가 있으면 EC2 구성(vLLM, GPU exporter 포함), 없으면 Mac 구성을 자동으로 쓴다.
COMPOSE_MAC := docker compose
COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
ifeq ($(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes),yes)
  COMPOSE  := $(COMPOSE_GPU)
  PLATFORM := ec2-gpu
else
  COMPOSE  := $(COMPOSE_MAC)
  PLATFORM := mac
endif
LOADGEN := $(COMPOSE) --profile tools run --rm --no-deps loadgen

# 부하 실험 기본값 (make load-llm RATE=4 DURATION=120 NAME=baseline-llm-r4 처럼 바꿔서 실행)
RATE     ?= 2
DURATION ?= 60
NAME     ?=
ARGS     ?=
IMG      ?= data/val2017/000000000139.jpg
EXP      ?= d3
export GIT_COMMIT := $(shell git describe --always --dirty 2>/dev/null || echo unknown)
PYTHON   ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help env llm-mac up down logs ps smoke dash load-llm load-cv load summary exp-pilot exp report use confs compare fault-worker fault-report gpu-up gpu-down gpu-logs gpu-ps

help: ## 명령 목록
	@echo "현재 환경: $(PLATFORM)"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-13s %s\n", $$1, $$2}'

env: ## .env 가 없으면 .env.example 로 만든다
	@test -f .env || (cp .env.example .env && echo ".env created")

llm-mac: ## [Mac] llama.cpp LLM 서버 실행 (다른 터미널에서 켜 두기)
	./scripts/llm_mac.sh

up: env ## 전체 실행 (빌드 포함). EC2면 vLLM·GPU 구성까지
	$(COMPOSE) up -d --build

down: ## 중지
	$(COMPOSE) down

logs: ## 로그 보기
	$(COMPOSE) logs -f --tail=100

ps: ## 상태 보기
	$(COMPOSE) ps

smoke: ## 헬스체크·LLM·CV 요청이 모두 성공하는지 확인 (IMG=이미지경로)
	./scripts/smoke_test.sh $(IMG)

dash: ## 대시보드 주소 안내
	@echo "Grafana:    http://localhost:3000   (대시보드: Inference Ops)"
	@echo "Prometheus: http://localhost:9090   (Status > Target health 에서 수집 대상 확인)"
	@echo "EC2라면 Mac에서: ssh -i ~/.ssh/mminfops.pem -L 3000:localhost:3000 -L 9090:localhost:9090 -L 8000:localhost:8000 ubuntu@<IP>"

load-llm: env ## LLM 부하 (RATE=초당 요청, DURATION=초, NAME=실행 이름, ARGS=추가 옵션)
	@mkdir -p results
	$(LOADGEN) run --workload llm --rate $(RATE) --duration $(DURATION) $(if $(NAME),--name $(NAME)) $(ARGS)

load-cv: env ## CV 부하 (RATE, DURATION, NAME, ARGS 동일)
	@mkdir -p results
	$(LOADGEN) run --workload cv --rate $(RATE) --duration $(DURATION) $(if $(NAME),--name $(NAME)) $(ARGS)

load: env ## 부하 생성기에 옵션을 직접 전달 (ARGS="run --workload cv --concurrency 8")
	@mkdir -p results
	$(LOADGEN) $(ARGS)

summary: ## 결과 비교표 (RUNS="results/a results/b", 생략하면 전체)
	$(LOADGEN) summary $(or $(RUNS),$(sort $(wildcard results/*)))

exp-pilot: env ## [EC2] 실험 시험 실행 (약 12분). 도착률 범위가 맞는지 먼저 확인 (EXP=이름, ARGS=추가 옵션)
	python3 experiments/run.py --name $(EXP)-pilot --repeat 1 --duration 20 --warmup 5 --cooldown 5 $(ARGS)

exp: env ## [EC2] 본 실험 (기본 3회 반복, 약 80분). ARGS="--llm-rates 1,2,4,8 --cv-rates 10,25,50" 처럼 변경
	python3 experiments/run.py --name $(EXP) $(ARGS)

report: ## [Mac] 실험 결과 분석 → results/<EXP>/report/ (표, 그래프, report.md)
	$(PYTHON) analysis/report.py results/$(EXP)

use: env ## [EC2] 실험 설정 적용 후 재시작·준비 대기 (CONF=e1-seqs64, 목록: make confs)
	@test -n "$(CONF)" || (echo "CONF=<설정 이름> 이 필요합니다. 목록: make confs"; exit 1)
	python3 experiments/apply_conf.py $(CONF)
	$(COMPOSE) up -d --build
	python3 experiments/apply_conf.py --wait

confs: ## 실험 설정 목록 (experiments/conf/*.env)
	@python3 experiments/apply_conf.py --list

compare: ## [Mac] 전후 비교 (EXPS="e2-base e2a-batch16", 첫 번째가 기준) → results/compare/...
	$(PYTHON) analysis/compare.py $(addprefix results/,$(EXPS))

fault-worker: env ## [EC2] 장애 재현: 부하 중 CV 워커 강제 종료 (NAME=e5-simple, MODE=down|crash)
	@test -n "$(NAME)" || (echo "NAME=<결과 이름> 이 필요합니다"; exit 1)
	python3 experiments/fault.py --name $(NAME) $(if $(MODE),--mode $(MODE)) $(ARGS)

fault-report: ## [Mac] 장애 재현 분석 (EXPS="e5-simple e5-reliable")
	$(PYTHON) analysis/fault_report.py $(addprefix results/,$(EXPS))

gpu-up: env ## [EC2] GPU 구성 실행 (make up 과 같음)
	$(COMPOSE_GPU) up -d --build

gpu-down: ## [EC2] 중지
	$(COMPOSE_GPU) down

gpu-logs: ## [EC2] 로그 보기
	$(COMPOSE_GPU) logs -f --tail=100

gpu-ps: ## [EC2] 상태 보기
	$(COMPOSE_GPU) ps
