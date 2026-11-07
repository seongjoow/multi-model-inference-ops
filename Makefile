# 자주 쓰는 명령 모음. `make help` 로 목록 확인
COMPOSE     := docker compose
COMPOSE_GPU := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
IMG         ?= data/val2017/000000000139.jpg

.PHONY: help env llm-mac up down logs ps smoke gpu-up gpu-down gpu-logs gpu-ps

help: ## 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

env: ## .env 가 없으면 .env.example 로 만든다
	@test -f .env || (cp .env.example .env && echo ".env created")

llm-mac: ## [Mac] llama.cpp LLM 서버 실행 (다른 터미널에서 켜 두기)
	./scripts/llm_mac.sh

up: env ## [Mac] 게이트웨이·Redis·CV 워커 빌드 후 실행
	$(COMPOSE) up -d --build

down: ## [Mac] 중지
	$(COMPOSE) down

logs: ## [Mac] 로그 보기
	$(COMPOSE) logs -f --tail=100

ps: ## [Mac] 상태 보기
	$(COMPOSE) ps

smoke: ## 헬스체크·LLM·CV 요청이 모두 성공하는지 확인 (IMG=이미지경로)
	./scripts/smoke_test.sh $(IMG)

gpu-up: env ## [EC2] vLLM 포함 GPU 구성 실행
	$(COMPOSE_GPU) up -d --build

gpu-down: ## [EC2] 중지
	$(COMPOSE_GPU) down

gpu-logs: ## [EC2] 로그 보기
	$(COMPOSE_GPU) logs -f --tail=100

gpu-ps: ## [EC2] 상태 보기
	$(COMPOSE_GPU) ps
