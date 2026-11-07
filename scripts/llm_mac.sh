#!/usr/bin/env bash
# Mac 개발용 LLM 서버 (llama.cpp, Metal GPU). 게이트웨이 컨테이너는 host.docker.internal:8080 으로 접속한다.
# 보고서용 측정은 EC2(vLLM)에서 하고, 이 서버는 기능 확인용으로만 쓴다.
set -euo pipefail

MODEL="${LLM_GGUF:-Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M}"
PORT="${LLM_PORT:-8080}"
PARALLEL="${LLM_PARALLEL:-4}"   # 동시에 처리할 요청 슬롯 수
CTX="${LLM_CTX:-16384}"         # 전체 컨텍스트 (슬롯마다 CTX/PARALLEL 씩 나눠 씀)
HOST="${LLM_HOST:-127.0.0.1}"   # 컨테이너에서 접속이 안 되면 LLM_HOST=0.0.0.0 (같은 Wi-Fi의 다른 기기에도 열림)

exec llama-server -hf "$MODEL" --host "$HOST" --port "$PORT" --parallel "$PARALLEL" -c "$CTX" --metrics
