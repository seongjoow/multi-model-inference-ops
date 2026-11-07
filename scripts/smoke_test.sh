#!/usr/bin/env bash
# 배포 직후 기본 동작 확인: 헬스체크 → LLM(일반/스트리밍) → CV(동기/비동기)
# 사용: ./scripts/smoke_test.sh [이미지 경로]   (GATEWAY_URL 로 주소 변경 가능)
set -euo pipefail

BASE="${GATEWAY_URL:-http://localhost:8000}"
IMG="${1:-data/val2017/000000000139.jpg}"
PY="$(command -v python3 || command -v python)"

step() { printf '\n== %s\n' "$*"; }
summarize_cv() {
  "$PY" -c 'import json,sys
r = json.load(sys.stdin)
if "result" in r and r["result"]: r = r["result"]
labels = [d["label"] for d in r.get("detections", [])]
print(json.dumps({"status": r.get("status"), "model": r.get("model"), "num_detections": r.get("num_detections"),
                  "labels": labels[:8], "timing": r.get("timing"), "gateway": r.get("gateway")}, ensure_ascii=False))'
}

[ -f "$IMG" ] || { echo "이미지가 없습니다: $IMG  (COCO val2017을 data/ 에 풀거나 경로를 인자로 주세요)"; exit 1; }

step "healthz"
curl -fsS "$BASE/healthz"; echo

readyz_hint() {
  case "$1" in
    *'"redis":"error'*)
      echo "→ Redis에 연결하지 못했습니다. docker compose ps 로 redis 상태를 확인하세요." ;;
    *'"llm":"error: ConnectError'*)
      echo "→ 게이트웨이가 LLM 서버에 연결하지 못했습니다."
      echo "  1) Mac: 다른 터미널에서 make llm-mac 이 켜져 있는지 확인"
      echo "  2) 켜져 있다면 컨테이너에서 Mac으로 접속이 막힌 것: 끄고 LLM_HOST=0.0.0.0 make llm-mac 으로 다시 실행"
      echo "  3) EC2: make gpu-ps 에서 vllm 이 healthy 인지 확인" ;;
    *'"llm":"error'*)
      echo "→ LLM 서버가 2초 안에 응답하지 않았습니다. LLM 서버 터미널의 로그를 확인하세요." ;;
    *'"llm":"http 503'*)
      echo "→ LLM 서버가 아직 모델을 불러오는 중입니다. 잠시 뒤 다시 실행하세요." ;;
    *'"llm":"http'*)
      echo "→ LLM 서버가 오류 코드를 돌려줬습니다. llm_base_url 이 맞는지 확인하세요." ;;
    '')
      echo "→ 게이트웨이가 응답하지 않습니다. make ps / make logs 로 확인하세요." ;;
  esac
}

step "readyz (Redis, LLM 연결) — 최대 ${READY_TIMEOUT:-30}초 대기"
for i in $(seq 1 "${READY_TIMEOUT:-30}"); do
  BODY=$(curl -sS "$BASE/readyz" 2>/dev/null || true)
  case "$BODY" in *'"ready":true'*) echo "$BODY"; break ;; esac
  if [ "$i" = "${READY_TIMEOUT:-30}" ]; then
    echo "$BODY"; readyz_hint "$BODY"; exit 1
  fi
  sleep 1
done

step "LLM 일반 응답"
curl -fsS "$BASE/v1/llm/chat" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":32}' \
  | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print("gateway:", r["gateway"])'

step "LLM 스트리밍 (마지막 3줄)"
curl -fsSN "$BASE/v1/llm/chat" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":32,"stream":true}' \
  | grep -v '^$' | tail -n 3

step "CV 동기 (wait=true)"
curl -fsS -F "image=@$IMG" "$BASE/v1/cv/detect?wait=true" | summarize_cv

step "CV 비동기 (등록 → 조회)"
JOB=$(curl -fsS -F "image=@$IMG" "$BASE/v1/cv/detect" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "job_id=$JOB"
for _ in $(seq 1 30); do
  STATUS=$(curl -fsS "$BASE/v1/cv/jobs/$JOB" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$STATUS" = "queued" ] || break
  sleep 0.5
done
curl -fsS "$BASE/v1/cv/jobs/$JOB" | summarize_cv

printf '\nSMOKE TEST PASSED\n'
