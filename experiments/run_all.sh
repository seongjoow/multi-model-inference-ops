#!/usr/bin/env bash
# 과제 실험 전체를 한 세션(같은 GPU, 같은 코드)에서 순서대로 실행한다. 약 4시간.
#   make exp-all        (= bash experiments/run_all.sh 2>&1 | tee results/run_all.log)
# 설정 적용에 실패하면 그 실험만 건너뛰고 다음으로 간다 (로그에 "!!! 실패").
# 순서: 기준 설정 실험을 먼저 모으고, vLLM 재시작이 필요한 실험(E1)은 뒤로 보내 재시작 횟수를 줄인다.
cd "$(dirname "$0")/.."
C="--duration 45 --yes"
CV_SWEEP="--scenarios contention-cv-sweep --llm-fixed 4 --cv-rates 25,40,50,75,100"
run() { local conf=$1; shift; echo "=== $(date +%T) CONF=$conf : $*"
        make use CONF="$conf" && "$@" || echo "!!! 실패: CONF=$conf $*"; }

# 1. Baseline + 경합 (기준 설정)
run base make exp EXP=base    ARGS="--scenarios llm-only,cv-only,contention-cv-sweep --llm-rates 2,4,8,10,12,14,16 --cv-rates 25,40,50,75,100 --llm-fixed 4 $C"
run base make exp EXP=base-h2 ARGS="--scenarios contention-llm-sweep --llm-rates 0.25,0.5,1,2,4,8,12 --cv-fixed 25 $C"
# 2. 장애 재현 (E5). reliable 은 워커 교체 경보(2분 창)가 지나간 뒤 시작
run base        make fault-worker NAME=e5-simple
run e5-reliable bash -c "sleep 150 && make fault-worker NAME=e5-reliable"
# 3. CV 쪽 설정 (vLLM 재시작 없음)
run e2a-batch16  make exp EXP=e2a-batch16  ARGS="$CV_SWEEP $C"
run e2b-workers2 make exp EXP=e2b-workers2 ARGS="$CV_SWEEP $C"
run e3-shed8     make exp EXP=e3-shed8     ARGS="$CV_SWEEP $C"
run e4-yolo26    make exp EXP=e4-yolo26    ARGS="--scenarios cv-only --cv-rates 25,40,50,75,100 $C"
# 4. LLM 설정 (vLLM 재시작)
run e1-seqs64 make exp EXP=e1-seqs64 ARGS="--scenarios llm-only --llm-rates 2,4,8,10,12,14,16 $C"
# 5. 대조군: 큐 상한 20 (상한 크기에 따른 차이)
run e3-shed20 make exp EXP=e3-shed20 ARGS="$CV_SWEEP $C"

make use CONF=base
echo "=== $(date +%T) 끝"
