#!/usr/bin/env bash
# digitizer local — 원샷 설치·실행 (macOS / Linux)
#   bash setup.sh          # Python + numpy·Pillow(없으면 venv) + VLM 서버 탐색(선택) + selftest + 웹 서버 + 브라우저
#   bash setup.sh stop
# 환경변수: LLM_BASE_URL / LLM_API / LLM_MODEL·VISION_MODEL (AI 자동 보정용 — 이미지 입력 모델, 없어도 동작), PORT (8783), WORKSPACE (그림·작업 이력)
# selftest 는 스스로 WORKSPACE 를 임시 폴더로 바꿔 돌고 지운다 (실데이터 폴더에 흔적 없음).
# 폐쇄망: wheels/ 폴더에 numpy·pillow(·openpyxl) 휠을 넣어 두면 인터넷 없이 설치. PDF 는 pdftoppm(poppler-utils) 이 있으면 됨.
set -euo pipefail
PORT="${PORT:-8783}"
if [ -t 1 ]; then B=$'\033[1m'; D=$'\033[2m'; C=$'\033[36m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; N=$'\033[0m'; else B= D= C= G= R= Y= N=; fi
STEP=0; step() { STEP=$((STEP+1)); printf '  %s[%d/6]%s %s%-14s%s ' "$D" "$STEP" "$N" "$B" "$1" "$N"; }
ok() { printf '%s✔%s %s\n' "$G" "$N" "${1:-}"; }; skip() { printf '%s–%s %s\n' "$D" "$N" "${1:-}"; }; warn() { printf '%s!%s %s\n' "$Y" "$N" "${1:-}"; }
die() { printf '%s✘ %s%s\n\n' "$R" "$*" "$N" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }; probe() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }
wait_for() { for _ in $(seq 1 "${2:-30}"); do probe "$1" && return 0; sleep 1; done; return 1; }
printf '\n%s  digitizer local%s  그래프 그림 → 데이터 — 축 보정(AI 보조) · 색으로 곡선·점 추출 · CSV/엑셀\n\n' "$B" "$N"

step "OS 감지"; case "$(uname -s)" in Darwin*) OS=mac ;; Linux*) OS=linux ;; MINGW*|MSYS*|CYGWIN*) OS=windows ;; *) die "지원하지 않는 OS: $(uname -s)" ;; esac; ok "$OS ($(uname -m))"
step "패키지 확보"; cd "$(dirname "${BASH_SOURCE[0]}")"; [ -f app.py ] || die "app.py 가 없습니다"; ok "$(pwd)"
if [ "${1:-}" = "stop" ]; then [ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null && rm -f .server.pid && ok "웹 서버 종료" || skip "실행 중인 서버 없음"; exit 0; fi

step "Python+numpy"
PY=""; for c in python3 python py; do has "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.9+ 가 없습니다 (sudo apt install python3 / brew install python)"
if [ -x venv/bin/python ]; then PY=venv/bin/python
elif ! "$PY" -c 'import numpy, PIL, sys; sys.exit(0 if int(PIL.__version__.split(".")[0]) >= 9 else 1)' 2>/dev/null; then
  if has uv; then uv venv -q --python "$(command -v $PY)" venv; else "$PY" -m venv venv; fi
  if [ -d wheels ]; then venv/bin/python -m pip install -q --no-index --find-links wheels -r requirements.txt \
       || venv/bin/python -m pip install -q --no-index --find-links wheels numpy pillow
  elif has uv; then VIRTUAL_ENV="$PWD/venv" uv pip install -q -r requirements.txt; else venv/bin/python -m pip install -q -r requirements.txt; fi \
    || die "numpy·Pillow 설치 실패 (폐쇄망이면 wheels/ 에 numpy·pillow 휠을 넣고 재실행)"
  PY=venv/bin/python
fi
EXTRA=$("$PY" -c 'import importlib.util as u; print(" ".join(m for m in ("openpyxl","matplotlib","scipy") if u.find_spec(m)))' 2>/dev/null || true)
PDFT=$(has pdftoppm && echo "pdftoppm" || ("$PY" -c 'import fitz' 2>/dev/null && echo "PyMuPDF" || echo "PDF 변환 없음"))
ok "$($PY --version 2>&1) + numpy $($PY -c 'import numpy; print(numpy.__version__)') · ${EXTRA:-추가 모듈 없음} · $PDFT"

step "VLM (선택)"
export LLM_API="${LLM_API:-}" LLM_BASE_URL="${LLM_BASE_URL:-}" LLM_MODEL="${LLM_MODEL:-}"
if [ -n "$LLM_BASE_URL" ]; then [ -n "$LLM_API" ] || { case "$LLM_BASE_URL" in *1143*) LLM_API=ollama ;; *) LLM_API=openai ;; esac; }
elif probe http://localhost:11434/api/tags; then LLM_API=ollama LLM_BASE_URL=http://localhost:11434
else for p in 8000 1234 8080; do probe "http://localhost:$p/v1/models" && { LLM_API=openai LLM_BASE_URL="http://localhost:$p/v1"; break; }; done; fi
if [ -n "$LLM_BASE_URL" ] && [ -z "$LLM_MODEL" ] && [ -z "${VISION_MODEL:-}" ]; then
  if [ "$LLM_API" = ollama ]; then LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/api/tags" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["models"][0]["name"])' 2>/dev/null || true)
  else LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/models" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true); fi
fi
if [ -n "$LLM_BASE_URL" ]; then ok "$LLM_API $LLM_BASE_URL ${VISION_MODEL:-$LLM_MODEL} (이미지 입력이 되는 모델이어야 함)"
else skip "없음 → 'AI 자동 보정'만 꺼짐 (눈금 값 직접 입력·수동 보정·추출은 그대로)"; LLM_API=ollama; fi

step "자가검증"; env -u WORKSPACE "$PY" selftest.py >/dev/null 2>&1 || die "selftest 실패 (python3 selftest.py 로 확인)"; ok "합성 그래프(선형·로그·2계열·산점·막대) 보정·추출 정확도 (임시 폴더에서)"

step "웹 서버"
[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
LLM_API=$LLM_API LLM_BASE_URL=$LLM_BASE_URL LLM_MODEL=$LLM_MODEL PORT=$PORT nohup "$PY" app.py > server.log 2>&1 & echo $! > .server.pid
wait_for "http://localhost:$PORT/api/health" 20 || { cat server.log; die "웹 서버 기동 실패 (server.log 확인)"; }
URL="http://localhost:$PORT"; ok "$URL"
case "$OS" in mac) open "$URL" ;; linux) [ -n "${WORKSPACE:-}" ] || { has xdg-open && xdg-open "$URL" >/dev/null 2>&1 || true; } ;; esac
printf '\n  %s준비 완료%s  %s%s%s   종료: bash setup.sh stop   로그: server.log\n\n' "$B" "$N" "$C" "$URL" "$N"
