#!/usr/bin/env bash
# openjev/run_training.sh -- durable, resumable, memory-safe LoRA training supervisor.
#
# Why this exists: on 2026-09-18 02:35 the host OOM'd (8/8 GB swap) and took every Hermes
# child process with it, killing an in-flight training run. This supervisor is launched
# DETACHED (setsid nohup) so a backend restart cannot reach it, runs the trainer in short
# wall-clock chunks so progress lands on disk every few minutes, resumes from the last
# checkpoint, and takes a global lock so only ONE model-heavy job is ever in memory at once
# (two concurrent runs are what caused the OOM).
#
# usage:
#   openjev/run_training.sh <run-name> <target-total-steps> [chunk-min] [batch]
# example:
#   openjev/run_training.sh openjev-v2 220 25 6
#
# launch detached:
#   cd ~/Code/jev-repro-test && setsid nohup openjev/run_training.sh openjev-v2 220 25 6 \
#       > openjev/runs/openjev-v2/supervisor.out 2>&1 &
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/Qwen-2.5-1B-RLCD/.venv/bin/python"
RUN="${1:?run-name required}"
TOTAL="${2:?target total steps required}"
CHUNK="${3:-25}"          # minutes per chunk (clean checkpoint + exit)
BATCH="${4:-6}"
LOCK=/tmp/jev-model.lock  # global model lock: shared by every model-heavy job on this box
# optional env overrides so one supervisor can drive any corpus/hyper-params
EXTRA=""
[ -n "${TRAIN_FILE:-}" ] && EXTRA="$EXTRA --train-file $TRAIN_FILE"
[ -n "${LR:-}" ] && EXTRA="$EXTRA --lr $LR"
[ -n "${MAXLEN:-}" ] && EXTRA="$EXTRA --max-len $MAXLEN"
[ -n "${SNAPSHOT_EVERY:-}" ] && EXTRA="$EXTRA --snapshot-every $SNAPSHOT_EVERY"
RUN_DIR="$HERE/runs/$RUN"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/supervisor.log"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "supervisor start run=$RUN target=$TOTAL chunk=${CHUNK}min batch=$BATCH pid=$$"
exec 9>"$LOCK"

for attempt in $(seq 1 400); do
  step=$("$PY" - <<PYEOF 2>/dev/null || echo 0
import json,pathlib
p = pathlib.Path("$RUN_DIR/progress.json")
print(json.loads(p.read_text()).get("step", 0) if p.exists() else 0)
PYEOF
)
  step=${step:-0}
  if [ "$step" -ge "$TOTAL" ]; then say "target reached (step $step >= $TOTAL) -- done"; exit 0; fi
  left=$(( TOTAL - step ))

  # wait up to 90 min for the model lock (another agent's smoke run / eval may hold it)
  if ! flock -w 5400 9; then say "lock busy >90min, retrying in 2min"; sleep 120; continue; fi
  say "attempt $attempt: step $step/$TOTAL, running up to $CHUNK min"
  cd "$ROOT" || exit 1
  nice -n 5 "$PY" openjev/train_v2.py --run-name "$RUN" --resume \
      --steps "$left" --target-total "$TOTAL" --batch "$BATCH" \
      --time-budget-min "$CHUNK" --max-rss-gb 7 $EXTRA 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  flock -u 9
  say "chunk exit=$rc"
  if [ "$rc" -eq 0 ]; then say "training complete at step >= $TOTAL"; exit 0; fi
  [ "$rc" -eq 10 ] && continue          # clean chunk end, more steps wanted
  say "chunk failed rc=$rc -- backing off 60s"; sleep 60
done
say "giving up after 400 attempts"; exit 1
