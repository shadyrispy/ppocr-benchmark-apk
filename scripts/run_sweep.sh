#!/usr/bin/env bash
# Run one benchmark sweep on one device and capture the logcat output.
#
#   run_sweep.sh <device> <tag> [--ez pipeline true] [--ei threads 4] ...
#
# Writes results/raw-<tag>.log and prints how many results were captured.
set -euo pipefail

DEV="$1"; shift
TAG="$1"; shift

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_SDK_ROOT:-$HOME/.local/share/mise/installs/android-sdk/22.0}"
ADB="$SDK/platform-tools/adb"
LOG="$ROOT/results/raw-$TAG.log"

# Wait for the app to finish: it logs "saved ->" once the result file is out.
# GPU backends add shader compilation, so allow a generous window.
MAX_WAIT="${MAX_WAIT:-900}"

"$ADB" -s "$DEV" logcat -c
"$ADB" -s "$DEV" shell am force-stop com.example.ppocrbench
sleep 1
"$ADB" -s "$DEV" shell am start -n com.example.ppocrbench/.MainActivity \
  -a com.example.ppocrbench.RUN --ez autostart true "$@" >/dev/null

echo "started on $DEV (tag=$TAG)"
for ((i = 0; i < MAX_WAIT / 5; i++)); do
  sleep 5
  if "$ADB" -s "$DEV" logcat -d | grep -q "saved ->"; then
    echo "finished after $((i * 5 + 5))s"
    break
  fi
done

"$ADB" -s "$DEV" logcat -d > "$LOG"
# -s <tag> filtering drops lines when the process dies, so filter afterwards.
grep -E "PpocrBench" "$LOG" > "$LOG.tmp" || true
mv "$LOG.tmp" "$LOG"

n=$(grep -c '"ok"' "$LOG" || true)
echo "captured $TAG = $n results -> $LOG"
