#!/usr/bin/env bash
# Run the config sweep for ONE engine on ONE device (engine runs in its own
# process, so a native crash cannot take down the other engines' data).
#
#   run_engine_sweep.sh <device> <engine> <tag> [threads] [iters]
#
# Writes results/raw-<tag>.log and prints how many result lines were captured.
set -euo pipefail

DEV="$1"; ENG="$2"; TAG="$3"; TH="${4:-4}"; IT="${5:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_SDK_ROOT:-$HOME/.local/share/mise/installs/android-sdk/22.0}"
ADB="$SDK/platform-tools/adb"
LOG="$ROOT/results/raw-$TAG.log"

"$ADB" -s "$DEV" logcat -c
"$ADB" -s "$DEV" shell am force-stop com.example.ppocrbench
sleep 1
"$ADB" -s "$DEV" shell am start -n com.example.ppocrbench/.MainActivity \
  -a com.example.ppocrbench.RUN --ez autostart true --ez configs true \
  --es single "$ENG" --ei threads "$TH" --ei iters "$IT" >/dev/null

echo "launched $ENG on $DEV (tag=$TAG)"
DONE=0
for i in $(seq 1 180); do
  sleep 5
  if "$ADB" -s "$DEV" logcat -d | grep -q "saved ->"; then
    echo "DONE after $((i * 5))s"
    DONE=1
    break
  fi
done
"$ADB" -s "$DEV" logcat -d -s PpocrBench:I > "$LOG"
n=$(grep -c '"ok"' "$LOG" 2>/dev/null || echo 0)
echo "captured $TAG = $n results (DONE=$DONE) -> $LOG"
