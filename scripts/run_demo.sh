#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run a HEADLESS demo (no Gazebo window) and collect evidence:
#   - check the four sensor topics
#   - drive the robot with the 'starter' node
#   - save frames from the robot camera and the sky camera
#
#   bash run_demo.sh              # 25 seconds
#   DURATION=60 bash run_demo.sh  # 60 seconds
#
# Results are written to /tmp/crc_shots (or $OUT).
# ---------------------------------------------------------------------------
set -u

# Workspace root. Derived from where this script lives, so the pack works
# no matter which directory you unpacked it into. Override with WS=...
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(dirname "$SELF")}"
OUT="${OUT:-/tmp/crc_shots}"
# Pre-declare: the EXIT trap references these, and under set -u an early
# failure would abort cleanup before it killed gzserver.
LAUNCH_PID=""
DRIVE_PID=""
DURATION="${DURATION:-25}"
TRACK_SCALE="${TRACK_SCALE:-1.0}"
LOG=/tmp/crc_demo

rm -rf "$OUT" "$LOG"; mkdir -p "$OUT" "$LOG"

DISTRO="${ROS_DISTRO:-}"
[ -z "$DISTRO" ] && for d in humble foxy iron jazzy; do [ -d "/opt/ros/$d" ] && DISTRO=$d && break; done
source "/opt/ros/$DISTRO/setup.bash"
# Inside the container the workspace is built into /ws_build, not $WS.
if [ -f /ws_build/install/setup.bash ]; then
  source /ws_build/install/setup.bash
else
  source "$WS/install/setup.bash"
fi
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

# Gazebo Classic needs some X display to render camera sensors even with no
# window, so on a headless machine this script must bring its own.
if [ -z "${DISPLAY:-}" ]; then
  command -v Xvfb >/dev/null || {
    echo "!! No DISPLAY and Xvfb is not installed: the cameras will never"
    echo "   publish. Install it with: sudo apt-get install -y xvfb"
    exit 1
  }
  Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp > /tmp/crc_xvfb.log 2>&1 &
  XVFB_PID=$!
  export DISPLAY=:99
  sleep 2
fi

cleanup() {
  echo ">> cleaning up..."
  kill ${LAUNCH_PID:-} ${DRIVE_PID:-} 2>/dev/null
  sleep 1
  [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null
  pkill -9 -f gzserver  2>/dev/null
  pkill -9 -f gzclient  2>/dev/null
  pkill -9 -f robot_state_publisher 2>/dev/null
  pkill -9 -f "crc_sim starter" 2>/dev/null
  true
}
trap cleanup EXIT

echo "=== [1/5] Starting Gazebo (headless, track_scale=$TRACK_SCALE) ==="
ros2 launch crc_sim sim.launch.py gui:=false track_scale:="$TRACK_SCALE" \
     > "$LOG/launch.log" 2>&1 &
LAUNCH_PID=$!

echo "=== [2/5] Waiting for the sensors (up to 90s) ==="
for i in $(seq 1 90); do
  T=$(ros2 topic list 2>/dev/null)
  if echo "$T" | grep -q '^/scan$' && echo "$T" | grep -q '^/sky_cam/image_raw$'; then
    echo "    OK after ${i}s"
    break
  fi
  if ! kill -0 $LAUNCH_PID 2>/dev/null; then
    echo "!!! launch died. Last 40 log lines:"; tail -40 "$LOG/launch.log"; exit 1
  fi
  sleep 1
done
sleep 3

echo "=== [3/5] Checking the sensors ==="
timeout 6 ros2 run crc_sim sensor_check 2>&1 | tail -12 | tee "$LOG/sensors.txt"

echo "=== [4/5] Driving for ${DURATION}s and capturing frames ==="
ros2 run crc_sim starter > "$LOG/starter.log" 2>&1 &
DRIVE_PID=$!

N=$(( DURATION / 5 ))
[ "$N" -lt 3 ] && N=3
ros2 run crc_sim snapshot --ros-args \
    -p out_dir:="$OUT" -p period:=5.0 -p count:="$N" \
    2>&1 | tail -30

kill $DRIVE_PID 2>/dev/null

echo "=== [5/5] Robot pose after the run ==="
timeout 4 ros2 topic echo /odom --once 2>/dev/null \
  | sed -n '/position/,/orientation/p' | tee "$LOG/final_odom.txt"

echo
echo "================= RESULTS ================="
ls -la "$OUT"
echo "Log: $LOG"
echo "==========================================="
