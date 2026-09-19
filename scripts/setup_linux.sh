#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install dependencies and build the UEH CRC 2026 simulation.
# Auto-detects the ROS 2 distro present (foxy / humble / iron / jazzy).
#
#   bash setup_linux.sh              # install dependencies, then build
#   bash setup_linux.sh --build-only # skip apt, build only
#
# The source must already be in  $WS/src  (default ~/crc_sim_ws/src).
# ---------------------------------------------------------------------------
set -e

# Workspace root. Derived from where this script lives, so the pack works
# no matter which directory you unpacked it into. Override with WS=...
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(dirname "$SELF")}"
BUILD_ONLY=0
[ "${1:-}" = "--build-only" ] && BUILD_ONLY=1

# ---- 1. Find the ROS distro ------------------------------------------------
if [ -n "${ROS_DISTRO:-}" ] && [ -d "/opt/ros/$ROS_DISTRO" ]; then
  DISTRO="$ROS_DISTRO"
else
  DISTRO=""
  for d in humble foxy iron jazzy; do
    [ -d "/opt/ros/$d" ] && DISTRO="$d" && break
  done
fi
if [ -z "$DISTRO" ]; then
  echo "!! No ROS 2 found in /opt/ros. Install ROS 2 first."
  exit 1
fi
echo "=== ROS 2 distro: $DISTRO  ($(lsb_release -ds 2>/dev/null)) ==="

# ---- 2. Dependencies -------------------------------------------------------
if [ "$BUILD_ONLY" -eq 0 ]; then
  echo "=== [1/3] Installing dependencies (needs sudo) ==="
  REQUIRED="
    ros-$DISTRO-gazebo-ros-pkgs
    ros-$DISTRO-gazebo-plugins
    ros-$DISTRO-robot-state-publisher
    ros-$DISTRO-xacro
    ros-$DISTRO-cv-bridge
    ros-$DISTRO-image-transport
    ros-$DISTRO-teleop-twist-keyboard
    python3-colcon-common-extensions
    python3-opencv
  "
  sudo apt-get update
  sudo apt-get install -y $REQUIRED

  # Optional: the TurtleBot3 STL meshes. If the distro does not ship this
  # package the simulation still runs with  use_mesh:=false  (box robot).
  OPTIONAL="ros-$DISTRO-turtlebot3-description ros-$DISTRO-rviz2 ros-$DISTRO-rqt-image-view"
  for p in $OPTIONAL; do
    if apt-cache show "$p" > /dev/null 2>&1; then
      sudo apt-get install -y "$p" || echo "   (skipped $p)"
    else
      echo "   (package $p not in the repository - skipped)"
    fi
  done
else
  echo "=== [1/3] Skipping apt (--build-only) ==="
fi

# ---- 3. Build --------------------------------------------------------------
echo "=== [2/3] Build $WS ==="
[ -d "$WS/src" ] || { echo "!! $WS/src not found"; exit 1; }
cd "$WS"
set +u; source "/opt/ros/$DISTRO/setup.bash"; set -u
colcon build --symlink-install

# ---- 4. bashrc -------------------------------------------------------------
echo "=== [3/3] Updating ~/.bashrc ==="
# Key the guard on THIS workspace, not a fixed directory name, or every
# re-run appends another block to ~/.bashrc.
if ! grep -qF "$WS/install/setup.bash" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF

# --- UEH CRC 2026 simulation ---
source /opt/ros/$DISTRO/setup.bash
source "$WS/install/setup.bash"
export TURTLEBOT3_MODEL=waffle
EOF
fi

echo
echo "=========================================================="
echo " DONE.  Try it:"
echo "   source $WS/install/setup.bash"
echo "   ros2 launch crc_sim sim.launch.py"
echo
echo " No display available (e.g. over SSH):"
echo "   bash $WS/scripts/run_demo.sh"
echo "=========================================================="
