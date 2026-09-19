#!/usr/bin/env bash
# Entry point for `run_docker.sh up-headless`.
#
# Brings up a virtual X display inside the container, then launches the
# simulation against it. Gazebo Classic needs some display to render camera
# sensors even when no window is shown, and this removes any dependency on the
# host having a desktop session.
#
# xvfb-run would be the obvious tool here, but it is a /bin/sh wrapper that
# waits for a USR1 signal from Xvfb. When it ends up as PID 1 of the container
# it never receives that signal and hangs silently, with Xvfb running and the
# real command never started. Driving Xvfb directly avoids the problem.

set -e

DISPLAY_NUM="${DISPLAY_NUM:-99}"
SCREEN="${SCREEN:-1280x1024x24}"

mkdir -p /tmp/.X11-unix /tmp/rt
chmod 1777 /tmp/.X11-unix
chmod 700 /tmp/rt

Xvfb ":${DISPLAY_NUM}" -screen 0 "$SCREEN" -nolisten tcp > /tmp/xvfb.log 2>&1 &

for _ in $(seq 60); do
  [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ] && break
  sleep 0.25
done

if [ ! -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  echo "Xvfb failed to start:" >&2
  cat /tmp/xvfb.log >&2
  exit 1
fi

export DISPLAY=":${DISPLAY_NUM}"
export XDG_RUNTIME_DIR=/tmp/rt

source /opt/ros/humble/setup.bash
source /ws_build/install/setup.bash

exec ros2 launch crc_sim sim.launch.py gui:=false "$@"
