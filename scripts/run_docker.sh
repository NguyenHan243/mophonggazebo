#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the UEH CRC 2026 simulation in Docker, showing the GUI on the Linux
# desktop you are sitting at.
# Verified on Ubuntu 24.04 + GNOME/Wayland (XWayland) + Intel integrated GPU.
#
#   bash run_docker.sh build        # build the image (first time, ~10 min)
#   bash run_docker.sh compile      # colcon build workspace
#   bash run_docker.sh up           # run Gazebo WITH the GUI
#   bash run_docker.sh up-headless  # no GUI, no host X server needed
#   bash run_docker.sh sh           # open a shell inside the container
#   bash run_docker.sh demo         # drive the robot + save frames to ~/crc_shots
#   bash run_docker.sh down         # stop
#
# Environment overrides: WS, IMAGE, DISPLAY
# ---------------------------------------------------------------------------
set -u

# Workspace root. Derived from where this script lives, so the pack works
# no matter which directory you unpacked it into. Override with WS=...
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(dirname "$SELF")}"
IMAGE="${IMAGE:-crc_sim:humble}"
NAME=crc

# --- Locate the Xauthority file ---------------------------------------------
# GNOME/Wayland: XWayland creates .mutter-Xwaylandauth.* in XDG_RUNTIME_DIR
# Plain X11    : uses ~/.Xauthority
find_xauth() {
  local f rt="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  # Whatever the session already told us wins.
  [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ] && { echo "$XAUTHORITY"; return; }
  f=$(ls "$rt"/.mutter-Xwaylandauth* 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return; }
  # GNOME/Xorg under GDM keeps the cookie here and leaves ~/.Xauthority absent.
  [ -f "$rt/gdm/Xauthority" ] && { echo "$rt/gdm/Xauthority"; return; }
  [ -f "$HOME/.Xauthority" ] && { echo "$HOME/.Xauthority"; return; }
  echo ""
}

gui_args() {
  local xa; xa=$(find_xauth)
  local a=(-e "DISPLAY=${DISPLAY:-:0}"
           -e XDG_RUNTIME_DIR=/tmp/rt
           -e QT_X11_NO_MITSHM=1
           -v /tmp/.X11-unix:/tmp/.X11-unix)
  [ -n "$xa" ] && a+=(-e XAUTHORITY=/tmp/.Xauth -v "$xa":/tmp/.Xauth:ro)
  # Only pass the GPU through if the host actually has one. Adding
  # --device /dev/dri on a machine without it makes docker run fail outright,
  # which is what happens inside most virtual machines.
  [ -d /dev/dri ] && a+=(--device /dev/dri)
  printf '%s\n' "${a[@]}"
}

common_args() {
  # PYTHONDONTWRITEBYTECODE stops the container dropping root-owned
  # __pycache__ directories into the mounted source tree, which you would
  # then need sudo to delete.
  printf '%s\n' --net=host --ipc=host \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$WS/src":/ws/src \
    -v "$WS/scripts":/ws/scripts \
    -v crc_build:/ws_build
}

SRC='source /opt/ros/humble/setup.bash && source /ws_build/install/setup.bash'

case "${1:-up}" in

  build)
    cd "$WS/docker" && docker build -t "$IMAGE" .
    ;;

  compile)
    mapfile -t CA < <(common_args)
    docker run --rm "${CA[@]}" "$IMAGE" bash -lc \
      "source /opt/ros/humble/setup.bash && cd /ws && colcon build --symlink-install \
       --build-base /ws_build/build --install-base /ws_build/install"
    ;;

  up)
    # Uses the X server of the desktop you are sitting at, so the Gazebo window
    # appears on your screen.
    docker rm -f "$NAME" >/dev/null 2>&1
    mapfile -t CA < <(common_args)
    mapfile -t GA < <(gui_args)
    docker run -d --name "$NAME" "${CA[@]}" "${GA[@]}" "$IMAGE" bash -lc \
      "mkdir -p /tmp/rt && chmod 700 /tmp/rt && $SRC && ros2 launch crc_sim sim.launch.py gui:=true"
    echo "Started. Follow the log with:  docker logs -f $NAME"
    ;;

  up-headless)
    # No host X server and no GPU needed: the entry point starts its own Xvfb
    # inside the container. See scripts/headless_entrypoint.sh.
    docker rm -f "$NAME" >/dev/null 2>&1
    mapfile -t CA < <(common_args)
    # DISPLAY is exported inside the entry-point process only, so a later
    # `run_docker.sh sh` would have none and rqt_image_view / rviz could not
    # reach the Xvfb server. Put it in the container environment instead.
    docker run -d --name "$NAME" "${CA[@]}" -e DISPLAY=":99" "$IMAGE" \
      bash /ws/scripts/headless_entrypoint.sh
    echo "Started headless. Follow the log with:  docker logs -f $NAME"
    ;;

  sh)
    docker exec -it "$NAME" bash -lc "$SRC && exec bash"
    ;;

  demo)
    docker exec -d "$NAME" bash -lc "$SRC && ros2 run crc_sim starter"
    sleep 2
    docker exec "$NAME" bash -lc \
      "$SRC && rm -rf /tmp/shots && ros2 run crc_sim snapshot --ros-args \
       -p out_dir:=/tmp/shots -p period:=8.0 -p count:=4"
    # Copy first, replace second, so a failed demo does not destroy the frames
    # from the previous one.
    rm -rf "$HOME/crc_shots.new"
    if docker cp "$NAME":/tmp/shots "$HOME/crc_shots.new"; then
      rm -rf "$HOME/crc_shots"; mv "$HOME/crc_shots.new" "$HOME/crc_shots"
      echo "Frames saved in $HOME/crc_shots:"; ls -la "$HOME/crc_shots"
    else
      echo "No frames copied; $HOME/crc_shots is untouched." >&2
    fi
    ;;

  down)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped."
    ;;

  *)
    echo "Usage: bash run_docker.sh {build|compile|up|up-headless|sh|demo|down}"
    exit 1
    ;;
esac
