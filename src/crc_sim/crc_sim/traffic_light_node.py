#!/usr/bin/env python3
"""
Traffic light controller for the simulation.

It does two things at once:
  1. CHANGES THE LAMP COLOUR in Gazebo, so the robot camera can SEE it.
     This is what your vision code has to deal with.
  2. PUBLISHES THE STATE on ROS topics as ground truth, for debugging.
     Using these topics during the graded run is NOT allowed.

    ros2 run crc_sim traffic_light
    ros2 run crc_sim traffic_light --ros-args -p backend:=none    # topics only
    ros2 run crc_sim traffic_light --ros-args -p config:=/path/to/lights.yaml

Each light runs RANDOMLY and INDEPENDENTLY: every phase change draws its
duration from the range declared in config/traffic_lights.yaml. The `seed`
parameter replays the exact same scenario (default 0 = fixed); `seed:=-1`
gives a different scenario on every run.

Published topics:
    /traffic_light/<id>/state   std_msgs/String   "RED" | "YELLOW" | "GREEN"
    /traffic_lights             std_msgs/String   JSON summary (id, colour, seconds left)
    /automobile/semaphores      std_msgs/String   alias named after the official BFMC topic

How the colour is changed (`backend` parameter):
    state : default. Moves a glowing sphere onto the lamp through the
            /set_entity_state service. Latency is milliseconds.
    gz    : calls `gz topic -p /gazebo/<world>/visual` to change the
            material directly. Correct, but ~2 seconds per call.
    none  : does not touch Gazebo, publishes topics only.
"""

import json
import math
import signal
import os
import random
import subprocess

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import GetEntityState, SetEntityState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

# Materials shipped with Gazebo Classic (gazebo.material)
MAT_ON = {'red': 'Gazebo/RedGlow',
          'yellow': 'Gazebo/YellowGlow',
          'green': 'Gazebo/GreenGlow'}
MAT_OFF = 'Gazebo/FlatBlack'

ORDER = ('green', 'yellow', 'red')      # phase order


class GzVisualBackend:
    """Change a visual material by publishing on the gazebo transport bus.

    Correct syntax (Gazebo Classic 11):
        gz topic -p /gazebo/<world>/visual -m '<protobuf text message>'

    The message must contain BOTH `name` (fully scoped visual name) AND
    `parent_name` (the link that owns it), otherwise Gazebo silently
    ignores it. `gz topic` exits 0 even on error, so stdout and stderr
    must be inspected.
    """

    def __init__(self, world, logger):
        self.topic = f'/gazebo/{world}/visual'
        self.log = logger

    @staticmethod
    def _msg(visual, parent, material):
        return (f'name: "{visual}" parent_name: "{parent}" '
                f'material {{ script {{ name: "{material}" }} }}')

    def set(self, visual, parent, material):
        cmd = ['gz', 'topic', '-p', self.topic,
               '-m', self._msg(visual, parent, material)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log.warn(f'gz topic failed: {e}')
            return False
        out = (r.stdout or '') + (r.stderr or '')
        if r.returncode != 0 or 'Error' in out or 'Invalid' in out:
            self.log.warn(f'gz topic error: {out.strip()[:160]}')
            return False
        return True


class EntityStateBackend:
    """Switch lamps by MOVING a glowing sphere into place.

    Every light owns three spheres (Red/Yellow/Green) parked at z = -5.
    Turning a colour on moves that sphere onto the lamp and pushes the
    other two back to z = -5.

    Much faster than the 'gz' backend: a ROS service call takes
    milliseconds, while `gz topic -p` costs about 2 seconds per call
    (Gazebo master handshake) - long enough to swallow the whole 2 s
    yellow phase.
    """

    HIDE_Z = -5.0
    # Lamp positions in the light model frame (matches worlds/*.xacro)
    LOCAL = {'red': (0.026, 0.0, 0.325),
             'yellow': (0.026, 0.0, 0.285),
             'green': (0.026, 0.0, 0.245)}
    BULGE = 0.005      # push the sphere slightly in front of the lens

    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        self.set_cli = node.create_client(SetEntityState, '/set_entity_state')
        self.get_cli = node.create_client(GetEntityState, '/get_entity_state')
        self.world_pos = {}          # (model, colour) -> (x, y, z)

    def ready(self, timeout=10.0):
        return (self.set_cli.wait_for_service(timeout_sec=timeout)
                and self.get_cli.wait_for_service(timeout_sec=timeout))

    def _call(self, client, req, timeout=3.0):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        return fut.result()

    def resolve(self, model):
        """Read the pole pose from Gazebo and derive the three lamp positions."""
        req = GetEntityState.Request()
        req.name = model
        res = self._call(self.get_cli, req)
        if res is None or not res.success:
            self.log.error(f'Could not read the pose of {model}')
            return False

        p = res.state.pose.position
        q = res.state.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)

        for colour, (lx, ly, lz) in self.LOCAL.items():
            lx2 = lx - self.BULGE
            self.world_pos[(model, colour)] = (p.x + lx2 * c - ly * s,
                                               p.y + lx2 * s + ly * c,
                                               p.z + lz)
        self.log.info(f'{model}: yaw={yaw:+.3f}, lamp positions resolved')
        return True

    def _move(self, name, xyz):
        """ASYNCHRONOUS call - must not wait for the result.

        This runs inside a timer callback, i.e. in the middle of a spin.
        Calling spin_until_future_complete here would be a nested spin
        and would deadlock. The result is not urgent: a failure shows up
        immediately on screen.
        """
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose.position.x = float(xyz[0])
        req.state.pose.position.y = float(xyz[1])
        req.state.pose.position.z = float(xyz[2])
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'
        self.set_cli.call_async(req)
        return True

    def show(self, model, colour):
        ok = True
        for c in ORDER:
            name = f'{model}_glow_{c.capitalize()}'
            xyz = (self.world_pos[(model, c)] if c == colour
                   else (0.0, 0.0, self.HIDE_Z))
            ok &= self._move(name, xyz)
        return ok


class NullBackend:
    def set(self, *_a, **_k):
        return True


class TrafficLights(Node):

    def __init__(self):
        super().__init__('traffic_light')
        default_cfg = os.path.join(
            get_package_share_directory('crc_sim'), 'config', 'traffic_lights.yaml')

        self.declare_parameter('config', default_cfg)
        self.declare_parameter('backend', 'state')   # state | gz | none
        self.declare_parameter('world', 'crc_track')
        self.declare_parameter('rate', 5.0)
        # Seed for the random generator. The same seed replays the same
        # scenario, which is convenient for grading. Use -1 for a fresh one.
        self.declare_parameter('seed', 0)

        cfg_path = self.get_parameter('config').value
        backend = self.get_parameter('backend').value
        world = self.get_parameter('world').value
        rate = float(self.get_parameter('rate').value)

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.lights = cfg['lights']

        self.kind = backend
        if backend == 'state':
            self.backend = EntityStateBackend(self)
            if not self.backend.ready():
                self.get_logger().error(
                    'Service /set_entity_state not found. The world must load '
                    'libgazebo_ros_state.so. Falling back to backend=none.')
                self.kind, self.backend = 'none', NullBackend()
            else:
                for lt in self.lights:
                    self.backend.resolve(lt['model'])
        elif backend == 'gz':
            self.backend = GzVisualBackend(world, self.get_logger())
        else:
            self.backend = NullBackend()

        self.pubs = {lt['id']: self.create_publisher(
            String, f"/traffic_light/{lt['id']}/state", 10) for lt in self.lights}
        self.pub_all = self.create_publisher(String, '/traffic_lights', 10)
        self.pub_bfmc = self.create_publisher(String, '/automobile/semaphores', 10)

        seed = int(self.get_parameter('seed').value)
        self.rng = random.Random(None if seed < 0 else seed)

        # Every light is an INDEPENDENT state machine holding its current
        # colour and the time of its next change. There is no fixed cycle
        # or phase offset: the lights are required to run randomly.
        self.shown = {}                       # id -> colour currently shown
        self.state = {}                       # id -> (colour, next change time)
        self.t0 = self.now()
        for lt in self.lights:
            # Start on GREEN or RED only - yellow is a transition phase, not
            # a state to begin in. Also cut into the phase at a random point
            # so the lights do not all switch at the same instant.
            colour = self.rng.choice(('green', 'red'))
            self.state[lt['id']] = (
                colour, self.now() + self.rng.uniform(0.2, self.draw(lt, colour)))
        self.create_timer(1.0 / rate, self.tick)

        names = ', '.join(lt['id'] for lt in self.lights)
        self.get_logger().info(
            f'{len(self.lights)} lights ({names}) | backend={backend} | cfg={cfg_path}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def draw(self, light, colour):
        """Draw a random duration (seconds) for this light's `colour` phase."""
        lo, hi = light['range'][colour]
        return self.rng.uniform(float(lo), float(hi))

    def phase(self, light, now):
        """Advance the state machine if due; return (colour, seconds left)."""
        lid = light['id']
        colour, until = self.state[lid]
        while now >= until:
            colour = ORDER[(ORDER.index(colour) + 1) % len(ORDER)]
            until += self.draw(light, colour)
        self.state[lid] = (colour, until)
        return colour, until - now

    def tick(self):
        now = self.now()
        summary = []

        for lt in self.lights:
            colour, left = self.phase(lt, now)
            lid = lt['id']

            msg = String()
            msg.data = colour.upper()
            self.pubs[lid].publish(msg)
            summary.append({'id': lid, 'color': colour.upper(),
                            'seconds_left': round(left, 1)})

            if self.shown.get(lid) != colour:
                self.apply(lt, colour)
                self.shown[lid] = colour
                self.get_logger().info(f'{lid} -> {colour.upper()}')

        payload = String()
        payload.data = json.dumps(summary)
        self.pub_all.publish(payload)
        self.pub_bfmc.publish(payload)

    def apply(self, light, colour):
        """Turn on the `colour` lamp and turn the other two off."""
        model = light['model']
        if self.kind == 'state':
            ok = self.backend.show(model, colour)
        elif self.kind == 'gz':
            parent = f'{model}::link'
            ok = True
            for c in ORDER:
                ok &= self.backend.set(f'{parent}::lamp_{c}_v', parent,
                                       MAT_ON[c] if c == colour else MAT_OFF)
        else:
            ok = True
        if not ok:
            self.get_logger().warn(
                f'Could not change the lamp colour of {model}. '
                'Topics keep working; see README section 7.')


def catch_sigterm():
    """Turn SIGTERM into a flag instead of letting rclpy tear down the
    context. Must run before the node is built.
    """
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            # Shutting down mid-callback is normal; anything else is not.
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = TrafficLights()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
