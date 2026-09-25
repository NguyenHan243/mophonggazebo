import math
import signal
import time
import cv2
import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan

try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False


class Starter(Node):

    def __init__(self):
        super().__init__('crc_starter')

        # --- CÁC THAM SỐ ĐIỀU KHIỂN ---
        self.declare_parameter('max_speed', 0.12)
        self.declare_parameter('max_turn', 1.0)
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('rate', 20.0)

        self.max_speed = self.get_parameter('max_speed').value
        self.max_turn = self.get_parameter('max_turn').value
        self.stop_distance = self.get_parameter('stop_distance').value
        rate = self.get_parameter('rate').value

        self.image = None
        self.scan = None
        self.x = self.y = self.yaw = 0.0
        self.pitch = 0.0  # Góc ngẩng/độ nghiêng dốc thân xe (đơn vị rad)
        self._last_log = {}

        self.last_error = 0.0
        self.prev_angular_z = 0.0
        
        # Biến lọc nhiễu Debounce cho Cua gắt
        self.hard_turn_confirm = 0

        # Tham chiếu vị trí tâm line outdoor từ frame trước
        self.last_outdoor_cx = None

        # --- CẤU HÌNH DỐC CẦU (RAMP) ---
        self.is_on_ramp = False
        self.ramp_start_time = None
        self.RAMP_DURATION = 3.5

        # --- CẤU HÌNH HẦM (TUNNEL) & CONTINUITY TRACKING ---
        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 7.0
        self.last_tunnel_cx = None

        # --- WATCHDOG THOÁT BẾ TẮC ---
        self.stuck_since = None
        self.stuck_ref_dist = 0.0
        self.STUCK_TIMEOUT = 1.0

        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.bridge = CvBridge() if HAVE_CV else None

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.create_timer(1.0 / rate, self.tick)

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        
        # Tính Yaw
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        # Tính Pitch (Độ ngẩng thân xe - dùng để phát hiện leo dốc cầu)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.pitch = math.copysign(math.pi / 2, sinp)
        else:
            self.pitch = math.asin(sinp)

    def drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(max(-self.max_speed, min(self.max_speed, v)))
        msg.angular.z = float(max(-self.max_turn, min(self.max_turn, w)))
        self.pub_cmd.publish(msg)

    def stop(self):
        if rclpy.ok():
            try:
                self.pub_cmd.publish(Twist())
            except Exception:
                pass

    def log_every(self, seconds, text):
        now = time.time()
        if now - self._last_log.get(text[:20], 0.0) >= seconds:
            self._last_log[text[:20]] = now
            self.get_logger().info(text)

    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error(f'control() raised: {e}')
            self.stop()

    def draw_lidar_overlay(self, img, threshold_dist):
        if self.scan is None or not self.scan.ranges:
            return float('inf'), None

        h, w, _ = img.shape
        center_x = w // 2
        center_y = int(h * 0.75)

        scan = self.scan
        min_dist = float('inf')
        closest_point = None

        for angle_deg in range(-15, 16, 2):
            angle_rad = math.radians(angle_deg)
            if scan.angle_min <= angle_rad <= scan.angle_max:
                idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
                if 0 <= idx < len(scan.ranges):
                    r = scan.ranges[idx]
                    if math.isfinite(r) and r > scan.range_min:
                        pt_x = int(center_x + (angle_deg / 15.0) * (w * 0.25))
                        pt_y = int(center_y - (r / 1.5) * (h * 0.4))
                        pt_y = max(20, min(h - 10, pt_y))

                        if r < threshold_dist:
                            cv2.circle(img, (pt_x, pt_y), 6, (0, 0, 255), -1)
                            if r < min_dist:
                                min_dist = r
                                closest_point = (pt_x, pt_y, angle_deg, r)
                        else:
                            cv2.circle(img, (pt_x, pt_y), 3, (0, 255, 0), -1)

                        if r < min_dist:
                            min_dist = r

        if closest_point:
            px, py, ang, dist = closest_point
            cv2.line(img, (center_x, h - 20), (px, py), (0, 0, 255), 2)
            cv2.putText(img, f"OBSTACLE: {dist:.2f}m at {ang}deg", (px - 60, py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return min_dist, closest_point

    def check_turn_path_clear(self, angular_z, base_half_width=15.0, max_bias_deg=28.0):
        if self.scan is None or not self.scan.ranges:
            return float('inf')

        scan = self.scan
        turn_ratio = max(-1.0, min(1.0, angular_z / 0.35))
        bias = max_bias_deg * turn_ratio
        half = base_half_width + abs(bias) * 0.5

        lo_deg = bias - half
        hi_deg = bias + half

        min_d = float('inf')

        for angle_deg in np.arange(lo_deg, hi_deg + 1.0, 2.0):
            angle_rad = math.radians(angle_deg)
            if scan.angle_min <= angle_rad <= scan.angle_max:
                idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
                if 0 <= idx < len(scan.ranges):
                    r = scan.ranges[idx]
                    if math.isfinite(r) and r > scan.range_min:
                        min_d = min(min_d, r)

        return min_d

    def get_obstacle_avoid_bias(self, safe_dist=1.0, max_bias=0.35):
        """Mở rộng dải quét hông từ 5 đến 60 độ mỗi bên để nhận diện cột/biển báo sát mép đường."""
        if self.scan is None or not self.scan.ranges:
            return 0.0

        scan = self.scan

        def min_in_sector(deg_min, deg_max):
            m = float('inf')
            for angle_deg in np.arange(deg_min, deg_max, 2.0):
                angle_rad = math.radians(angle_deg)
                if scan.angle_min <= angle_rad <= scan.angle_max:
                    idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
                    if 0 <= idx < len(scan.ranges):
                        r = scan.ranges[idx]
                        if math.isfinite(r) and r > scan.range_min:
                            m = min(m, r)
            return m

        left_dist = min_in_sector(5.0, 60.0)
        right_dist = min_in_sector(-60.0, -5.0)

        bias = 0.0
        if left_dist < safe_dist:
            push = (safe_dist - left_dist) / safe_dist
            bias -= push * max_bias

        if right_dist < safe_dist:
            push = (safe_dist - right_dist) / safe_dist
            bias += push * max_bias

        return bias

    def process_image_mask(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 2 / 3)
        roi = img[crop_h:h, :]

        roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        gray = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        roi_enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 110])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        side_crop = 30
        mask[:, :side_crop] = 0
        mask[:, w - side_crop:] = 0

        return mask, roi

    def get_lane_cx(self, mask, roi_w, ref_x):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cx = None
        best_score = float('inf')

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)

            if bw > roi_w * 0.6:
                continue

            longest_dim = max(bw, bh)
            if longest_dim >= 12 and len(cnt) >= 5:
                vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                angle_from_vertical = math.degrees(math.atan2(abs(vx), abs(vy) + 1e-6))
                
                if angle_from_vertical > 55.0:
                    continue
            else:
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 99.0
                if aspect_ratio > 3.0:
                    continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            score = abs(cx - ref_x)
            if score < best_score:
                best_score = score
                best_cx = cx

        return best_cx

    def largest_blob_ratio(self, mask):
        total = mask.shape[0] * mask.shape[1]
        if total == 0:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(cv2.contourArea(c) for c in contours)
        return float(largest) / float(total)

    def process_tunnel_center_line(self, img):
        h, w, _ = img.shape
        crop_h = int(h * 0.6)
        roi = img[crop_h:h, :]
        roi_h, roi_w, _ = roi.shape

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        _, mask = cv2.threshold(enhanced, 175, 255, cv2.THRESH_BINARY)

        margin = int(roi_w * 0.20)
        mask[:, :margin] = 0
        mask[:, roi_w - margin:] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cx = None
        min_dist = float('inf')
        ref_x = self.last_tunnel_cx if (self.last_tunnel_cx is not None) else (roi_w / 2.0)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 35:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh) if bh > 0 else 99.0
                
                if aspect_ratio > 2.2:
                    continue

                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    dist = abs(cx - ref_x)

                    if dist < min_dist:
                        min_dist = dist
                        best_cx = cx

        if best_cx is not None:
            self.last_tunnel_cx = best_cx

        return best_cx, mask, roi

    def control(self):
        if self.image is None:
            self.stop()
            return

        now = time.time()
        vis_img = self.image.copy()
        h, w, _ = self.image.shape

        mask_normal, roi_normal = self.process_image_mask(self.image)
        blob_ratio = self.largest_blob_ratio(mask_normal)

        # --------------------------------------------------------------------
        # 1. NHẬN BIẾT DỐC CẦU VÀ HẦM
        # Kết hợp Góc Pitch (IMU/Odom) + blob_ratio để nhận biết dốc cầu chuẩn xác 100%
        # --------------------------------------------------------------------
        top_brightness = np.mean(self.image[:int(h / 3), :])

        is_ramp_like = (abs(self.pitch) > 0.08) or (blob_ratio > 0.20 and top_brightness > 85.0)
        is_tunnel_like = (top_brightness < 65.0) and (blob_ratio > 0.18 or self.is_on_ramp)

        if not self.is_on_ramp and not self.is_in_tunnel and is_ramp_like:
            self.is_on_ramp = True
            self.ramp_start_time = now
            self.stuck_since = None
            self.get_logger().info(f'>>> KÍCH HOẠT DỐC CẦU (Pitch={self.pitch:.3f} rad, blob={blob_ratio:.2f}) <<<')

        if not self.is_in_tunnel and is_tunnel_like:
            self.is_in_tunnel = True
            self.tunnel_start_time = now
            self.stuck_since = None
            self.last_tunnel_cx = None
            self.get_logger().info(f'>>> KÍCH HOẠT VÀO HẦM (top_bright={top_brightness:.1f}) <<<')

        if self.is_on_ramp and (now - self.ramp_start_time > self.RAMP_DURATION):
            self.is_on_ramp = False
            self.get_logger().info('>>> THOÁT DỐC CẦU <<<')

        if self.is_in_tunnel and (now - self.tunnel_start_time > self.TUNNEL_DURATION):
            self.is_in_tunnel = False
            self.last_tunnel_cx = None
            self.last_outdoor_cx = None
            self.last_error = 0.0
            self.prev_angular_z = 0.0
            self.hard_turn_confirm = 0
            self.get_logger().info('>>> THOÁT HẦM: RESET ERROR & TRANSITION OUTDOOR <<<')

        # --------------------------------------------------------------------
        # 2. XỬ LÝ LIDAR CHÍNH DIỆN & WATCHDOG
        # --------------------------------------------------------------------
        disable_lidar = self.is_on_ramp

        if not disable_lidar:
            min_front_dist, closest_info = self.draw_lidar_overlay(vis_img, self.stop_distance)
            will_stop = min_front_dist < self.stop_distance

            if will_stop:
                if self.stuck_since is None:
                    self.stuck_since = now
                    self.stuck_ref_dist = min_front_dist
                stuck_elapsed = now - self.stuck_since
                dist_stable = abs(min_front_dist - self.stuck_ref_dist) < 0.05
            else:
                self.stuck_since = None
                stuck_elapsed = 0.0
                dist_stable = False

            if will_stop and stuck_elapsed > self.STUCK_TIMEOUT and dist_stable:
                self.get_logger().warn(f'>>> STUCK-ESCAPE: Kẹt {stuck_elapsed:.1f}s -> Chuyển TUNNEL MODE <<<')
                self.is_in_tunnel = True
                self.tunnel_start_time = now
                self.stuck_since = None
                will_stop = False

            if will_stop:
                self.stop()
                self.log_every(1.0, f'Obstacle Stop at {min_front_dist:.2f}m')
                cv2.putText(vis_img, f"OBSTACLE STOP ({min_front_dist:.2f}m)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
        else:
            self.stuck_since = None
            cv2.putText(vis_img, "LIDAR IGNORED (RAMP MODE)", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --------------------------------------------------------------------
        # 3. ĐIỀU KHIỂN CHUYỂN ĐỘNG & TÍNH NĂNG NÉ CỘT HÔNG
        # --------------------------------------------------------------------
        if self.is_in_tunnel:
            tunnel_cx, tunnel_mask, _ = self.process_tunnel_center_line(self.image)
            
            if tunnel_cx is not None:
                image_center = w / 2.0
                error = tunnel_cx - image_center

                Kp = 0.0050
                Kd = 0.008
                derivative = error - self.last_error
                self.last_error = error

                angular_z = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.45, min(0.45, angular_z))

                self.drive(self.max_speed * 0.8, angular_z)

                cv2.circle(vis_img, (tunnel_cx, int(h * 0.75)), 8, (255, 0, 255), -1)
                cv2.putText(vis_img, f"TUNNEL TRACKING (cx={tunnel_cx}, err={error:.1f}px)", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                self.drive(self.max_speed * 0.5, -0.1)
                cv2.putText(vis_img, "TUNNEL: SEARCHING CENTER LINE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        else:
            # BÁM LINE NORMAL / RAMP NGOÀI TRỜI
            ref_x = self.last_outdoor_cx if (self.last_outdoor_cx is not None) else (w / 2.0)
            cx = self.get_lane_cx(mask_normal, w, ref_x)

            # Lực né tránh cột bên hông
            avoid_bias = self.get_obstacle_avoid_bias(safe_dist=1.0, max_bias=0.35) if not disable_lidar else 0.0

            if cx is not None:
                self.last_outdoor_cx = cx
                image_center = w / 2.0
                raw_error = cx - image_center
                abs_error = abs(raw_error)

                if abs_error >= 60.0:
                    self.hard_turn_confirm = min(self.hard_turn_confirm + 1, 99)
                else:
                    self.hard_turn_confirm = 0

                is_confirmed_hard_turn = self.hard_turn_confirm >= 3

                if is_confirmed_hard_turn:
                    current_speed = self.max_speed * 0.55
                    Kp, Kd = 0.0055, 0.0080
                    max_turn_limit = 0.55
                    error = raw_error
                else:
                    current_speed = self.max_speed
                    Kp, Kd = 0.0035, 0.0060
                    max_turn_limit = 0.35

                    if abs_error < 15.0:
                        error = 0.0
                    else:
                        error = raw_error

                derivative = error - self.last_error
                self.last_error = error

                raw_angular = -float(error * Kp + derivative * Kd)

                # Cộng lực né cột hông vào góc lái
                angular_z = raw_angular + avoid_bias
                angular_z = max(-max_turn_limit, min(max_turn_limit, angular_z))

                turn_clear_dist = self.check_turn_path_clear(angular_z)
                if turn_clear_dist < 0.32 and not disable_lidar:
                    current_speed = min(current_speed, self.max_speed * 0.3)
                    angular_z *= 0.5
                    self.log_every(0.5, f'[TURN GUARD] Vật cản hướng rẽ {turn_clear_dist:.2f}m -> Giảm tốc/lái')
                    cv2.putText(vis_img, f"TURN GUARD! dist={turn_clear_dist:.2f}m", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

                self.drive(current_speed, angular_z)

                cv2.circle(vis_img, (cx, int(h * 5 / 6)), 8, (0, 255, 0), -1)
                mode_str = "HARD_TURN" if is_confirmed_hard_turn else ("RAMP" if self.is_on_ramp else "NORMAL")
                cv2.putText(vis_img, f"TRACKING [{mode_str}] (Err={error:.1f}px, avoid={avoid_bias:.2f})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                fallback_w = avoid_bias if abs(avoid_bias) > 0.05 else (-0.10 if self.last_error > 0 else 0.10)
                self.drive(self.max_speed * 0.6, fallback_w)
                cv2.putText(vis_img, f"LINE LOST - FALLBACK (w={fallback_w:.2f})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("Robot Debug View", vis_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Starter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
            time.sleep(0.05)
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()