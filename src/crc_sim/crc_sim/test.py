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
        self._last_log = {}

        self.last_error = 0.0
        self.prev_angular_z = 0.0
        
        # --- CẤU HÌNH DỐC CẦU (RAMP) ---
        self.is_on_ramp = False
        self.ramp_start_time = None
        self.RAMP_DURATION = 3.5      # Thời gian ngắt LiDAR để vượt dốc (giây)
        
        # --- CẤU HÌNH HẦM (TUNNEL) & CONTINUITY TRACKING ---
        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 7.0    # Thời gian bám vạch hầm (giây)
        self.last_tunnel_cx = None    # Vị trí vết vạch giữa ở frame trước

        # --- WATCHDOG THOÁT BẾ TẮC ---
        self.stuck_since = None
        self.stuck_ref_dist = 0.0
        self.STUCK_TIMEOUT = 1.0      # Giây đứng yên trước khi tự giải phóng

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
            self.get_logger().warn(f'image conversion failed: {e}')

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(max(-self.max_speed, min(self.max_speed, v)))
        msg.angular.z = float(max(-self.max_turn, min(self.max_turn, w)))
        self.pub_cmd.publish(msg)

    def stop(self):
        self.pub_cmd.publish(Twist())

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
        """Hiển thị các điểm quét LiDAR chính diện (-15 deg đến +15 deg)."""
        if self.scan is None or not self.scan.ranges:
            return float('inf'), None

        h, w, _ = img.shape
        center_x = w // 2
        center_y = int(h * 0.75)

        n = len(self.scan.ranges)
        min_dist = float('inf')
        closest_point = None

        for angle in range(-15, 16, 2):
            idx = int((angle % 360) * n / 360)
            r = self.scan.ranges[idx]

            if math.isfinite(r) and r > self.scan.range_min:
                pt_x = int(center_x + (angle / 15.0) * (w * 0.25))
                pt_y = int(center_y - (r / 1.5) * (h * 0.4))
                pt_y = max(20, min(h - 10, pt_y))

                if r < threshold_dist:
                    cv2.circle(img, (pt_x, pt_y), 6, (0, 0, 255), -1)
                    cv2.circle(img, (pt_x, pt_y), 10, (0, 0, 255), 2)
                    if r < min_dist:
                        min_dist = r
                        closest_point = (pt_x, pt_y, angle, r)
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

    def process_image_mask(self, img):
        """Tạo mask lọc vạch kẻ đường ngoài trời."""
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

        side_crop = 35
        mask[:, :side_crop] = 0
        mask[:, w - side_crop:] = 0

        return mask, roi

    def largest_blob_ratio(self, mask):
        """Tính tỉ lệ diện tích khối liên thông LỚN NHẤT / tổng diện tích ROI."""
        total = mask.shape[0] * mask.shape[1]
        if total == 0:
            return 0.0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(cv2.contourArea(c) for c in contours)
        return float(largest) / float(total)

    def process_tunnel_center_line(self, img):
        """DÒ VẠCH GIỮA HẦM VỚI CONTINUITY TRACKING & LỌC TƯỜNG/GỜ LỀ."""
        h, w, _ = img.shape
        crop_h = int(h * 2 / 3)
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
        # --------------------------------------------------------------------
        top_brightness = np.mean(self.image[:int(h / 3), :])

        is_ramp_like = (blob_ratio > 0.22)
        is_tunnel_like = (top_brightness < 65.0) and (blob_ratio > 0.18 or self.is_on_ramp)

        # Kích hoạt Dốc
        if not self.is_on_ramp and not self.is_in_tunnel and is_ramp_like:
            self.is_on_ramp = True
            self.ramp_start_time = now
            self.stuck_since = None
            self.get_logger().info(f'>>> KÍCH HOẠT DỐC CẦU (blob_ratio={blob_ratio:.2f}) <<<')

        # Kích hoạt Hầm
        if not self.is_in_tunnel and is_tunnel_like:
            self.is_in_tunnel = True
            self.tunnel_start_time = now
            self.stuck_since = None
            self.last_tunnel_cx = None
            self.get_logger().info(f'>>> KÍCH HOẠT VÀO HẦM (top_bright={top_brightness:.1f}) <<<')

        # Hết thời gian Dốc
        if self.is_on_ramp and (now - self.ramp_start_time > self.RAMP_DURATION):
            self.is_on_ramp = False
            self.get_logger().info('>>> THOÁT DỐC CẦU <<<')

        # Hết thời gian Hầm
        if self.is_in_tunnel and (now - self.tunnel_start_time > self.TUNNEL_DURATION):
            self.is_in_tunnel = False
            self.last_tunnel_cx = None
            self.get_logger().info('>>> THOÁT HẦM <<<')

        # --------------------------------------------------------------------
        # 2. XỬ LÝ LIDAR & WATCHDOG
        # --------------------------------------------------------------------
        disable_lidar = self.is_on_ramp or self.is_in_tunnel

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
                disable_lidar = True

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
            cv2.putText(vis_img, "LIDAR IGNORED (RAMP/TUNNEL MODE)", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --------------------------------------------------------------------
        # 3. ĐIỀU KHIỂN CHUYỂN ĐỘNG & BÁM VẠCH KHI CUA GẮT
        # --------------------------------------------------------------------
        if self.is_in_tunnel:
            tunnel_cx, tunnel_mask, _ = self.process_tunnel_center_line(self.image)
            
            if tunnel_cx is not None:
                image_center = w / 2.0
                error = tunnel_cx - image_center

                Kp = 0.0045
                Kd = 0.008
                derivative = error - self.last_error
                self.last_error = error

                angular_z = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.4, min(0.4, angular_z))

                self.drive(self.max_speed, angular_z)

                cv2.circle(vis_img, (tunnel_cx, int(h * 5 / 6)), 8, (255, 0, 255), -1)
                cv2.putText(vis_img, f"TUNNEL TRACKING (cx={tunnel_cx}, err={error:.1f}px)", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                self.drive(self.max_speed * 0.6, -0.1)
                cv2.putText(vis_img, "TUNNEL: SEARCHING CENTER LINE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        else:
            # BÁM LINE NORMAL / RAMP NGOÀI TRỜI
            M = cv2.moments(mask_normal)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                image_center = w / 2.0
                error = cx - image_center

                # Thuật toán PID tự điều chỉnh theo độ gắt của cua
                abs_error = abs(error)

                if abs_error > 40.0:
                    # KHI VÀO CUA GẮT (|Err| > 40px, ví dụ Err = -108px):
                    # 1. Giảm tốc độ tiến để tránh lao thẳng chạm va cột
                    # 2. Tăng mạnh Kp và mở rộng max_turn góc lái lên 0.7 rad/s
                    current_speed = self.max_speed * 0.45
                    Kp = 0.0065
                    Kd = 0.009
                    max_turn_limit = 0.75
                else:
                    # KHI ĐI ĐƯỜNG THẲNG HOẶC CUA NHẸ:
                    current_speed = self.max_speed
                    Kp = 0.0035
                    Kd = 0.006
                    max_turn_limit = 0.35

                derivative = error - self.last_error
                self.last_error = error

                raw_angular = -float(error * Kp + derivative * Kd)
                angular_z = max(-max_turn_limit, min(max_turn_limit, raw_angular))

                self.drive(current_speed, angular_z)

                cv2.circle(vis_img, (cx, int(h * 5 / 6)), 8, (0, 255, 0), -1)
                mode_str = "RAMP" if self.is_on_ramp else "NORMAL"
                cv2.putText(vis_img, f"TRACKING [{mode_str}] (Err={error:.1f}px, v={current_speed:.2f})", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                fallback_w = -0.35 if self.last_error > 0 else 0.35
                self.drive(self.max_speed * 0.5, fallback_w)

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
        node.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()