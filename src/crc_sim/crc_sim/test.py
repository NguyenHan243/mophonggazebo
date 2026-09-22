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
        self.RAMP_DURATION = 2.0      # Thời gian leo dốc (giây)
        
        # --- CẤU HÌNH HẦM (TUNNEL) ---
        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 5.5    # Thời gian CHỈ ĐỊNH bám vạch giữa trong hầm (giây)

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
        """Xử lý hình ảnh bám làn ngoài trời thông thường."""
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

    def process_tunnel_center_line(self, img):
        """HÀM ỔN ĐỊNH CŨ: Tìm và bám vạch giữa trong hầm."""
        h, w, _ = img.shape
        crop_h = int(h * 2 / 3)
        roi = img[crop_h:h, :]
        roi_h, roi_w, _ = roi.shape

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        _, mask = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

        # Cắt bớt 25% biên hai bên để bỏ vạch lề
        margin = int(roi_w * 0.25)
        mask[:, :margin] = 0
        mask[:, roi_w - margin:] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cx = None
        min_dist_to_center = float('inf')
        image_center_x = roi_w / 2.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 40:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    dist = abs(cx - image_center_x)
                    if dist < min_dist_to_center:
                        min_dist_to_center = dist
                        best_cx = cx

        return best_cx, mask, roi

    def control(self):
        if self.image is None:
            self.stop()
            return

        now = time.time()
        vis_img = self.image.copy()
        h, w, _ = self.image.shape

        mask_normal, roi_normal = self.process_image_mask(self.image)
        white_pixel_count = cv2.countNonZero(mask_normal)

        # --------------------------------------------------------------------
        # 1. KÍCH HOẠT NHẬN BIẾN DỐC / HẦM
        # --------------------------------------------------------------------
        top_brightness = np.mean(self.image[:int(h/3), :])
        
        is_ramp_like = (white_pixel_count < 220)
        is_tunnel_like = (top_brightness < 55.0) and (white_pixel_count > 180)

        # Kích hoạt Dốc (Ramp)
        if not self.is_on_ramp and not self.is_in_tunnel and is_ramp_like:
            self.is_on_ramp = True
            self.ramp_start_time = now
            self.get_logger().info('>>> KÍCH HOẠT DỐC CẦU (LIDAR DISABLED) <<<')

        # Kích hoạt Hầm (Tunnel)
        if not self.is_in_tunnel and is_tunnel_like:
            self.is_in_tunnel = True
            self.tunnel_start_time = now
            self.get_logger().info('>>> KÍCH HOẠT VÀO HẦM: BÁM VẠCH GIỮA (LIDAR DISABLED) <<<')

        # Kiểm tra Hết thời gian Dốc
        if self.is_on_ramp and (now - self.ramp_start_time > self.RAMP_DURATION):
            self.is_on_ramp = False
            self.get_logger().info('>>> THOÁT DỐC CẦU (LIDAR RE-ENABLED) <<<')

        # Kiểm tra Hết thời gian Hầm
        if self.is_in_tunnel and (now - self.tunnel_start_time > self.TUNNEL_DURATION):
            self.is_in_tunnel = False
            self.get_logger().info('>>> THOÁT HẦM: QUAY LẠI BÁM LÀN NGOÀI TRỜI (LIDAR RE-ENABLED) <<<')

        # --------------------------------------------------------------------
        # 2. XỬ LÝ LIDAR (NGẮT TẠM THỜI KHI LÊN DỐC HOẶC TRONG HẦM)
        # --------------------------------------------------------------------
        disable_lidar = self.is_on_ramp or self.is_in_tunnel

        if not disable_lidar:
            min_front_dist, closest_info = self.draw_lidar_overlay(vis_img, self.stop_distance)

            if min_front_dist < self.stop_distance:
                self.stop()
                self.log_every(1.0, f'Obstacle Stop at {min_front_dist:.2f}m')
                cv2.putText(vis_img, f"OBSTACLE STOP ({min_front_dist:.2f}m)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
        else:
            # Hiển thị thông báo LiDAR bị ngắt để chạy mượt qua gờ/sàn xám
            cv2.putText(vis_img, "LIDAR IGNORED (RAMP/TUNNEL MODE)", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --------------------------------------------------------------------
        # 3. ĐIỀU KHIỂN CHUYỂN ĐỘNG
        # --------------------------------------------------------------------
        # TH1: Đang leo dốc -> Đi thẳng
        if self.is_on_ramp:
            self.drive(self.max_speed, 0.0)
            cv2.putText(vis_img, f"MODE: RAMP ({now - self.ramp_start_time:.1f}s)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # TH2: Đang trong hầm -> Bám vạch giữa (Code cũ ổn định)
        elif self.is_in_tunnel:
            tunnel_cx, tunnel_mask, _ = self.process_tunnel_center_line(self.image)
            
            if tunnel_cx is not None:
                image_center = w / 2.0
                error = tunnel_cx - image_center

                Kp = 0.004
                Kd = 0.007
                derivative = error - self.last_error
                self.last_error = error

                angular_z = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.3, min(0.3, angular_z))

                self.drive(self.max_speed, angular_z)

                cv2.circle(vis_img, (tunnel_cx, int(h * 5 / 6)), 8, (255, 0, 255), -1)
                cv2.putText(vis_img, f"TUNNEL CENTER LINE TRACKING ({now - self.tunnel_start_time:.1f}s)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                self.drive(self.max_speed * 0.7, 0.0)
                cv2.putText(vis_img, "TUNNEL: SEARCHING CENTER LINE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # TH3: Ở ngoài trời -> Bám làn đường thông thường
        else:
            M = cv2.moments(mask_normal)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                image_center = w / 2.0
                error = cx - image_center

                Kp = 0.0035
                Kd = 0.006

                derivative = error - self.last_error
                self.last_error = error

                raw_angular = -float(error * Kp + derivative * Kd)
                angular_z = max(-0.35, min(0.35, raw_angular))

                self.drive(self.max_speed, angular_z)

                cv2.circle(vis_img, (cx, int(h * 5 / 6)), 8, (0, 255, 0), -1)
                cv2.putText(vis_img, f"NORMAL TRACKING (Err={error:.1f}px)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                fallback_w = -0.25 if self.last_error > 0 else 0.25
                self.drive(self.max_speed * 0.7, fallback_w)

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