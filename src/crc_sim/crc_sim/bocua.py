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
        self.pitch = 0.0
        self._last_log = {}

        self.last_error = 0.0
        self.prev_angular_z = 0.0
        self.hard_turn_confirm = 0
        self.last_outdoor_cx = None

        # --- TRẠNG THÁI NÉ VẬT CẢN (OVERTAKE STATE MACHINE) ---
        self.overtake_state = "IDLE"  # "IDLE", "LANE_CHANGE", "PASSING", "RETURN_LANE"
        self.overtake_start_time = 0.0

        # Thời gian cấu hình cho từng giai đoạn (giây)
        self.TIME_LANE_CHANGE = 0.8   # Thời gian lách sang làn phải
        self.TIME_PASSING = 1.8       # Thời gian giữ lái đi thẳng vượt qua xe
        self.TIME_RETURN_LANE = 0.8   # Thời gian lách ngược trở lại làn trái

        # --- DỐC CẦU & HẦM ---
        self.is_on_ramp = False
        self.ramp_start_time = None
        self.RAMP_DURATION = 3.5

        self.is_in_tunnel = False
        self.tunnel_start_time = None
        self.TUNNEL_DURATION = 7.0
        self.last_tunnel_cx = None

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

        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

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

    def detect_blue_car_camera(self, img):
        h, w, _ = img.shape
        roi = img[int(h * 0.2):int(h * 0.75), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(mask)

        # Trả về True nếu phát hiện xe xanh ngay trước mặt
        if blue_pixels > (roi.shape[0] * roi.shape[1] * 0.07):
            return True
        return False

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

    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error(f'control() raised: {e}')
            self.stop()

    def control(self):
        if self.image is None:
            self.stop()
            return

        now = time.time()
        vis_img = self.image.copy()
        h, w, _ = self.image.shape

        has_blue_car = self.detect_blue_car_camera(self.image)

        # --------------------------------------------------------------------
        # MÁY TRẠNG THÁI VƯỢT XE (OVERTAKE STATE MACHINE)
        # --------------------------------------------------------------------
        # Khi phát hiện xe xanh và chưa ở trong chu trình vượt -> Kích hoạt Vượt
        if has_blue_car and self.overtake_state == "IDLE":
            self.overtake_state = "LANE_CHANGE"
            self.overtake_start_time = now
            self.get_logger().info(">>> KÍCH HOẠT CHU TRÌNH VƯỢT XE XANH <<<")

        if self.overtake_state != "IDLE":
            elapsed = now - self.overtake_start_time

            # 1. Giai đoạn Lách Làn Phải (Đánh lái cố định)
            if self.overtake_state == "LANE_CHANGE":
                if elapsed < self.TIME_LANE_CHANGE:
                    self.drive(self.max_speed * 0.8, 0.35) # Lách sang trai
                    cv2.putText(vis_img, "OVERTAKE: LATCHING RIGHT", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    self.overtake_state = "PASSING"
                    self.overtake_start_time = now

            # 2. Giai đoạn Giữ Thẳng Lái để Vượt Mặt
            elif self.overtake_state == "PASSING":
                if elapsed < self.TIME_PASSING:
                    self.drive(self.max_speed, 0.0) # GIỮ NGUYÊN GÓC LÁI THẲNG
                    cv2.putText(vis_img, "OVERTAKE: KEEP STRAIGHT & PASSING", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    self.overtake_state = "RETURN_LANE"
                    self.overtake_start_time = now

            # 3. Giai đoạn Nhập Làn Về Lại Line Trái
            elif self.overtake_state == "RETURN_LANE":
                if elapsed < self.TIME_RETURN_LANE:
                    self.drive(self.max_speed * 0.8, -0.30) # Đánh lái nhẹ sang phai về lại làn
                    cv2.putText(vis_img, "OVERTAKE: RETURNING TO LANE", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                else:
                    # Hoàn tất vượt xe, reset trạng thái
                    self.overtake_state = "IDLE"
                    self.last_outdoor_cx = None
                    self.last_error = 0.0
                    self.get_logger().info(">>> HOÀN THÀNH VƯỢT XE - TRỞ VỀ DÒ LINE <<<")

            cv2.imshow("Robot Debug View", vis_img)
            cv2.waitKey(1)
            return  # Tạm bỏ qua dò line khi đang chạy chu trình vượt xe

        # --------------------------------------------------------------------
        # BÌNH THƯỜNG: CHẠY DÒ LINE PID
        # --------------------------------------------------------------------
        mask_normal, roi_normal = self.process_image_mask(self.image)
        ref_x = self.last_outdoor_cx if (self.last_outdoor_cx is not None) else (w / 2.0)
        cx = self.get_lane_cx(mask_normal, w, ref_x)

        if cx is not None:
            self.last_outdoor_cx = cx
            image_center = w / 2.0
            error = cx - image_center

            Kp, Kd = 0.0035, 0.0060
            derivative = error - self.last_error
            self.last_error = error

            angular_z = -float(error * Kp + derivative * Kd)
            angular_z = max(-0.50, min(0.50, angular_z))

            self.drive(self.max_speed, angular_z)

            cv2.circle(vis_img, (cx, int(h * 5 / 6)), 8, (0, 255, 0), -1)
            cv2.putText(vis_img, f"TRACKING LINE (Err={error:.1f}px)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            self.drive(self.max_speed * 0.5, 0.0)
            cv2.putText(vis_img, "LINE LOST - SLOW DOWN STRAIGHT", (20, 40),
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