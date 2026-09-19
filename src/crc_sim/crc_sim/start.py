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

        # --------------------------------------------------------------------
        # BIẾN KHỞI TẠO CHO BỘ LỌC CHỐNG VẠCH NGANG & HẾT LINE
        # --------------------------------------------------------------------
        self.no_line_frame_count = 0  # Bộ đếm số frame liên tiếp không thấy line
        self.REQUIRED_NO_LINE_FRAMES = 5  # Cần ít nhất 5 frame liên tục để xác nhận hết line
        self.is_line_ended = False    # Cờ trạng thái đã kích hoạt HẾT LINE

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

    def range_at(self, angle_deg, width_deg=10.0):
        if self.scan is None or not self.scan.ranges:
            return float('inf')
        n = len(self.scan.ranges)
        best = float('inf')
        half = int(round(width_deg / 2.0))
        centre = int(round(angle_deg)) % 360
        for d in range(-half, half + 1):
            r = self.scan.ranges[int((centre + d) % 360 * n / 360)]
            if math.isfinite(r) and r > self.scan.range_min:
                best = min(best, r)
        return best

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

    # ------------------------------------------------------------------------
    # THUẬT TOÁN BÁM LÀN - CHỐNG NHẦM VẠCH NGANG & LỌC HẾT LINE
    # ------------------------------------------------------------------------

    def get_line_center_at_row(self, thresh_img, row_y):
        """Lấy trung điểm x của vạch màu trắng trên một hàng quét chỉ định."""
        row_pixels = np.where(thresh_img[row_y, :] > 0)[0]
        if len(row_pixels) > 5:  # Lọc nhiễu pixel nhỏ lẻ
            return int(np.mean(row_pixels))
        return None

    def control(self):
        if self.image is None:
            self.stop()
            return

        # An toàn LIDAR
        front = self.range_at(0, width_deg=30)
        if front < self.stop_distance:
            self.stop()
            self.log_every(2.0, f'Obstacle at {front:.2f}m')
            return

        h, w, _ = self.image.shape

        # 1. Tiền xử lý ảnh sang nhị phân
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 2. Định nghĩa ROI chính (1/3 dưới ảnh)
        roi_start_y = int(h * 0.7)
        roi_end_y = int(h * 0.95)

        # Tìm trung điểm line ở 2 hàng trong ROI chính để xác định tâm và góc nghiêng
        cx_bottom = self.get_line_center_at_row(thresh, roi_end_y)
        cx_mid = self.get_line_center_at_row(thresh, roi_start_y)

        # 3. Quét thêm 2-3 hàng phía trên ROI chính (Scanlines chống vạch ngang)
        upper_scanlines_y = [int(h * 0.55), int(h * 0.45), int(h * 0.35)]
        upper_centers = [self.get_line_center_at_row(thresh, y) for y in upper_scanlines_y]
        has_line_ahead = any(c is not None for c in upper_centers)

        # 4. Kiểm tra hướng góc nghiêng của line
        is_horizontal_line = False
        if cx_bottom is not None and cx_mid is not None:
            dx = cx_mid - cx_bottom
            dy = roi_start_y - roi_end_y  # dy luôn âm
            angle_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
            
            # Nếu góc nghiêng > 55° so với phương dọc, nghi vấn là vạch ngang người đi bộ
            if angle_deg > 55.0:
                is_horizontal_line = True

        # Đánh giá sự xuất hiện hợp lệ của Line
        has_valid_current_line = (cx_bottom is not None or cx_mid is not None) and not is_horizontal_line

        # 5. Bộ lọc thời gian (Temporal Filter) cho sự kiện HẾT LINE
        if not has_valid_current_line and not has_line_ahead:
            self.no_line_frame_count += 1
        else:
            # Nếu thấy line hợp lệ hoặc vẫn còn line phía trước -> Reset bộ đếm
            self.no_line_frame_count = 0

        # Kiểm tra điều kiện kích hoạt HẾT LINE
        if self.no_line_frame_count >= self.REQUIRED_NO_LINE_FRAMES:
            self.is_line_ended = True

        # --------------------------------------------------------------------
        # ĐIỀU KHIỂN ROBOT
        # --------------------------------------------------------------------
        if self.is_line_ended:
            # Xử lý khi xác nhận ĐÃ HẾT LINE THẬT SỰ
            self.log_every(1.0, 'XÁC NHẬN: Hết line thật sự -> Dừng xe hoặc rẽ tìm line mới')
            self.stop()
            return

        if is_horizontal_line or (not has_valid_current_line and has_line_ahead):
            # Đi qua vạch người đi bộ: Giữ nguyên hướng lái, đi thẳng tiếp
            self.log_every(1.0, 'Phát hiện vạch ngang/vạch sang đường -> Giữ thẳng tay lái')
            self.drive(self.max_speed, 0.0)
            return

        # Bám làn đường bình thường
        target_cx = cx_bottom if cx_bottom is not None else cx_mid
        if target_cx is not None:
            image_center = w / 2.0
            error = target_cx - image_center
            Kp = 0.005
            angular_z = -Kp * float(error)
            self.drive(self.max_speed, angular_z)
        else:
            # Trường hợp tạm thời mất dấu (đang trong bộ đếm N frame)
            self.drive(self.max_speed * 0.7, 0.0)
def catch_sigterm():
    """Chuyển signal SIGTERM thành cờ dừng để tắt rclpy an toàn."""
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    """Vòng lặp spin nhận dữ liệu ROS 2 an toàn."""
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = Starter()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()