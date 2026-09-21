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
        # BIẾN ĐIỀU CHỈNH LỆCH LÀN (OFFSET)
        # --------------------------------------------------------------------
        self.line_offset = 50

        # --------------------------------------------------------------------
        # BIẾN QUẢN LÝ TRẠNG THÁI & LEO DỐC / CẦU (RAMP OVERRIDE)
        # --------------------------------------------------------------------
        self.no_line_frame_count = 0  # Bộ đếm frame mất line
        self.REQUIRED_NO_LINE_FRAMES = 12  # Số frame nghi vấn bắt đầu lên cầu

        # Thời gian tối đa (giây) cho phép xe chạy thẳng vượt cầu khi không có line
        self.RAMP_DRIVE_DURATION = 18.0
        self.ramp_start_time = None
        self.is_on_ramp = False
        self.is_line_ended = False
        self.target_yaw = None  # Góc hướng ban đầu cần giữ thẳng khi leo cầu

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

    def get_line_center_at_row(self, thresh_img, row_y):
        """
        Tìm tâm làn đường bằng cách tách riêng biên trái và biên phải,
        bỏ qua các vạch nằm quá sát biên ngoài cùng của ảnh (tránh ăn vào cột đèn/lề).
        """
        h, w = thresh_img.shape
        mid_x = w // 2

        # Cắt bỏ 10% viền mép ngoài cùng trái/phải để không ăn nhầm lề đường/cột đèn
        margin = int(w * 0.10)
        row = thresh_img[row_y, margin : w - margin]

        # Tìm các điểm trắng trên hàng quét đã cắt margin
        white_pts = np.where(row > 0)[0]
        if len(white_pts) < 10:
            return None

        # Trả về tọa độ pixel thật trên ảnh
        actual_pts = white_pts + margin

        # Phân loại điểm trắng thuộc nửa trái hay nửa phải ảnh
        left_pts = [p for p in actual_pts if p < mid_x]
        right_pts = [p for p in actual_pts if p >= mid_x]

        # Lọc nhiễu dải trắng quá rộng (như vạch ngựa vằn / chân dốc)
        if len(actual_pts) > int(w * 0.40):
            return None

        # TH 1: Bắt được cả vạch trái và vạch phải -> Lấy trung điểm của 2 vạch (Chuẩn nhất)
        if len(left_pts) > 0 and len(right_pts) > 0:
            left_center = np.mean(left_pts)
            right_center = np.mean(right_pts)
            return int((left_center + right_center) / 2.0)

        # TH 2: Chỉ thấy vạch bên phải -> Giữ khoảng cách an toàn (Offset sang trái 120px)
        elif len(right_pts) > 0:
            right_center = np.mean(right_pts)
            # Không được lao thẳng vào vạch phải, phải duy trì khoảng cách an toàn
            safe_center = right_center - 120  
            return int(safe_center)

        # TH 3: Chỉ thấy vạch bên trái -> Offset sang phải 120px
        elif len(left_pts) > 0:
            left_center = np.mean(left_pts)
            safe_center = left_center + 120
            return int(safe_center)

        return None

    def control(self):
        if self.image is None:
            self.stop()
            return

        # --------------------------------------------------------------------
        # KHỞI TẠO BIẾN AN TOÀN TRÁNH LỖI UNASSIGNED VARIABLE
        # --------------------------------------------------------------------
        angular_z = 0.0
        error = 0.0
        now = time.time()

        # Tạo ảnh hiển thị Debug bằng cv2
        vis_img = self.image.copy()
        h, w, _ = self.image.shape

        # 1. An toàn LIDAR
        front = self.range_at(0, width_deg=30)
        if front < self.stop_distance:
            self.stop()
            self.log_every(2.0, f'Obstacle at {front:.2f}m')
            
            # --- CV2 IMSHOW: BƯỚC 1 (DỪNG DO VẬT CẢN) ---
            cv2.putText(vis_img, f"STEP 1: OBSTACLE STOP ({front:.2f}m)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Robot Debug View", vis_img)
            cv2.waitKey(1)
            return

        # 2. Xử lý ảnh nhị phân
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 3. Lấy mẫu vạch ở 3 hàng quét
        scan_rows = [int(h * 0.90), int(h * 0.80), int(h * 0.70)]
        valid_centers = []
        for r in scan_rows:
            cx = self.get_line_center_at_row(thresh, r)
            if cx is not None:
                valid_centers.append(cx)
                # Vẽ điểm quét lên ảnh
                cv2.circle(vis_img, (cx, r), 5, (0, 255, 0), -1)

        # 4. Kiểm tra vạch xa phía trên
        upper_scanlines_y = [int(h * 0.55), int(h * 0.45), int(h * 0.35)]
        upper_centers = [self.get_line_center_at_row(thresh, y) for y in upper_scanlines_y]
        has_line_ahead = any(c is not None for c in upper_centers)

        # Phát hiện vạch ngang/vạch mép bị méo
        is_horizontal_line = False
        if len(valid_centers) >= 2:
            dx = abs(valid_centers[0] - valid_centers[-1])
            if dx > int(w * 0.35):
                is_horizontal_line = True

        has_valid_current_line = (len(valid_centers) > 0) and not is_horizontal_line

        # --------------------------------------------------------------------
        # KÍCH HOẠT LEO DỐC CẦU (RAMP MODE)
        # --------------------------------------------------------------------
        if not self.is_on_ramp:
            if (not has_line_ahead or is_horizontal_line) and (0.35 < front < 0.75):
                self.is_on_ramp = True
                self.ramp_start_time = now
                self.get_logger().info('>>> CHÂN CẦU: KÍCH HOẠT LEO DỐC CẦU (ÉP ĐI THẲNG)! <<<')

        # --------------------------------------------------------------------
        # BỘ ĐIỀU KHIỂN TÍN HIỆU
        # --------------------------------------------------------------------
        
        # TRƯỜNG HỢP 1: LEO CẦU (Khóa cứng bẻ lái = 0)
        if self.is_on_ramp:
            elapsed = now - self.ramp_start_time
            if elapsed < 1.8:
                self.drive(self.max_speed, 0.0)
                self.get_logger().info(f"[RAMP MODE ACTIVE] Time={elapsed:.1f}s/1.8s | Cmd_w=0.000")
                
                # --- CV2 IMSHOW: BƯỚC LEO CẦU (RAMP MODE) ---
                cv2.putText(vis_img, f"STEP: RAMP MODE ({elapsed:.1f}s/1.8s)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Robot Debug View", vis_img)
                cv2.waitKey(1)
                return
            else:
                self.is_on_ramp = False
                self.get_logger().info('>>> ĐÃ VƯỢT DỐC CẦU -> CHUYỂN SANG BÁM LÀN CAMERA <<<')

        # TRƯỜNG HỢP 2: BÁM LÀN BẰNG CAMERA VỚI LỰC BẺ LÁI TỐI ƯU
        # Trong phần TRƯỜNG HỢP 2: BÁM LÀN BẰNG CAMERA
        if has_valid_current_line:
            target_cx = int(np.mean(valid_centers))
            image_center = w / 2.0
            error = target_cx - image_center

            if abs(error) > 250.0:
                raw_angular = 0.0
            elif abs(error) <= 10.0:
                raw_angular = 0.0
            else:
                Kp = 0.003
                raw_angular = -Kp * float(error)
                
                # KHỐNG CHẾ LỰC BẺ LÁI:
                # Bẻ trái (raw > 0) tối đa +0.3 rad/s
                # Bẻ phải (raw < 0) SIẾT CHẶT hơn, tối đa -0.15 rad/s để KHÔNG ĐÂM VÀO CỘT ĐÈN BÊN PHẢI
                if raw_angular < 0:
                    raw_angular = max(-0.15, raw_angular)  # Giới hạn góc bẻ phải
                else:
                    raw_angular = min(0.30, raw_angular)

            # Lọc mượt Low-pass
            alpha = 0.3
            angular_z = alpha * raw_angular + (1.0 - alpha) * getattr(self, 'prev_angular_z', 0.0)
            self.prev_angular_z = angular_z

            self.drive(self.max_speed, angular_z)

            # --- CV2 IMSHOW: BƯỚC BÁM LÀN CAMERA ---
            cv2.line(vis_img, (int(image_center), 0), (int(image_center), h), (255, 0, 0), 1)
            cv2.line(vis_img, (target_cx, 0), (target_cx, h), (0, 255, 0), 2)
            #cv2.putText(vis_img, status_txt, (20, 40),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            angular_z = 0.0
            self.drive(self.max_speed * 0.8, angular_z)

            # --- CV2 IMSHOW: BƯỚC MẤT LÀN (NO LINE) ---
            cv2.putText(vis_img, "STEP: NO VALID LINE (SLOW DOWN)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # In log chẩn đoán
        self.log_every(0.5, f"[LANE TRACKING] Ramp={self.is_on_ramp} | Error={error:.1f}px | Angular_w={angular_z:.3f}")

        # --- HIỂN THỊ CỬA SỔ OPEN CV REALTIME ---
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
        cv2.destroyAllWindows()  # Đóng cửa sổ cv2 khi dừng node
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()