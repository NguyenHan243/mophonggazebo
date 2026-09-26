#!/usr/bin/env python3
"""
traffic_perception_test.py
---------------------------------------------------------------
Node Nhận diện Đèn Giao thông, Biển báo & Vạch dừng (traffic_node)
  - Lắng nghe hình ảnh từ /camera/image_raw
  - Xử lý nhận diện:
      1. Đèn Giao thông (RED, YELLOW, GREEN) kết hợp Tracker & Lọc nhiễu kích thước
      2. Biển báo (STOP, NO_HIGHWAY)
      3. Vạch dừng ngang (Stop Line)
  - Publish tín hiệu ra Topic '/traffic_signal' (std_msgs/String)
---------------------------------------------------------------
"""

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Khởi tạo cv_bridge an toàn
try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False


# =====================================================================
# HÀM BỔ TRỢ & LỚP TRACKER LỌC NHIỄU ĐÈN GIAO THÔNG
# =====================================================================
def sample_lamp_color(img, circle):
    cx, cy, r = circle
    h, w, _ = img.shape

    y1, y2 = max(0, cy - r), min(h, cy + r)
    x1, x2 = max(0, cx - r), min(w, cx + r)
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return 'UNKNOWN'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))

    r_count = cv2.countNonZero(red_mask)
    y_count = cv2.countNonZero(yellow_mask)
    g_count = cv2.countNonZero(green_mask)

    counts = {'RED': r_count, 'YELLOW': y_count, 'GREEN': g_count}
    max_color = max(counts, key=counts.get)

    if counts[max_color] > 10:
        return max_color
    return 'UNKNOWN'


def position_to_color(box, circle):
    x, y, bw, bh = box
    cx, cy, r = circle

    rel_y = (cy - y) / float(bh)
    if rel_y < 0.35:
        return 'RED'
    elif rel_y < 0.68:
        return 'YELLOW'
    else:
        return 'GREEN'


class LightTracker:
    def __init__(self, tracker_id, box):
        self.id = tracker_id
        self.box = box
        self.last_seen = time.time()
        self.history = []

    def update(self, color, now):
        self.last_seen = now
        if color != 'UNKNOWN':
            self.history.append(color)
            if len(self.history) > 5:
                self.history.pop(0)

        if not self.history:
            return 'UNKNOWN', False

        most_common = max(set(self.history), key=self.history.count)
        confirmed = self.history.count(most_common) >= 2
        return most_common, confirmed


# =====================================================================
# NODE ROS 2 TRAFFIC PERCEPTION
# =====================================================================
class TrafficPerceptionNode(Node):

    def __init__(self):
        super().__init__('traffic_perception_node')

        self.image = None
        self.bridge = CvBridge() if HAVE_CV else None

        self.light_trackers = []
        self.next_tracker_id = 1

        # Publisher tín hiệu nhận diện
        self.pub_signal = self.create_publisher(String, '/traffic_signal', 10)

        # Subscriber nhận ảnh từ Camera
        self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data
        )
        self.create_timer(1.0 / 20.0, self.tick)

        self.get_logger().info('=== TRAFFIC PERCEPTION NODE STARTED ===')

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def tick(self):
        if self.image is None:
            return
        try:
            self.process_frame()
        except Exception as e:
            self.get_logger().error(f'process_frame() error: {e}')

    # --- NHẬN DIỆN ĐÈN GIAO THÔNG ---
    def find_traffic_light_housing(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 25 or bw < 10:
                continue
            aspect = bh / float(bw)
            if 1.7 < aspect < 2.8:
                boxes.append((x, y, bw, bh))
        return boxes

    def find_lamp_circle(self, img, box):
        x, y, bw, bh = box
        roi = img[y:y + bh, x:x + bw]
        if roi.size == 0:
            return None
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < 15:
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(best)
        if radius < 4:
            return None
        return (int(x + cx), int(y + cy), int(radius))

    def match_or_create_light_tracker(self, box, now):
        for tracker in self.light_trackers:
            tx, ty, tw, th = tracker.box
            x, y, w, h = box
            if abs(x - tx) < 30 and abs(y - ty) < 30:
                tracker.box = box
                return tracker
        new_tracker = LightTracker(self.next_tracker_id, box)
        self.next_tracker_id += 1
        self.light_trackers.append(new_tracker)
        return new_tracker

    def cleanup_stale_light_trackers(self, now):
        self.light_trackers = [t for t in self.light_trackers if now - t.last_seen < 1.0]

    def detect_traffic_light(self, img, vis_img, now):
        boxes = self.find_traffic_light_housing(img)
        best = None

        MIN_LIGHT_HEIGHT = 25
        MIN_LIGHT_AREA = 300

        for box in boxes:
            x, y, bw, bh = box
            area = bw * bh
            if bh < MIN_LIGHT_HEIGHT or area < MIN_LIGHT_AREA:
                continue

            circle = self.find_lamp_circle(img, box)
            cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (255, 255, 0), 1)
            if circle is None:
                continue

            hue_color = sample_lamp_color(img, circle)
            pos_color = position_to_color(box, circle)

            if hue_color != 'UNKNOWN' and hue_color == pos_color:
                color = hue_color
            elif hue_color != 'UNKNOWN' and pos_color == 'UNKNOWN':
                color = hue_color
            elif hue_color == 'UNKNOWN' and pos_color != 'UNKNOWN':
                color = pos_color
            else:
                color = 'UNKNOWN'

            tracker = self.match_or_create_light_tracker(box, now)
            state, confirmed = tracker.update(color, now)

            draw_color = {'RED': (0, 0, 255), 'YELLOW': (0, 255, 255),
                          'GREEN': (0, 255, 0), 'UNKNOWN': (200, 200, 200)}[state]
            cx, cy, r = circle
            cv2.circle(vis_img, (cx, cy), r, draw_color, 2)
            cv2.putText(vis_img, f"L{tracker.id}:{state}{'OK' if confirmed else '?'}",
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

            if best is None or area > best[0]:
                best = (area, state, confirmed, box)

        self.cleanup_stale_light_trackers(now)

        if best is None:
            return 'UNKNOWN', False, False
        return best[1], best[2], True

    # --- NHẬN DIỆN BIỂN BÁO ---
    def detect_signs(self, img, vis_img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        stop_seen = False
        no_highway_seen = False

        for cnt in contours:
            if cv2.contourArea(cnt) < 600:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            n = len(approx)

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh + 1e-6)
            if aspect < 0.6 or aspect > 1.7:
                continue

            # Kiểm tra màu
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            pixels = hsv[mask == 255]
            if len(pixels) < 10:
                continue
            sat_ok = pixels[pixels[:, 1] > 60]
            if len(sat_ok) < 10:
                continue
            mean_hue = float(np.median(sat_ok[:, 0]))

            is_red = (mean_hue < 10 or mean_hue > 170)

            if n == 8 and is_red:
                stop_seen = True
                cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(vis_img, "STOP SIGN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif n == 4 and is_red:
                no_highway_seen = True
                cv2.rectangle(vis_img, (x, y), (x + bw, y + bh), (0, 140, 255), 2)
                cv2.putText(vis_img, "NO HIGHWAY", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        return stop_seen, no_highway_seen

    # --- NHẬN DIỆN VẠCH NGANG DỪNG ---
    def detect_stop_line(self, img, vis_img):
        # --- NHẬN DIỆN VẠCH NGANG DỪNG (ĐÃ SỬA LỖI BẮT NHẦM VẠCH DỌC/CHÉO) ---
        h, w, _ = img.shape

        # 1. Cắt ROI sát chân robot (22% phía dưới) và bóp nhẹ 2 bên mép để tránh vạch lề
        roi_top = int(h * 0.78)
        roi_margin = int(w * 0.05) # Bỏ 5% mép trái/phải
        roi = img[roi_top:, roi_margin:w - roi_margin]

        # 2. Lọc dải màu trắng (HSV)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 175])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

        # 3. Dùng Kernel làm nổi bật đường NGANG (Morphology horizontal)
        kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        thresh = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_horiz)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 350:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            abs_x = x + roi_margin  # Tọa độ X gốc
            abs_y = y + roi_top     # Tọa độ Y gốc

            aspect_ratio = bw / float(bh + 1e-6)

            # 4. Kiểm tra góc xoay minAreaRect
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0

            while angle > 90: angle -= 180
            while angle < -90: angle += 180

            # 5. SIẾT CHẶT ĐIỀU KIỆN LỌC VẠCH NGANG CHUẨN:
            # - Bề rộng vạch bw > 35% chiều rộng ảnh w
            # - Tỉ lệ ngang/dọc aspect_ratio > 3.0
            # - GÓC NGHIÊNG CHỈ CHO PHÉP LECH RẤT NHỎ (abs(angle) < 12.0°) để loại bỏ vạch chéo làn
            if bw > int(w * 0.35) and aspect_ratio > 3.0 and abs(angle) < 12.0:
                
                # Vẽ khung nhận diện chính xác
                cv2.rectangle(vis_img, (abs_x, abs_y), (abs_x + bw, abs_y + bh), (255, 0, 255), 2)
                cv2.putText(vis_img, f"STOP LINE OK ({bw/float(w):.2f})", (abs_x, abs_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                return True, float(bw / w)

        return False, 0.0

    # --- XỬ LÝ CHÍNH & PUBLISH TÍN HIỆU ---
    def process_frame(self):
        now = time.time()
        vis_img = self.image.copy()

        stop_seen, no_highway_seen = self.detect_signs(self.image, vis_img)
        light_state, light_confirmed, light_present = self.detect_traffic_light(self.image, vis_img, now)
        at_stop_line, line_ratio = self.detect_stop_line(self.image, vis_img)

        # Tổng hợp tín hiệu gửi sang Node Control
        signal = "CLEAR"
        if light_present and light_confirmed:
            if light_state in ['RED', 'YELLOW']:
                signal = light_state
        elif stop_seen:
            signal = "STOP_SIGN"
        elif no_highway_seen:
            signal = "NO_HIGHWAY"

        if at_stop_line:
            signal += "+LINE"

        msg = String()
        msg.data = signal
        self.pub_signal.publish(msg)

        # Frame Debug
        cv2.putText(vis_img, f"PUB SIGNAL: [{signal}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Traffic Perception View", vis_img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficPerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()