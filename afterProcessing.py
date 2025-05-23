import cv2
import os
import numpy as np

# 후처리 -> 복구, 오탐, 저장 등
class PostprocessorStep:
    def __init__(self, output_diff, output_base, save_conf=True):
        self.output_diff = output_diff
        self.output_base = output_base
        self.save_conf = save_conf

    # YOLO 포맷으로 바운딩 박스 변환
    def convert_to_yolo_format(self, bbox, img_shape):
        x1, y1, x2, y2 = bbox
        img_h, img_w = img_shape[:2]
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        return x_center, y_center, width, height
    
    # 라벨 원상 복구
    def recover(self, original_image, left_detections, right_detections, crop_dimensions, cut_height=1000):
        original_height, original_width = original_image.shape[:2]
        mid_x = original_width // 2
        restored = []

        for detections, offset in [(left_detections, 0), (right_detections, mid_x)]:
            if detections:
                for det in detections:
                    for box, cls, conf in zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy(), det.boxes.conf.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2 + offset
                        y_center = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        restored.append((int(cls), float(conf), x_center / original_width, y_center / original_height, w / original_width, h / original_height))
        return restored
    
    # 결과 저장
    def save_results(self, image, detections, base_name, output_dir, log_file=None):
        os.makedirs(output_dir, exist_ok=True)
        if detections:
            label_path = os.path.join(output_dir, base_name + ".txt")
            with open(label_path, 'w') as f:
                for cls, conf, x, y, w, h in detections:
                    if self.save_conf:
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.2f}\n")
                    else:
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        else:
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, 'a') as f:
                    f.write(f"{base_name}\n")

    # 결과 시각화
    def draw_detections(self, image, detections, false_detections=None, output_path=None):
        h, w = image.shape[:2]
        vis_image = image.copy()
        COLOR_VALID = (0, 255, 0)
        COLOR_FALSE_BG = (0, 0, 255)
        COLOR_FALSE_BRIGHT = (0, 255, 255)

        for cls, conf, x, y, bw, bh in detections:
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), COLOR_VALID, 2)
            cv2.putText(vis_image, f"cls:{cls} conf:{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VALID, 1)

        if false_detections:
            for cls, conf, x, y, bw, bh, reason in false_detections:
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                color = COLOR_FALSE_BG if reason == "background" else COLOR_FALSE_BRIGHT
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_image, f"X cls:{cls} ({reason})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)

    # 이미지 가장자리 기반 배경 스캔
    def scan_corner_background(self, image, threshold=5, visualize=False, vis_path=None):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy() if visualize else None
        COLOR_BOX = (0, 255, 255)  # 노란색

        def scan_dir(x, y, direction):
            if direction == "down":
                for y2 in range(y, h):
                    if gray[y2, x] > threshold:
                        return y2 - 1
                return h - 1
            elif direction == "up":
                for y2 in range(y, -1, -1):
                    if gray[y2, x] > threshold:
                        return y2 + 1
                return 0
            elif direction == "right":
                for x2 in range(x, w):
                    if gray[y, x2] > threshold:
                        return x2 - 1
                return w - 1
            elif direction == "left":
                for x2 in range(x, -1, -1):
                    if gray[y, x2] > threshold:
                        return x2 + 1
                return 0

        # 각 코너별 x, y 범위 추출
        tl_x = scan_dir(0, 0, "right")
        tl_y = scan_dir(0, 0, "down")

        bl_x = scan_dir(0, h - 1, "right")
        bl_y = scan_dir(0, h - 1, "up")

        tr_x = scan_dir(w - 1, 0, "left")
        tr_y = scan_dir(w - 1, 0, "down")

        br_x = scan_dir(w - 1, h - 1, "left")
        br_y = scan_dir(w - 1, h - 1, "up")

        boxes = []

        # 왼쪽 영역 (상단/하단 모두 어두운 경우)
        if tl_x > 10 and bl_x > 10:
            min_w = min(tl_x, bl_x)
            boxes.append((0, 0, min_w, h))
            # if visualize:
            #     cv2.rectangle(vis_img, (0, 0), (min_w, h), COLOR_BOX, 2)
            #     cv2.putText(vis_img, "LEFT_BG", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1)

        # 오른쪽 영역 (상단/하단 모두 어두운 경우)
        if tr_x < w - 10 and br_x < w - 10:
            max_x = max(tr_x, br_x)
            boxes.append((max_x, 0, w - 1, h))
        #     if visualize:
        #         cv2.rectangle(vis_img, (max_x, 0), (w - 1, h), COLOR_BOX, 2)
        #         cv2.putText(vis_img, "RIGHT_BG", (max_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 1)

        # if visualize and vis_path:
        #     os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        #     cv2.imwrite(vis_path, vis_img)

        return boxes


    # 객체 검출 후 상-하와 밝기 차이 확인
    def check_brightness_difference(self, image, x1, y1, x2, y2, threshold=10, brightness_diff_threshold=15):
        h, w = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_center = int((x1 + x2) / 2) # 객체 중심 x 좌표 계산

        # 상-하 영역 크기
        region_width = 20
        region_height = 30

        # x좌표 범위 계산
        x_start = max(0, x_center - region_width // 2)
        x_end = min(w, x_center + region_width // 2)

        # 상단 영역 (y=800부터 30픽셀)
        top_y1 = max(0, 800)
        top_y2 = min(h, 800 + region_height)

        # 하단 영역 (y=1000부터 30픽셀)
        bottom_y1 = max(0, 1000)
        bottom_y2 = min(h, 1000 + region_height)

        # 객체 영역
        top_region = gray_image[top_y1:top_y2, x_start:x_end]
        bottom_region = gray_image[bottom_y1:bottom_y2, x_start:x_end]
        object_region = gray_image[y1:y2, x1:x2]

        if object_region.size == 0:
            return False
        
        object_brightness = np.mean(object_region) # 객체 영역 평균 밝기

        # 조건 2: 상하 밝기와 비교 -> 상하 모두 어두운 경우 or 탐지 부분과 상하 밝기 차이가 큰 경우
        if top_region.size > 0 and bottom_region.size > 0:
            top_brightness = np.mean(top_region)
            bottom_brightness = np.mean(bottom_region)
            top_diff = abs(object_brightness - top_brightness)
            bottom_diff = abs(object_brightness - bottom_brightness)

            if top_diff > brightness_diff_threshold and bottom_diff > brightness_diff_threshold:
                return True # 객체와 상-하 밝기 차이가 큰 경우
            if top_brightness < threshold and bottom_brightness < threshold:
                return True # 상-하 모두 어두운 경우
            
        return False

    # 오탐 스캔 후 객체 검출 결과 필터링
    def background_scan(self, image, detections, base_name, output_dir, threshold=5, process_type="original", brightness_threshold=10, brightness_diff_threshold=15):
        h, w = image.shape[:2]
        background_boxes = self.scan_corner_background(image, threshold)
        filtered = []
        false_detections = []

        for det in detections:
            cls, conf, x, y, w_norm, h_norm = det
            x1 = int((x - w_norm / 2) * w)
            y1 = int((y - h_norm / 2) * h)
            x2 = int((x + w_norm / 2) * w)
            y2 = int((y + h_norm / 2) * h)

            in_background = False
            for bx1, by1, bx2, by2 in background_boxes:
                overlap_x = max(0, min(x2, bx2) - max(x1, bx1))
                overlap_y = max(0, min(y2, by2) - max(y1, by1))
                overlap_area = overlap_x * overlap_y
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > 0 and (overlap_area / box_area) > 0.7:
                    in_background = True
                    false_detections.append((cls, conf, x, y, w_norm, h_norm, "background"))
                    break
            if in_background:
                continue

            is_false = self.check_brightness_difference(
                image, x1, y1, x2, y2,
                threshold=brightness_threshold,
                brightness_diff_threshold=brightness_diff_threshold
            )
            if is_false:
                false_detections.append((cls, conf, x, y, w_norm, h_norm, "brightness"))
                continue

            filtered.append(det)

        if false_detections:
            false_dir = os.path.join(self.output_diff, process_type, os.path.relpath(output_dir, self.output_base))
            os.makedirs(false_dir, exist_ok=True)
            false_label_path = os.path.join(false_dir, base_name + ".txt")
            with open(false_label_path, 'w') as f:
                for cls, conf, x, y, w, h, reason in false_detections:
                    if self.save_conf:
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.2f} {reason}\n")
                    else:
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {reason}\n")
                        
        return filtered, false_detections