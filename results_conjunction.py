import os
import cv2
import numpy as np


class YOLOLabelCombiner:
    def __init__(self, codetr_base_dir, yolov8_base_dir, output_base_dir, visualization_dir, images_base_dir, save_visual=True):
        self.codetr_base_dir = codetr_base_dir
        self.yolov8_base_dir = yolov8_base_dir
        self.output_base_dir = output_base_dir
        self.visualization_dir = visualization_dir
        self.images_base_dir = images_base_dir
        self.save_visual = save_visual
        self.folder_types = ['original']

    def read_yolo_label(self, label_path, img_width, img_height):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split()
                if len(data) >= 5:
                    class_id = int(data[0])
                    x_center = float(data[1]) * img_width
                    y_center = float(data[2]) * img_height
                    width = float(data[3]) * img_width
                    height = float(data[4]) * img_height
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    boxes.append([class_id, x_min, y_min, x_max, y_max])
        return boxes

    def convert_to_yolo_format(self, box, img_width, img_height):
        class_id, x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return [class_id, x_center, y_center, width, height]

    def compute_iou(self, box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1[1:]
        x_min2, y_min2, x_max2, y_max2 = box2[1:]
        x_min_inter = max(x_min1, x_min2)
        y_min_inter = max(y_min1, y_min2)
        x_max_inter = min(x_max1, x_max2)
        y_max_inter = min(y_max1, y_max2)
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return 0.0
        inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
        box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
        return inter_area / (box1_area + box2_area - inter_area)

    def get_box_area(self, box):
        x_min, y_min, x_max, y_max = box[1:]
        return (x_max - x_min) * (y_max - y_min)

    def combine_labels(self, codetr_label_path, yolov8_label_path, img_width, img_height, iou_threshold=0.6):
        codetr_boxes = self.read_yolo_label(codetr_label_path, img_width, img_height)
        yolov8_boxes = self.read_yolo_label(yolov8_label_path, img_width, img_height)
        combined_boxes = []
        removed_boxes = []
        combined_boxes.extend(codetr_boxes)
        for yolo_box in yolov8_boxes:
            should_add = True
            indices_to_remove = []
            for i, combined_box in enumerate(combined_boxes):
                if yolo_box[0] != combined_box[0]:
                    continue
                iou = self.compute_iou(yolo_box, combined_box)
                if iou >= iou_threshold:
                    yolo_area = self.get_box_area(yolo_box)
                    combined_area = self.get_box_area(combined_box)
                    if yolo_area > combined_area:
                        indices_to_remove.append(i)
                        removed_boxes.append(combined_box)
                    else:
                        should_add = False
                        removed_boxes.append(yolo_box)
                        break
            for idx in sorted(indices_to_remove, reverse=True):
                del combined_boxes[idx]
            if should_add:
                combined_boxes.append(yolo_box)
        yolo_format_boxes = [self.convert_to_yolo_format(box, img_width, img_height) for box in combined_boxes]
        return yolo_format_boxes, removed_boxes

    def save_yolo_label(self, output_path, boxes):
        with open(output_path, 'w') as f:
            for box in boxes:
                class_id, x_center, y_center, width, height = box
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def visualize_labels(self, image_path, codetr_label_path, yolov8_label_path, output_path, removed_boxes=None):
        codetr_color = (0, 255, 0)
        yolov8_color = (255, 0, 0)
        removed_color = (0, 165, 255)
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            return
        height, width = image.shape[:2]

        if codetr_label_path and os.path.exists(codetr_label_path):
            codetr_boxes = self.read_yolo_label(codetr_label_path, width, height)
            for box in codetr_boxes:
                _, x_min, y_min, x_max, y_max = box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), codetr_color, 2)
                cv2.putText(image, "Co-DeTR", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, codetr_color, 1)

        if yolov8_label_path and os.path.exists(yolov8_label_path):
            yolov8_boxes = self.read_yolo_label(yolov8_label_path, width, height)
            for box in yolov8_boxes:
                _, x_min, y_min, x_max, y_max = box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), yolov8_color, 2)
                cv2.putText(image, "YOLOv8", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolov8_color, 1)

        if removed_boxes:
            for box in removed_boxes:
                _, x_min, y_min, x_max, y_max = box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), removed_color, 2)
                cv2.putText(image, "REMOVED", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, removed_color, 1)

        if self.save_visual:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"시각화 이미지 저장됨: {output_path}")

    def get_all_image_paths(self):
        image_paths = {}
        for root, _, files in os.walk(self.images_base_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.images_base_dir)
                    rel_dir, filename = os.path.split(rel_path)
                    image_paths.setdefault(rel_dir, []).append((filename, full_path))
        return image_paths

    def run(self):
        image_paths = self.get_all_image_paths()

        for folder_type in self.folder_types:
            print(f"전처리 폴더 처리 중: {folder_type}")
            for rel_path, image_list in image_paths.items():
                for image_filename, image_path in image_list:
                    print(f"처리 중: {os.path.join(rel_path, image_filename)}")
                    label_filename = os.path.splitext(image_filename)[0] + '.txt'
                    rel_label_path = os.path.join(rel_path, label_filename)

                    codetr_label_path = os.path.join(self.codetr_base_dir, folder_type, rel_label_path)
                    yolov8_label_path = os.path.join(self.yolov8_base_dir, folder_type, rel_label_path)

                    codetr_exists = os.path.exists(codetr_label_path)
                    yolov8_exists = os.path.exists(yolov8_label_path)

                    if not codetr_exists and not yolov8_exists:
                        print("라벨 없음: 건너뜀")
                        continue

                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"이미지 열기 실패: {image_path}")
                        continue
                    img_height, img_width = img.shape[:2]

                    if codetr_exists and yolov8_exists:
                        combined_boxes, removed_boxes = self.combine_labels(codetr_label_path, yolov8_label_path, img_width, img_height)
                    elif codetr_exists:
                        codetr_boxes = self.read_yolo_label(codetr_label_path, img_width, img_height)
                        combined_boxes = [self.convert_to_yolo_format(box, img_width, img_height) for box in codetr_boxes]
                        removed_boxes = []
                    else:
                        yolov8_boxes = self.read_yolo_label(yolov8_label_path, img_width, img_height)
                        combined_boxes = [self.convert_to_yolo_format(box, img_width, img_height) for box in yolov8_boxes]
                        removed_boxes = []

                    output_label_path = os.path.join(self.output_base_dir, folder_type, rel_label_path)
                    output_viz_path = os.path.join(self.visualization_dir, folder_type, rel_path, os.path.splitext(image_filename)[0] + '_viz.jpg')

                    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
                    self.save_yolo_label(output_label_path, combined_boxes)
                    self.visualize_labels(image_path, codetr_label_path if codetr_exists else "", yolov8_label_path if yolov8_exists else "", output_viz_path, removed_boxes=removed_boxes)
                    print(f"저장 완료: {output_label_path}")


if __name__ == "__main__":
    combiner = YOLOLabelCombiner(
        codetr_base_dir='/media/fourind/hdd/home/tmp/dataset/20250514/co-detr/detected',
        yolov8_base_dir='/media/fourind/hdd/home/tmp/dataset/20250514/yolov8l/detected',
        output_base_dir='/media/fourind/hdd/home/tmp/dataset/20250514/total_result/labels',
        visualization_dir='/media/fourind/hdd/home/tmp/dataset/20250514/total_result/visuals',
        images_base_dir='/media/fourind/hdd/home/tmp/Downloads',
        save_visual=True
    )
    combiner.run()