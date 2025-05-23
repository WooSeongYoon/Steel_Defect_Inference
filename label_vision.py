import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def draw_bboxes_from_labels(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None or not os.path.exists(label_path):
        return

    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, bw, bh = map(float, parts[:5])
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def process_all_labels_with_boxes(label_root, image_root, output_root, move_labeled_images=False, move_dir=None):
    label_paths = list(Path(label_root).rglob("*.txt"))

    for label_path in tqdm(label_paths, desc="[🔄 라벨 시각화 진행 중]"):
        rel_path = label_path.relative_to(label_root).with_suffix(".jpg")
        image_path = Path(image_root) / rel_path
        output_path = Path(output_root) / rel_path

        draw_bboxes_from_labels(image_path, str(label_path), str(output_path))

        if move_labeled_images and move_dir:
            target_path = Path(move_dir) / rel_path
            os.makedirs(target_path.parent, exist_ok=True)
            if image_path.exists():
                shutil.copy(image_path, target_path)

def process_all_labels_without_boxes(label_root, image_root, move_dir):
    label_paths = list(Path(label_root).rglob("*.txt"))

    for label_path in tqdm(label_paths, desc="[📁 라벨 존재 이미지 이동 중]"):
        rel_path = label_path.relative_to(label_root).with_suffix(".jpg")
        image_path = Path(image_root) / rel_path
        target_path = Path(move_dir) / rel_path

        if image_path.exists():
            os.makedirs(target_path.parent, exist_ok=True)
            shutil.move(str(image_path), str(target_path))

if __name__ == '__main__':
    LABEL_ROOT = "/media/fourind/hdd/home/tmp/dataset/test_out/total_resultV1/labels/original"
    IMAGE_ROOT = "/media/fourind/hdd/home/tmp/dataset/test"
    OUTPUT_ROOT = "/media/fourind/hdd/home/tmp/dataset/test_out/total_resultV1/vision"
    MOVE_DIR = "/media/fourind/hdd/home/tmp/dataset/test_out/total_resultV1/original"
    DRAW_BOXES = False  # ← True: 바운딩박스 시각화 / False: 라벨 있는 이미지만 이동

    if DRAW_BOXES:
        process_all_labels_with_boxes(LABEL_ROOT, IMAGE_ROOT, OUTPUT_ROOT, move_labeled_images=True, move_dir=MOVE_DIR)
    else:
        process_all_labels_without_boxes(LABEL_ROOT, IMAGE_ROOT, MOVE_DIR)

    print("✅ 처리 완료")

