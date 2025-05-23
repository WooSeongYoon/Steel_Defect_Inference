import sys
sys.path.insert(0, "/home/fourind/programming/hankum/CoDETR")

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from afterProcessing import PostprocessorStep  # 후처리 클래스
from beforeProcessing import PreprocessorStep  # 전처리 클래스

class CoDETRPipeline:
    def __init__(self, config_path, checkpoint_path, input_dir, output_dir, score_thr=0.3, visual=False):
        self.model = init_detector(Config.fromfile(config_path), checkpoint_path, device='cuda:0')
        self.input_dir = input_dir
        self.output_base = os.path.join(output_dir, "detected")
        self.output_diff = os.path.join(output_dir, "diff")
        self.output_visual = os.path.join(output_dir, "visuals")
        self.output_stop = os.path.join(output_dir, "stop")
        self.output_undetected = os.path.join(output_dir, "undetected")
        self.stop_txt_path = os.path.join(self.output_stop, 'stop.txt')
        self.undetected_txt_path = os.path.join(self.output_undetected, 'undetected.txt')
        self.score_thr = score_thr
        self.visual = visual

        self.preprocessor = PreprocessorStep()
        self.postprocessor = PostprocessorStep(self.output_diff, self.output_base)

        for d in [self.output_base, self.output_diff, self.output_visual, self.output_stop, self.output_undetected]:
            os.makedirs(d, exist_ok=True)

    def detect(self, image):
        return inference_detector(self.model, image)

    def run(self):
        image_files = sorted(Path(self.input_dir).rglob("*.jpg")) + \
                      sorted(Path(self.input_dir).rglob("*.jpeg")) + \
                      sorted(Path(self.input_dir).rglob("*.png"))

        for img_path in tqdm(image_files, desc="Running inference"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rel_path = os.path.relpath(img_path, self.input_dir)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            version_out_dir = os.path.join(self.output_base, os.path.dirname(rel_path))
            os.makedirs(version_out_dir, exist_ok=True)
            output_txt_path = os.path.join(version_out_dir, base_name + ".txt")

            if self.preprocessor.black_image(img):
                with open(self.stop_txt_path, 'a') as f:
                    f.write(rel_path + '\n')
                continue

            try:
                left, right, crop_shape = self.preprocessor.split_image(img)
            except ValueError:
                continue

            left_result = self.detect(left)
            right_result = self.detect(right)

            detections = []
            for result, offset_x in zip([left_result, right_result], [0, crop_shape[1] // 2]):
                for cls_id, bboxes in enumerate(result):
                    for bbox in bboxes:
                        if bbox[4] >= self.score_thr:
                            x1 = bbox[0] + offset_x
                            y1 = bbox[1]
                            x2 = bbox[2] + offset_x
                            y2 = bbox[3]
                            yolo_box = self.postprocessor.convert_to_yolo_format((x1, y1, x2, y2), img.shape)
                            detections.append((cls_id, bbox[4], *yolo_box))

            if not detections:
                with open(self.undetected_txt_path, 'a') as f:
                    f.write(rel_path + '\n')
                continue

            # 오탐 필터링 적용
            filtered_detections, false_detections = self.postprocessor.background_scan(img, detections, base_name, version_out_dir, threshold=5, process_type="original", brightness_threshold=10, brightness_diff_threshold=15)

            # 결과 저장
            self.postprocessor.save_results(img, filtered_detections, base_name, version_out_dir, self.undetected_txt_path)

            # 시각화 저장
            if self.visual:
                vis_path = os.path.join(self.output_visual, rel_path)
                self.postprocessor.draw_detections(img, filtered_detections, false_detections, vis_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Co-DETR inference directly with .py')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--img_folder', required=True, help='Path to folder of input images')
    parser.add_argument('--out_dir', default='inference_results', help='Path to save results')
    parser.add_argument('--score_thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--visual', action='store_true', help='Enable saving visualized images')
    args = parser.parse_args()

    pipeline = CoDETRPipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_dir=args.img_folder,
        output_dir=args.out_dir,
        score_thr=args.score_thr,
        visual=args.visual
    )
    pipeline.run()
