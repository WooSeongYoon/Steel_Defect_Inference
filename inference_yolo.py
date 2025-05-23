import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from afterProcessing import PostprocessorStep  # 후처리 클래스
from beforeProcessing import PreprocessorStep  # 전처리 클래스


class YOLOProcessor:
    def __init__(
        self, model_path, input_dir,
        output_base, output_diff, output_stop, output_visual,
        stop_check=True, save_conf=True, visualize=False
    ):
        # YOLO 모델 로드
        self.model = YOLO(model_path)

        # 입출력 디렉토리 경로 설정
        self.input_dir = input_dir
        self.output_base = output_base
        self.output_diff = output_diff
        self.output_stop = output_stop
        self.output_visual = output_visual

        # 설정 옵션
        self.stop_check = stop_check      # 정지 이미지 체크 여부
        self.save_conf = save_conf        # confidence 저장 여부
        self.visualize = visualize        # 시각화 이미지 저장 여부

        # 전처리 및 후처리 클래스 초기화
        self.preprocessor = PreprocessorStep()
        self.postprocessor = PostprocessorStep(output_diff=output_diff, output_base=output_base, save_conf=save_conf)

    def detect(self, image):
        # YOLO 추론 수행
        return self.model.predict(image, conf=0.2, iou=0.4, verbose=False)

    def process_image(self, img_path, rel_path, process_type="original"):
        # 이미지 읽기
        image = cv2.imread(img_path)
        if image is None:
            return False

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 정지 이미지 검사
        if self.stop_check and self.preprocessor.black_image(image):
            stop_dir = os.path.join(self.output_stop, rel_path)
            os.makedirs(stop_dir, exist_ok=True)
            stop_log = os.path.join(stop_dir, 'stop.txt')
            with open(stop_log, 'a') as f:
                f.write(f"{base_name}\n")
            return False

        # 결과 저장 디렉토리 설정
        output_dir = os.path.join(self.output_base, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        undetected_log = os.path.join(output_dir, 'undetected.txt')

        try:
            # 이미지 분할 (좌/우) 및 추론
            left, right, cropped_shape = self.preprocessor.split_image(image)
            results_left = self.detect(left)
            results_right = self.detect(right)

            # 좌/우 결과 복원 및 결합 (정규화된 YOLO 포맷 반환)
            detections_combined = self.postprocessor.recover(image, results_left, results_right, cropped_shape)

            # 오탐 필터링
            detections_filtered, false_detections = self.postprocessor.background_scan(image, detections_combined, base_name, output_dir, threshold=5, process_type=process_type, brightness_threshold=10, brightness_diff_threshold=15)

            # 배경 영역 시각화
            # if self.visualize:
            #     bg_vis_path = os.path.join(self.output_visual, rel_path, base_name + "_bg.jpg")
            #     self.postprocessor.scan_corner_background(image, threshold=5, visualize=True, vis_path=bg_vis_path)

            # 시각화 이미지 저장
            if self.visualize and (detections_filtered or false_detections):
                vis_path = os.path.join(self.output_visual, rel_path, base_name + "_vis.jpg")
                self.postprocessor.draw_detections(image, detections_filtered, false_detections, vis_path)

            # 탐지 결과 저장
            if detections_filtered:
                self.postprocessor.save_results(image, detections_filtered, base_name, output_dir, undetected_log)
                return True
            else:
                with open(undetected_log, 'a') as f:
                    f.write(f"{base_name}\n")
                return False

        except ValueError as e:
            print(f"[ERROR - Split 실패] {img_path}: {e}")
            return False

    def process_directory(self, directory=None, rel_path=""):
        # 디렉토리 탐색 및 이미지 처리
        if directory is None:
            directory = self.input_dir

        entries = os.listdir(directory)
        for entry in tqdm(entries, desc=f"yolo Processing {os.path.basename(directory)}"):
            full_path = os.path.join(directory, entry)
            current_rel_path = os.path.join(rel_path, entry) if rel_path else entry

            if os.path.isdir(full_path):
                self.process_directory(full_path, current_rel_path)
            elif os.path.isfile(full_path) and entry.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    self.process_image(full_path, rel_path, "original")
                    os.makedirs(self.output_visual, exist_ok=True)
                except Exception as e:
                    print(f"[ERROR] {full_path}: {e}")

    def run(self):
        # 기본 폴더 생성 후 전체 이미지 추론 실행
        os.makedirs(self.output_base, exist_ok=True)
        os.makedirs(self.output_stop, exist_ok=True)
        os.makedirs(self.output_diff, exist_ok=True)
        print("모든 이미지 추론 중")
        self.process_directory()
        print("추론 완료")


# 명령어로 직접 실행될 경우
if __name__ == "__main__":
    processor = YOLOProcessor(
        model_path="./weights/yolov8l_best.pt",
        input_dir="/media/fourind/hdd/home/tmp/dataset/test/02/19/10",
        output_base="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8l/detected/original",
        output_diff="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8l/diff",
        output_stop="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8l/stop",
        output_visual="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8l/visuals",
        stop_check=True,
        save_conf=True,
        visualize=True
    )
    processor.run()
