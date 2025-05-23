import os
import sys
import threading
from inference_yolo import YOLOProcessor
from CoDETR.inference_custom import CoDETRPipeline
from results_conjunction import YOLOLabelCombiner

IMG_DSIR = "/media/fourind/hdd/home/tmp/dataset/test"

def run_yolo():
    processor = YOLOProcessor(
        model_path="./weights/yolov8l_best.pt",
        input_dir=IMG_DSIR,
        output_base="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8lV1/detected/original",
        output_diff="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8lV1/diff",
        output_stop="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8l/stop",
        output_visual="/media/fourind/hdd/home/tmp/dataset/test_out/yolov8lV1/visuals",
        stop_check=True,
        save_conf=True,
        visualize=False
    )
    processor.run()


def run_codetr():
    pipeline = CoDETRPipeline(
    config_path="./CoDETR/projects/configs/co_deformable_detr/co_detr_custom_4060ti.py",
    checkpoint_path="./weights/co_detr_best.pth",
    input_dir=IMG_DSIR,
    output_dir="/media/fourind/hdd/home/tmp/dataset/test_out/co-detrV1",
    score_thr=0.7,
    visual=False
    )
    pipeline.run()

def run_label_combiner():
    combiner = YOLOLabelCombiner(
        codetr_base_dir='/media/fourind/hdd/home/tmp/dataset/test_out/co-detrV1/detected',
        yolov8_base_dir='/media/fourind/hdd/home/tmp/dataset/test_out/yolov8lV1/detected',
        output_base_dir='/media/fourind/hdd/home/tmp/dataset/test_out/total_resultV1/labels',
        visualization_dir='/media/fourind/hdd/home/tmp/dataset/test_out/total_resultV1/visuals',
        images_base_dir=IMG_DSIR,
        save_visual=False
    )
    combiner.run()

def main():
    # 1. YOLO와 Co-DETR 병렬 추론
    yolo_thread = threading.Thread(target=run_yolo)
    codetr_thread = threading.Thread(target=run_codetr)

    yolo_thread.start()
    codetr_thread.start()

    yolo_thread.join()
    codetr_thread.join()
    print("🔍 YOLO & Co-DETR 추론 완료")

    # 2. 라벨 결합 및 저장
    run_label_combiner()
    print("✅ 모든 처리 완료")


if __name__ == "__main__":
    main()
