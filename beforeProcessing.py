import cv2
import os

# 전처리 -> 이미지 자르기, 밝기 조정, CLAHE 적용 등
class PreprocessorStep:
    def __init__(self, check_height=1000, threshold=5, cut_height=1000):
        self.check_height = check_height
        self.threshold = threshold
        self.cut_height = cut_height

    # 상단 일부 평균 밝기 기준으로 정지 이미지 판단
    def black_image(self, image):
        return image[:self.check_height, :].mean() < self.threshold

    # 이미지 하단 자르고 좌우로 분할
    def split_image(self, image):
        h, w = image.shape[:2]
        if h <= self.cut_height:
            raise ValueError("이미지 높이가 너무 작습니다.")
        cropped = image[:h - self.cut_height, :]
        mid_x = w // 2
        return cropped[:, :mid_x], cropped[:, mid_x:], cropped.shape[:2]

    # 이미지 밝기 조정
    def brightness_preprocess(self, image, alpha=1.5, beta=30):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # CLAHE 적용
    def clahe_preprocess(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    def clahe_brightness_preprocess(self, image, alpha=1.5, beta=30):
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
