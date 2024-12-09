import cv2
import numpy as np
from PIL import Image

def denoise_with_bm3d(input_path, output_path):
    # 이미지 로드
    image = cv2.imread(input_path)
    # RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # BM3D 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)
    # 결과 저장
    denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, denoised_bgr)

# 사용 예시
denoise_with_bm3d('Lena_restored_kmeans_a.png', 'denoised_output.jpg')
