import torch
import torch.nn as nn
import numpy as np
import cv2
from cpu_upscaler import CPUImageUpscaler

class FFDNetDenoiser:
    def __init__(self):
        """FFDNet 디노이징 모델 초기화"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.build_model()
        self.model.eval()
        self.model = self.model.to(self.device)
        self.noise_level = 15

    def build_model(self):
        """RGB 입력을 올바르게 처리하는 FFDNet 모델 구조 정의"""
        return nn.Sequential(
            # 첫 번째 층 - RGB 입력 처리
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 중간 층들
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            # 출력 층 - RGB로 변환
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def preprocess_image(self, img):
        """모델 입력을 위한 이미지 전처리"""
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def add_noise_channel(self, img_tensor):
        """채널 추가 대신 노이즈 정보를 처리에 반영"""
        b, c, h, w = img_tensor.shape
        noise_level_tensor = torch.full((b, 1, h, w), self.noise_level/255.).to(self.device)
        return img_tensor * (1 + noise_level_tensor)

    def postprocess_image(self, tensor):
        """텐서를 이미지로 변환"""
        img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def denoise_image(self, img):
        """이미지 노이즈 제거 수행"""
        with torch.no_grad():
            input_tensor = self.preprocess_image(img)
            noisy_input = self.add_noise_channel(input_tensor)
            denoised = self.model(noisy_input)
            return self.postprocess_image(denoised)

class EnhancedImageProcessor:
    def __init__(self):
        """이미지 처리 파이프라인 초기화"""
        self.denoiser = FFDNetDenoiser()
        self.upscaler = CPUImageUpscaler()
    
    def process_image(self, input_path, output_path, scale=4):
        """노이즈 제거와 업스케일링을 수행하는 이미지 처리"""
        # 이미지 로드
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {input_path}")
        
        # BGR을 RGB로 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 노이즈 제거
        print("노이즈 제거 적용 중...")
        denoised_img = self.denoiser.denoise_image(img)
        
        # RGB를 BGR로 변환
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)
        
        # 중간 결과 저장
        cv2.imwrite('denoised_temp.png', denoised_img)
        
        # 업스케일링
        print("해상도 증가 처리 중...")
        success, message = self.upscaler.upscale_image(
            input_path='denoised_temp.png',
            output_path=output_path,
            scale=scale
        )
        
        return success, message
