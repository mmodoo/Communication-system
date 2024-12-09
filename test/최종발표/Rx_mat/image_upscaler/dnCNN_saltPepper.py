import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# DnCNN 모델 정의 (간단한 버전)
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return x - out

# 모델 초기화 및 가중치 로드
model = DnCNN(channels=3, num_of_layers=17)
model.load_state_dict(torch.load('DnCNN.pth', map_location='cpu'))
model.eval()

# 노이즈 제거 함수
def denoise_image(input_path, output_path):
    # 이미지 로드 및 전처리
    image = Image.open(input_path).convert('RGB')
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    # 모델 적용
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    
    # 후처리 및 저장
    denoised_np = denoised_tensor.squeeze().permute(1, 2, 0).numpy()
    denoised_np = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)
    denoised_image = Image.fromarray(denoised_np)
    denoised_image.save(output_path)

# 사용 예시
denoise_image('Lena_restored_kmeans_a.png', 'denoised_output.png')
