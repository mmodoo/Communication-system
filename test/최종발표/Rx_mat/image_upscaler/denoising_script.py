import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from urllib.request import urlretrieve

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # 첫 번째 층
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, 
                              kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 중간 층들
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                  kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # 마지막 층
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                              kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise

def download_weights():
    # 가중치 파일을 저장할 디렉토리 생성
    weights_dir = os.path.join(os.getcwd(), 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # 가중치 파일 경로
    weights_path = os.path.join(weights_dir, 'dncnn_rgb.pth')
    
    # 가중치 다운로드 (예시 URL - 실제 작동하는 URL로 교체 필요)
    weights_url = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color.pth"
    
    if not os.path.exists(weights_path):
        print("가중치 파일 다운로드 중...")
        urlretrieve(weights_url, weights_path)
        print("다운로드 완료!")
    
    return weights_path

def denoise_image(input_path, output_path):
    # CPU 디바이스 설정
    device = torch.device('cpu')
    
    # 모델 초기화
    model = DnCNN().to(device)
    
    # 가중치 다운로드 및 로드
    weights_path = download_weights()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 이미지 로드
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 전처리
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    # 노이즈 제거
    with torch.no_grad():
        denoised = model(img)
    
    # 결과 후처리
    denoised = denoised.squeeze().permute(1, 2, 0).numpy()
    denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
    
    # 결과 저장
    cv2.imwrite(output_path, denoised)

if __name__ == "__main__":
    input_image = "Lena_restored_kmeans_a.png"
    output_image = "Lena_salt_pepper_removed.png"
    denoise_image(input_image, output_image)
