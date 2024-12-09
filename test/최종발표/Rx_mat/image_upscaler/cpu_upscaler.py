import os
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union

class CPUImageUpscaler:
    def __init__(self, model_name: str = 'RealESRGAN_x4plus') -> None:
        """
        Python 3.9 호환 초기화 함수
        Args:
            model_name: 사용할 모델 이름 (기본값: 'RealESRGAN_x4plus')
        """
        self.model_name = model_name
        self.device = torch.device('cpu')
        
        # 모델 초기화
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # CPU 최적화 설정의 upsampler
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth',
            model=model,
            tile=400,  # CPU 메모리 최적화 값
            tile_pad=10,
            pre_pad=0,
            half=False,  # CPU 처리를 위한 FP32 사용
            device=self.device
        )

    def upscale_image(self, 
                      input_path: str, 
                      output_path: str, 
                      scale: int = 2) -> Tuple[bool, str]:
        """
        이미지 업스케일링 함수
        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 저장 경로
            scale: 업스케일 비율 (기본값: 2)
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        try:
            # 이미지 존재 확인
            if not os.path.exists(input_path):
                return False, f"입력 파일이 존재하지 않습니다: {input_path}"

            # 이미지 읽기
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, f"이미지 로딩 실패: {input_path}"
            
            # 이미지 크기 확인
            height, width = img.shape[:2]
            print(f"입력 이미지 크기: {width}x{height}")
            
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print("업스케일링 처리 중...")
            output, _ = self.upsampler.enhance(img, outscale=scale)
            
            if len(output.shape) == 3 and output.shape[2] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # 결과 저장
            cv2.imwrite(output_path, output)
            
            output_height, output_width = output.shape[:2]
            success_msg = (f"처리 완료\n"
                         f"원본 크기: {width}x{height}\n"
                         f"변환 크기: {output_width}x{output_height}")
            print(success_msg)
            
            return True, success_msg

        except Exception as e:
            error_msg = f"처리 중 오류 발생: {str(e)}"
            print(error_msg)
            return False, error_msg

    def process_matlab_array(self, 
                           input_array: np.ndarray, 
                           scale: int = 2) -> Optional[np.ndarray]:
        """
        MATLAB 배열 직접 처리 함수
        Args:
            input_array: MATLAB에서 전달받은 numpy 배열
            scale: 업스케일 비율
        Returns:
            Optional[np.ndarray]: 처리된 이미지 배열 또는 None
        """
        try:
            if input_array is None or input_array.size == 0:
                print("유효하지 않은 입력 배열")
                return None

            # RGB 순서 변환 (MATLAB RGB -> OpenCV BGR)
            if len(input_array.shape) == 3 and input_array.shape[2] == 3:
                input_array = cv2.cvtColor(input_array, cv2.COLOR_RGB2BGR)

            # 업스케일링 수행
            output, _ = self.upsampler.enhance(input_array, outscale=scale)

            # BGR -> RGB 변환
            if len(output.shape) == 3 and output.shape[2] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            return output

        except Exception as e:
            print(f"배열 처리 중 오류 발생: {str(e)}")
            return None

# 사용 예시
if __name__ == "__main__":
    upscaler = CPUImageUpscaler()
    
    # 파일 기반 처리
    success, message = upscaler.upscale_image(
        input_path="input.jpg",
        output_path="output.jpg",
        scale=2
    )
    
    # MATLAB 연동 예시
    """
    # MATLAB에서 다음과 같이 사용:
    # Python 설정
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    # 이미지 로드 및 변환
    img = eng.imread('input.jpg')
    python_array = np.array(img)
    
    # 업스케일링 수행
    upscaler = CPUImageUpscaler()
    result = upscaler.process_matlab_array(python_array)
    
    # 결과 처리
    if result is not None:
        matlab_array = matlab.double(result.tolist())
        eng.imwrite(matlab_array, 'output.jpg')
    """
