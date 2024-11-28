% 샘플링 주파수 설정 (Hz)
fs = 48000;

% 재생 시간 설정 (초)
duration = 5;

% 사인파 주파수 설정 (Hz)
f = 15000;

% 시간 벡터 생성
t = (0:1/fs:duration).';

% 사인파 신호 생성
y = sin(2 * pi * f * t);

% 사인파 재생
sound(y, fs);
