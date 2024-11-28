% 파라미터 설정
fs = 48000;          % 샘플링 주파수 (Hz)
duration = 3;        % 지속 시간 (초)
freq = 16000;        % 원하는 주파수 (Hz)

% 시간 벡터 생성
t = (0:1/fs:duration-1/fs)';

% 순음 생성
signal = sin(2*pi*freq*t);

% 소리 재생
sound(signal, fs);

% 또는 audioplayer 사용
% player = audioplayer(signal, fs);
% play(player);

% wav 파일로 저장하고 싶다면
% audiowrite('sine_14khz.wav', signal, fs);

% 생성된 신호의 정보 출력
fprintf('신호 길이: %d 샘플\n', length(signal));
fprintf('실제 재생 시간: %.2f 초\n', length(signal)/fs);