% --------------------------------------------
% 10초간 오디오 녹음 후 스펙트로그램 생성
% --------------------------------------------

% 1. 기본 설정
fs = 44100;        % 샘플링 주파수 (Hz)
duration = 10;     % 녹음 시간 (초)
nBits = 16;        % 비트 깊이
nChannels = 1;     % 채널 수 (1: 모노, 2: 스테레오)

% 2. 오디오 녹음 객체 생성
recObj = audiorecorder(fs, nBits, nChannels);

% 녹음 시작
disp('녹음을 시작합니다...');
recordblocking(recObj, duration);
disp('녹음이 완료되었습니다.');

% 녹음된 오디오 데이터 가져오기
audioData = getaudiodata(recObj);

% 시간 축 생성 (필요시)
t = linspace(0, duration, length(audioData));

% 3. 스펙트로그램 생성 및 표시
figure;
window = 1024;          % 윈도우 크기
noverlap = 512;         % 윈도우 겹침
nfft = 1024;            % FFT 포인트 수

% 스펙트로그램 생성
spectrogram(audioData, window, noverlap, nfft, fs, 'yaxis');

% 그래프 제목 및 라벨 설정
title('채널 노이즈의 스펙트로그램');
xlabel('시간 (초)');
ylabel('주파수 (kHz)');
colorbar;               % 컬러바 표시

% 그래프 보기 좋게 설정
colormap jet;           % 색상맵 설정
axis tight;             % 축을 데이터에 맞게 조정
