% 오디오 녹음 및 스펙트로그램 분석 코드
fs = 44100;           % 샘플링 주파수 (Hz)
duration = 10;        % 녹음 시간 (초)
nBits = 16;          % 비트 심도
nChannels = 1;       % 모노 채널

% 오디오 레코딩 객체 생성
recObj = audiorecorder(fs, nBits, nChannels);

% 녹음 시작
disp('녹음을 시작합니다...');
recordblocking(recObj, duration);
disp('녹음이 완료되었습니다.');

% 녹음된 데이터 가져오기
y = getaudiodata(recObj);

% 스펙트로그램 생성
figure;
spectrogram(y, hamming(1024), 512, 1024, fs, 'yaxis');
title('채널 노이즈 스펙트로그램');
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
colorbar;
colormap('jet');

% 시간 도메인에서의 신호도 표시
figure;
t = (0:length(y)-1)/fs;
plot(t, y);
title('시간 도메인 신호');
xlabel('시간 (초)');
ylabel('진폭');
grid on;