% 파일 경로 설정 (필요에 따라 주석 해제)
% MAT_path_and_file_name = 'C:\Users\user\Desktop\졸업드가자\통신시스템설계\최종발표\kmeans_logical_data.mat';
MAT_path_and_file_name = 'C:\Users\user\Desktop\졸업드가자\통신시스템설계\최종발표\test.mat';

% 파라미터 설정
Modulation_Number = 4;
N = 256;
Subcarrier_Freq_Divided_by = 6;

Sampling_Freq = 48000;          % 샘플링 주파수 (Hz)
Cut_off_Freq = 16000;           % 컷오프 주파수 (Hz)
Cut_off_Freq_normalised = Cut_off_Freq / (Sampling_Freq / 2);

T_preamble_sec = 2;             % 프리앰블 지속 시간 (초)
t_preamble_vector = (0:1/Sampling_Freq:T_preamble_sec).';  % 시간 벡터

Freq_Preamble_Start = 18000;    % 프리앰블 시작 주파수 (Hz)
Freq_Preamble_End = 19000;      % 프리앰블 종료 주파수 (Hz)

% 프리앰블 신호 생성 (주파수 변조된 코사인 신호)
Phase_Preamble = 2 * pi * (Freq_Preamble_Start * t_preamble_vector + ...
    ((Freq_Preamble_End - Freq_Preamble_Start)/(2*T_preamble_sec)) * t_preamble_vector.^2);
Preamble = cos(Phase_Preamble);

% 스펙트로그램 파라미터 설정
window_length = 1024;            % 윈도우 길이 (샘플)
noverlap = 512;                  % 오버랩 길이 (샘플)
nfft = 2048;                     % FFT 포인트 수

% 스펙트로그램 계산
[S, F, T, P] = spectrogram(Preamble, window_length, noverlap, nfft, Sampling_Freq, 'yaxis');

% 스펙트로그램 시각화
figure;
surf(T, F, 10*log10(P), 'EdgeColor', 'none');
axis tight;
view(0, 90);
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('Preamble 신호의 스펙트로그램');
colorbar;
caxis([-100 -30]);  % 색상 범위 조정 (필요에 따라 조절)
colormap jet;

% 추가: 그래프 보기 개선
set(gca, 'FontSize', 12);
