Fs = 10000;
repetition_factor = 3;
% recording_time_sec = 17 * 5 + 10;
recording_time_sec = 20;

% Preamble
omega = 10;
mu = 0.1;
Tp = 1000;
tp = (1:Tp).';
preamble = cos(omega * tp + mu * tp.^2 / 2);
% M = length(symbols);
M = 16384 * 3;
N = 256;
N_cp = N / 4;
cn = M / (N / 2);
N_blk = cn + cn / 4;

% 수신 시작
devicereader = audioDeviceReader(Fs);
setup(devicereader);
disp('Recording...')
rx_signal = [];
tic;
while toc < recording_time_sec  % recording_time_sec 동안 녹음
    acquiredAudio = devicereader();
    rx_signal = [rx_signal; acquiredAudio];
end
disp('Recording Completed')

% Time Synchronisation
[xC, lags] = xcorr(rx_signal, preamble);
[~, idx] = max(xC);
start_pt = lags(idx);

rx_signal = rx_signal(start_pt + Tp + 1 : end);

% Serial to Parallel
OFDM_blks = {};
for i = 1 : N_blk
    OFDM_blks{end + 1} = rx_signal(N_cp + 1 : N + N_cp);
    rx_signal = rx_signal(N_cp + N + 1 : end);
end

% Discrete Fourier Transform (DFT)
demode_OFDM_blks = {};
for i = 1 : length(OFDM_blks)
    demode_OFDM_blks{end + 1} = fft(OFDM_blks{i} / sqrt(N));
end

% Channel Estimation & Equalisation
symbols_eq = {};
for i = 1 : length(demode_OFDM_blks)
    if rem(i, 5) == 1
        channel = demode_OFDM_blks{i} ./ ones(N, 1);
    else
        symbols_eq{end + 1} = demode_OFDM_blks{i} ./ channel;
    end
end

% Detection
symbols_detect = {};
for i = 1 : length(symbols_eq)
    symbols_detect{end + 1} = sign(real(symbols_eq{i}));
end

% Demodulation
symbols_est = [];
for i = 1 : length(symbols_detect)
    symbols_est = [symbols_est; symbols_detect{i}(2 : N / 2 + 1)];
end

% BPSK Demodulation
decoded_bits = (symbols_est + 1) / 2;

% Repetition Decoding
% 3개씩 묶어서 reshape
decoded_bits_reshaped = reshape(decoded_bits, repetition_factor, []);

% 각 열의 합을 계산 (다수결 원칙 적용)
sums = sum(decoded_bits_reshaped);

% 합이 2 이상이면 1, 미만이면 0으로 판정
repetition_decoded_bits = (sums > repetition_factor / 2).';

% Source Decoding & Show img
% 이미지 크기 계산 시 repetition decoding으로 인한 크기 변화 고려
img_size = sqrt(length(repetition_decoded_bits));
estimated_img = reshape(repetition_decoded_bits, [img_size, img_size]); 
resized_estimated_img = imresize(estimated_img, 1 / imresize_scale);
imshow(resized_estimated_img);