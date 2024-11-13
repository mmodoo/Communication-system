clc;
clear;
close all;


Fs = 10000;
% repetition_factor = 3;
% recording_time_sec = 17 * 5 + 10;
recording_time_sec = 12;


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

% Source Encoding
imresize_scale = 0.5;
img = imread( 'Lena_color.png');
resized_img = imresize(img, imresize_scale);
gray_img = rgb2gray(resized_img);
binarized_img = imbinarize(gray_img);
bits = binarized_img(:);

k = 4; % Number of valid bits in 1 coded bits
c = 3; % Number of parities in 1 coded bits
n = k + c; % Number of total bits in 1 coded bits
A = [1 1 1;
     1 1 0;
     1 0 1;
     0 1 1]; % Parity generator matrix
G = [eye(k), A]; % Code generator matrix

% Parity-check matrix H for decoding
H = [A', eye(c)];

reshaped_bits = reshape(bits, [k, length(bits)/k])';

% Step 3: Encode each 4-bit block using Hamming (7,4)
hamming_coded_bits = mod(reshaped_bits * G, 2);

transpose_hamming_coded_bits = hamming_coded_bits.';
channel_coded_bits = transpose_hamming_coded_bits(:);

% disp('symbol len, 전체 OFDM 심볼의 개수, pilot 신호를 포함한 전체 블록의 개수를 설정합니다.');
M = length(channel_coded_bits);
N = 256;
N_cp = N / 4;
cn = M / (N / 2);
N_blk = cn + cn / 4;

% Preamble
omega = 10;
mu = 0.1;
Tp = 1000;
tp = (1:Tp).';
preamble = cos(omega * tp + mu * tp.^2 / 2);


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
symbols_detect_a = {};
symbols_detect = {};

for i = 1 : length(symbols_eq)
    symbols_detect{end + 1} = sign(real(symbols_eq{i}));
    symbols_detect_a{end + 1} = real(symbols_eq{i});

end

% Demodulation
symbols_est = [];
received_symbols = [];
for i = 1 : length(symbols_detect)
    symbols_est = [symbols_est; symbols_detect{i}(2 : N / 2 + 1)];
    received_symbols = [received_symbols; symbols_detect_a{i}(2 : N / 2 + 1)];
end

% BPSK Demodulation
demodulated_bits = (symbols_est + 1) / 2;

bits = bits';

% Step 5: Reshape received bits for decoding
received_bits = reshape(demodulated_bits, [n, length(demodulated_bits)/n])';

% Step 6: Hard Decision Decoding (error correction using syndrome)
syndrome_matrix = mod(received_bits * H', 2);
HD_decoded_bits = zeros(size(reshaped_bits));  % Initialize for storing decoded bits

for i = 1:size(syndrome_matrix, 1)
    % Convert syndrome to decimal to identify error position
    uncoded_bits(i,:)= received_bits(i, 1:k);

    syndrome_decimal = bi2de(syndrome_matrix(i, :), 'left-msb');
    correction_bits = 8-syndrome_decimal;
    if syndrome_decimal ~= 0
        % If an error is detected, correct it by flipping the bit at the error position
        if syndrome_decimal == 3
            received_bits(i, 4) = ~received_bits(i, 4);
        elseif syndrome_decimal == 4
            received_bits(i, 5) = ~received_bits(i, 5);
        else
            received_bits(i, correction_bits) = ~received_bits(i, correction_bits);
        end
    end
    % Extract the corrected data bits (first 4 bits)
    HD_decoded_bits(i, :) = received_bits(i, 1:k);
    
end

%% uncoded

uncoded_bits = uncoded_bits';
uncoded_bits = uncoded_bits(:)';
uncoded_bits = uncoded_bits(1:length(bits));  % Truncate any padding


img_size = sqrt(length(uncoded_bits));
estimated_img = reshape(uncoded_bits, [img_size, img_size]); 
resized_estimated_img = imresize(estimated_img, 1 / imresize_scale);

figure;
subplot(1, 3, 1);
imshow(resized_estimated_img);
title('Uncoded Image');


[~, BER_uncoded] = biterr(bits, uncoded_bits);
disp('BER for Uncoded:');
disp(BER_uncoded);

%% Hard Decision



% Reshape decoded bits back to the original size
HD_decoded_bits = HD_decoded_bits';
HD_decoded_bits = HD_decoded_bits(:)';
HD_decoded_bits = HD_decoded_bits(1:length(bits));  % Truncate any padding



estimated_img_HD = reshape(HD_decoded_bits, [img_size, img_size]); 
resized_estimated_img_HD = imresize(estimated_img_HD, 1 / imresize_scale);

subplot(1, 3, 2);

imshow(resized_estimated_img_HD);
title('Hard Decision Decoding');


[~, BER_HD_decoded_bits] = biterr(bits, HD_decoded_bits);
disp('BER for Hard Decision Decoding:');
disp(BER_HD_decoded_bits);

%% Soft decision w/ noise

% Define all possible Hamming (7,4) codewords (for soft decision decoding)
valid_codewords = mod(de2bi(0:15, k, 'left-msb') * G, 2); % All possible 7-bit codewords
codebook = 2 * valid_codewords - 1; % Map 0 -> -1, 1 -> 1 for Euclidean distance calculation

% Step 5: Soft Decision Decoding
soft_decoded_bits = zeros(size(reshaped_bits));  % Initialize for storing decoded bits

for i = 1:size(received_bits, 1)
    % Extract the received 7-bit segment (noisy symbols) for this codeword
    received_segment = received_symbols((i-1)*n + 1:i*n);
    
    % Calculate Euclidean distances between received segment and all valid codewords
    distances = sum((codebook - received_segment').^2, 2);
    
    % Find the index of the closest codeword
    [~, min_index] = min(distances);
    
    % Decode by selecting the corresponding data bits from the closest codeword
    soft_decoded_bits(i, :) = valid_codewords(min_index, 1:k);
end

% Reshape decoded bits back to the original size
soft_decoded_bits = soft_decoded_bits';
soft_decoded_bits = soft_decoded_bits(:)';
soft_decoded_bits = soft_decoded_bits(1:length(bits));  % Truncate any padding

subplot(1, 3, 3);

estimated_img_SD = reshape(soft_decoded_bits, [img_size, img_size]); 
resized_estimated_img_SD = imresize(estimated_img_SD, 1 / imresize_scale);
imshow(resized_estimated_img_SD);
title('Soft Decision Decoding');


% Calculate BER for Soft Decision
[~, BER_SD_decoded_bits] = biterr(bits, soft_decoded_bits);
disp('BER for Soft Decision Decoding:');
disp(BER_SD_decoded_bits);