clc;
clear;


% Source Encoding
imresize_scale = 0.5; % The scale of resize of the image
img = imread('Lena_color.png'); % Load image
resized_img = imresize(img, imresize_scale); % Resize the image via bicubic interpolation
gray_img = rgb2gray(resized_img); % Color image to grayscale image via NTSC
binarized_img = imbinarize(gray_img); % Binarize the grayscale image via Otsu algorithm
bits = binarized_img(:); 

%% Uncoded w/ noise
% Transmitter
transmit_symbols = 2 * bits - 1;

% Channel (Noise)
noise = randn(length(transmit_symbols), 1);
received_symbols = transmit_symbols + noise;

% Receiver
demodulated_bits = (sign(received_symbols) + 1) / 2;
[~, uncoded_BER] = biterr(bits, demodulated_bits)

%% Hard Decision w/ noise

bits = bits';

% Transmitter
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

transmit_symbols = 2 * channel_coded_bits - 1; % BPSK Modulation

% Channel (Noise)
noise = randn(length(transmit_symbols), 1);
received_symbols = transmit_symbols + noise;

% Receiver (with simulated noise - optional, remove if unnecessary)
demodulated_bits = (sign(received_symbols) + 1) / 2; % BPSK Demodulation

% Step 5: Reshape received bits for decoding
received_bits = reshape(demodulated_bits, [n, length(demodulated_bits)/n])';

% Step 6: Hard Decision Decoding (error correction using syndrome)
syndrome_matrix = mod(received_bits * H', 2);
HD_decoded_bits = zeros(size(reshaped_bits));  % Initialize for storing decoded bits

for i = 1:size(syndrome_matrix, 1)
    % Convert syndrome to decimal to identify error position
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

% Reshape decoded bits back to the original size
HD_decoded_bits = HD_decoded_bits';
HD_decoded_bits = HD_decoded_bits(:)';
HD_decoded_bits = HD_decoded_bits(1:length(bits));  % Truncate any padding

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
SD_decoded_bits = soft_decoded_bits';
SD_decoded_bits = SD_decoded_bits(:)';
SD_decoded_bits = SD_decoded_bits(1:length(bits));  % Truncate any padding

% Calculate BER for Soft Decision
[~, BER_SD_decoded_bits] = biterr(bits, SD_decoded_bits);
disp('BER for Soft Decision Decoding:');
disp(BER_SD_decoded_bits);