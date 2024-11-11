clc;
clear;

% Step 1: Define parameters
k = 4; % Number of data bits in each coded word
c = 3; % Number of parity bits in each coded word
n = k + c; % Total number of bits in each coded word

% Parity generator matrix for Hamming (7,4) code
A = [1 1 1;
     1 1 0;
     1 0 1;
     0 1 1];

% Code generator matrix G
G = [eye(k), A];

% Define all possible Hamming (7,4) codewords (for soft decision decoding)
valid_codewords = mod(de2bi(0:15, k, 'left-msb') * G, 2); % All possible 7-bit codewords
codebook = 2 * valid_codewords - 1; % Map 0 -> -1, 1 -> 1 for Euclidean distance calculation

% Step 2: Input 8 bits and split into two 4-bit blocks
bits = [0 0 0 0 0 0 0 1];  % Example 8-bit input
reshaped_bits = reshape(bits, [k, 2])';

% Step 3: Encode each 4-bit block using Hamming (7,4)
hamming_coded_bits = mod(reshaped_bits * G, 2);

% Step 4: Simulate transmission (BPSK Modulation)
channel_coded_bits = hamming_coded_bits';
transmit_symbols = 2 * channel_coded_bits(:) - 1;  % BPSK modulation

% % % Add noise (optional)
% % noise = 0.5 * randn(size(transmit_symbols)); % Adjust noise level as needed
% % noisy_symbols = transmit_symbols + noise;

% Step 5: Soft Decision Decoding
soft_decoded_bits = zeros(2, k);  % Initialize for storing decoded bits

for i = 1:2
    % Extract the received 7-bit segment (noisy symbols) for this codeword
    received_segment = transmit_symbols((i-1)*n + 1:i*n);
    
    % Calculate Euclidean distances between received segment and all valid codewords
    distances = sum((codebook - received_segment').^2, 2);
    
    % Find the index of the closest codeword
    [~, min_index] = min(distances);
    
    % Decode by selecting the corresponding data bits from the closest codeword
    soft_decoded_bits(i, :) = valid_codewords(min_index, 1:k);
end

SD_decoded_bits = reshape(soft_decoded_bits, [1,length(bits)]);


% Display results
disp('Original Data Bits:');
disp(bits);
disp('Hamming Encoded Bits:');
disp(hamming_coded_bits);
% disp('Received Bits (noisy symbols):');
% disp(reshape(noisy_symbols, [n, 2])');
disp('Decoded Data Bits (soft decision):');
disp(SD_decoded_bits);
