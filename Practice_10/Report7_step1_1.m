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

% Parity-check matrix H for decoding
H = [A', eye(c)];

% Step 2: Input 8 bits and split into two 4-bit blocks
bits = [0 0 0 0, 0 0 0 1];  % Example 8-bit input
reshaped_bits = reshape(bits, [k, length(bits)/k])';

% Step 3: Encode each 4-bit block using Hamming (7,4)
hamming_coded_bits = mod(reshaped_bits * G, length(bits)/k);

% Step 4: Simulate transmission (BPSK Modulation)
channel_coded_bits = hamming_coded_bits';
transmit_symbols = 2 * channel_coded_bits(:) - 1;  % BPSK modulation

% Receiver (with simulated noise - optional, remove if unnecessary)
demodulated_bits = (sign(transmit_symbols) + 1) / 2; % BPSK Demodulation

% Step 5: Reshape received bits for decoding
received_bits = reshape(demodulated_bits, [n, 2])';

% Step 6: Hard Decision Decoding (error correction using syndrome)
syndrome_matrix = mod(received_bits * H', 2);
HD_decoded_bits = zeros(2, k);  % Initialize for storing decoded bits

for i = 1:2
    % Convert syndrome to decimal to identify error position
    syndrome_decimal = bi2de(syndrome_matrix(i, :), 'left-msb');
    if syndrome_decimal ~= 0
        % If an error is detected, correct it
        received_bits(i, syndrome_decimal) = ~received_bits(i, syndrome_decimal);
    end
    % Extract the corrected data bits (first 4 bits)
    HD_decoded_bits(i, :) = received_bits(i, 1:k);
end

HD_decoded_bits = reshape(HD_decoded_bits, [1,length(bits)]);

% Display results
disp('Original Data Bits:');
disp(reshaped_bits);
disp('Hamming Encoded Bits:');
disp(hamming_coded_bits);
disp('Received Bits (after demodulation):');
disp(received_bits);
disp('Decoded Data Bits (after error correction):');
disp(HD_decoded_bits);
