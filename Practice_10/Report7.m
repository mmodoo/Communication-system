clc;
clear;
%% step1-1
bits = [0 0 0 0, 0 0 0 1, 0 0 1 0, 0 0 1 1, ...
        0 1 0 0, 0 1 0 1, 0 1 1 0, 0 1 1 1, ...
        1 0 0 0, 1 0 0 1, 1 0 1 0, 1 0 1 1, ...
        1 1 0 0, 1 1 0 1, 1 1 1 0, 1 1 1 1];

% Transmitter
k = 4; % Number of valid bits in 1 coded bits
c = 3; % Number of parities in 1 coded bits
n = k + c; % Number of total bits in 1 coded bits
A = [1 1 1;
     1 1 0;
     1 0 1;
     0 1 1]; % Parity generator matrix
G = [eye(k), A]; % Code generator matrix

reshaped_bits = reshape(bits, [k, length(bits)/k]);

hamming_coded_bits = [reshaped_bits' * G]