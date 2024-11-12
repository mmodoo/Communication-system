clc;
close all;

% Parameter setting
% Sampling frequency
sampling_freq = 10000;
% N, number of sub-carriers
N = 256;
% Cyclic prefix length
N_cp = N / 4;
% Preamble length
Tp = 1000;
% Image path
img_path = 'C:\Users\Donghwi\Desktop\DoHW\24-2\통설\Report4';

disp('################################### Creating tx_signal.');

% Step 1: Load and process image
img = imread(fullfile(img_path, 'Lena_color.png'));
img_resize_scale_rate = 0.5;
resized_img = imresize(img, img_resize_scale_rate);
gray_img = rgb2gray(resized_img);
binarised_img = imbinarize(gray_img);
bits = binarised_img(:);

% Step 2: Apply repetition coding
repetition_factor = 3;
channel_coded_bits = repelem(bits, repetition_factor);

% Step 3: BPSK Modulation
symbols = 2 * channel_coded_bits - 1;

% Step 4: Calculate symbol and block lengths
M = length(symbols);
cn = M / (N / 2);
N_blk = cn + cn / 4;

% Step 5: Serial-to-Parallel Conversion
symbols_freq = {};
for i = 1:cn
    symbols_freq{end + 1} = [0; symbols(N/2*(i-1)+1 : N/2*i)];
    symbols_freq{end} = [symbols_freq{end}; flip(symbols_freq{end}(2 : end-1))];
end

% Step 6: Inverse Discrete Fourier Transform (IDFT)
symbols_time = {};
for i = 1:length(symbols_freq)
    symbols_time{end + 1} = ifft(symbols_freq{i}, N) * sqrt(N);
end

% Step 7: Add Cyclic Prefix
for i = 1 : length(symbols_time)
    symbols_time{i} = [symbols_time{i}(end - N_cp + 1 : end); symbols_time{i}];
end

% Step 8: Add Pilot Signal
pilot_freq = ones(N, 1);
pilot_time = ifft(pilot_freq) * sqrt(N);
pilot_time = [pilot_time(end - N_cp + 1 : end); pilot_time];

% Step 9: Preamble
omega = 10;
mu = 0.1;
tp = (1:Tp).';
preamble = cos(omega * tp + mu * tp.^2 / 2);

% Step 10: Convert to Serial for Transmission
tx_signal = [preamble; pilot_time];
for i = 1 : length(symbols_time)
    tx_signal = [tx_signal; symbols_time{i}];
    if rem(i, 4) == 0 && i ~= length(symbols_time)
        tx_signal = [tx_signal; pilot_time];
    end
end

disp("############ Tx ready.");

% Audio Transmission
player = audioplayer(tx_signal, sampling_freq);    
start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
disp(['Transmission start time: ', char(start_time)]);
play(player);

disp('Playing sound...');
while isplaying(player)
    pause(0.5);
    fprintf('.');
end
end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
disp('.');
disp(['Transmission end time: ', char(end_time)]);
elapsed_time = end_time - start_time;
disp(['Total elapsed time [s]: ', char(elapsed_time)]);

% Deinterleaving and Decoding
% Set random seed for reproducibility
rng('default');
deintrlv_bits = deintrlv(decoded_bits, randperm(length(decoded_bits)));

% Channel Decoding
bits_reshape = reshape(deintrlv_bits, [3, length(deintrlv_bits)/3]).';
bits_sum = bits_reshape * ones(3,1);
channel_decoded_bits = ge(bits_sum, 2);

% Transmission
sound(tx_signal, sampling_freq);

% Plot Tx Signal
figure;
plot(0:length(tx_signal)-1, tx_signal, 'LineWidth', 1.5);
xlim([0 length(tx_signal)-1]);
ylim([-20 20]);
xlabel('Time');
ylabel('Amplitude');
grid on;

% Calculate and display PAPR
max_pow = max(abs(tx_signal));
avg_pow = mean(abs(tx_signal));
PAPR = max_pow / avg_pow;
disp(['PAPR: ', num2str(PAPR)]);
