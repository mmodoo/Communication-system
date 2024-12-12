% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 1;
% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 2;
Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 3;
% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 4;


clearvars -except Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening;
clc;
close all;

Original_MAT_path_and_file_name = "kmeans_logical_data.mat";
Tx_Setting_MAT_path_and_file_name = "Tx_Setting.mat";
Tx_signal_MAT_path_and_file_name = "Tx_signal.mat";
Tx_signal_WAV_path_and_file_name = "Tx_signal.wav";
Rx_signal_MAT_path_and_file_name = "Rx_signal.mat";
test_image = 'test3_256X256.png';

load(Tx_Setting_MAT_path_and_file_name);

buffer_for_Recording_Time_Sec = 5;
Recording_Time_Sec = Recording_Time_Sec + buffer_for_Recording_Time_Sec

if Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 1
    load(Tx_signal_MAT_path_and_file_name);
elseif Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 2
    Tx_signal = audioread(Tx_signal_WAV_path_and_file_name);
elseif Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 3
    try
            release(devicereader);
        catch ME
            switch ME.identifier  % 에러 타입에 따라 다른 처리
                case 'MATLAB:UndefinedFunction'
                    fprintf('devicereader가 정의되지 않았습니다.\n');
                case 'MATLAB:class:InvalidHandle'
                    fprintf('devicereader가 유효하지 않은 핸들입니다.\n');
                otherwise
                    fprintf('예상치 못한 오류 발생: %s\n', ME.message);
            end
            
            % 에러 로그 기록이나 추가 처리가 필요한 경우
            % disp(ME.stack);  % 에러가 발생한 위치 추적
    end

    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['# Rx 시작 시각: ', char(start_time)]);
    disp('# Rx을 실행합니다.');
    devicereader = audioDeviceReader(Sampling_Freq);
    setup(devicereader);
    disp(['총 녹음 시간: ', num2str(Recording_Time_Sec), '초']);
    Tx_signal = [];
    elapsed = 0;
    tic;
    
    while toc < Recording_Time_Sec  % recording_time_sec 동안 녹음
        acquiredAudio = devicereader();
        Tx_signal = [Tx_signal; acquiredAudio];
        
        % 1초마다 진행 상태 업데이트
        if floor(toc) > elapsed
            elapsed = floor(toc);
            fprintf('\r녹음 진행: %d초 / %d초 (%.1f%%)', ...
                elapsed, ceil(Recording_Time_Sec), (elapsed/Recording_Time_Sec)*100);
        end
    end
    fprintf('\n');  % 줄바꿈 추가
    disp('Recording Completed');
end
  

if Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 1 ...
        || Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 2 ...
        || Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening == 3

    [xC, lags] = xcorr(Tx_signal, Preamble);
    [~, idx] = max(xC);
    start_pt = lags(idx);

    Tx_signal = Tx_signal(start_pt + length(Preamble) + 1:end);

    % Serial to Parallel
    OFDM_blks = {};
    for i = 1:Total_OFDM_Symbol_Number_including_Pilot
        OFDM_blks{end + 1} = Tx_signal(N_cp__that_is__Cyclic_Prefix_Length + 1 : (N + 2) + N_cp__that_is__Cyclic_Prefix_Length);
        Tx_signal = Tx_signal(N_cp__that_is__Cyclic_Prefix_Length + (N + 2) + 1 : end);
    end

    % Discrete Fourier Transform (DFT)
    demode_OFDM_blks = {};
    for i = 1 : length(OFDM_blks)
        demode_OFDM_blks{end + 1} = fft(OFDM_blks{i} / sqrt(N + 2));
    end

    % Channel Estimation & Equalisation. Pilot 여기서 활용됨
    symbols_eq = {};
    rng('default');
    temp_pilot_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
    temp_pilot_2 = 2 * (randi([0, 1], floor(N/Subcarrier_Freq_Divided_by)/2, 1) - 0.5);
    temp_pilot_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
    temp_pilot = [temp_pilot_1; temp_pilot_2; temp_pilot_3];
    pilot_freq = [0; temp_pilot; 0; flip(conj(temp_pilot))];  % DC, 중간, 허미션 처리
    for i = 1 : length(demode_OFDM_blks)
        if rem(i, 5) == 1
            channel = demode_OFDM_blks{i} ./ pilot_freq;
        else
            symbols_eq{end + 1} = demode_OFDM_blks{i} ./ channel;
        end
    end

    % Detection
    symbols_detect = {};
    for i = 1 : length(symbols_eq)
        % symbols_detect{end + 1} = sign(real(symbols_eq{i}));
        symbols_detect{end + 1} = symbols_eq{i};
    end
    
    % Demodulation
    symbols_est = [];
    for i = 1 : length(symbols_detect)
        % symbols_est = [symbols_est; symbols_detect{i}(2 + (Subcarrier_Freq_Divided_by-1)*N / (2*Subcarrier_Freq_Divided_by) : N / 2 + 1)];
        symbols_est = [symbols_est; symbols_detect{i}(2+floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2:1+floor(N*(1-1*(1/Subcarrier_Freq_Divided_by)))/2)];
    end
    
    % QPSK Demodulation
    % decoded_bits = (symbols_est + 1) / 2;
    decoded_symbols = pskdemod(symbols_est, 4, pi/4);
    decoded_bits = int2bit(decoded_symbols, 2);
    
    % 동일한 interleaver_order 생성
    rng('default'); % 동일한 시드로 초기화하여 Tx와 같은 순서 생성
    interleaver_order = randperm(length(decoded_bits));
    
    % Disinterleaving
    [~, deinterleaver_order] = sort(interleaver_order); % 원래 순서를 찾기 위해 역순서를 계산
    decoded_bits = decoded_bits(deinterleaver_order); % 원래 순서로 복원

    if Whether_NOT_Repetition_coding__OR__Repetition_How_Many ~= 1
        % Repetition Decoding
        % 3개씩 묶어서 reshape
        decoded_bits_reshaped = reshape(decoded_bits, Whether_NOT_Repetition_coding__OR__Repetition_How_Many, []);
        
        % 각 열의 합을 계산 (다수결 원칙 적용)
        sums = sum(decoded_bits_reshaped);
        
        % 합이 2 이상이면 1, 미만이면 0으로 판정
        decoded_bits = (sums > (Whether_NOT_Repetition_coding__OR__Repetition_How_Many / 2)).';
    end

    decoded_bits = decoded_bits.';
end

save(Rx_signal_MAT_path_and_file_name, 'decoded_bits');

load('kmeans_logical_data.mat');

different_bits = (combined_logical_data ~= decoded_bits);

different_bits_Number = sum(different_bits)

BER = different_bits_Number / length(combined_logical_data)

%% Part 1: Image Reconstruction from Binary Data

% Read the binary file
load('Rx_signal.mat');

% Split bits into Indices and Centroids
indices_bits = decoded_bits(1:98304);
centroids_bits = decoded_bits(98305:99840);

% Convert bits to indices
indices_bits_mat = reshape(indices_bits, 6, []).';
indices_bits_mat = double(indices_bits_mat);
weights = [32; 16; 8; 4; 2; 1];
indices_dec = indices_bits_mat * weights;
indices_dec = indices_dec + 1;

% Convert bits to centroids
function decimal_values = bits_to_decimal(bits)
    if mod(length(bits), 8) ~= 0
        error('비트 수는 8의 배수여야 합니다.');
    end
    bytes_matrix = reshape(bits, 8, []).';
    bytes_matrix = double(bytes_matrix);
    weights = [128; 64; 32; 16; 8; 4; 2; 1];
    decimal_values = bytes_matrix * weights;
end

centroids_dec = bits_to_decimal(centroids_bits);
centroids_dec = reshape(centroids_dec,[64 3]);

% Reconstruct and save the image
quantized_img = reshape(centroids_dec(indices_dec, :), 128, 128, 3);
quantized_img = uint8(quantized_img);
imwrite(uint8(quantized_img), 'image_compressed.png');

% Display initial image
subplot(1, 3, 1);
imshow(uint8(quantized_img));
title('Initial K-Means Quantized Image');

%% Part 2: Call Python for Upscaling
% 현재 MATLAB 작업 디렉토리 저장
current_dir = pwd;
%% Python 환경 설정 부분 (스크립트 시작 부분에 추가)
% 아나콘다 resolution 환경의 Python 실행 경로 설정
pyenv('Version', 'C:\Users\Donghwi\anaconda3\envs\myenv\python.exe');

%% 기존 코드에서 Python 호출 부분을 다음과 같이 수정
% upscale.py를 통한 이미지 처리
cd('C:\Users\Donghwi\Desktop\image_upscaler');  % 작업 디렉토리 변경
imwrite(uint8(quantized_img), 'image_compressed.png');

system('C:\Users\Donghwi\anaconda3\envs\myenv\python.exe upscale.py');

% 파일이 저장될 때까지 대기
pause(2);

% 업스케일된 이미지를 MATLAB 작업 디렉토리로 복사
copyfile('upscaled_image.png', fullfile(current_dir, 'upscaled_image.png'));

% 원래 MATLAB 작업 디렉토리로 복귀
cd(current_dir);

%% Part 3: Image Quality Assessment

% Load upscaled image
img = imread('upscaled_image.png');
resized_img = imresize(img, [224, 224]);
gray_img = rgb2gray(resized_img);
binaryImg = imbinarize(gray_img);

% Load comparison images
imageFiles = {test_image};
numImages = length(imageFiles);

% Initialize variables
ssimValues = zeros(1, numImages);
psnrValues = zeros(1, numImages);
cnnSimilarities = zeros(1, numImages);
similarityScores = zeros(1, numImages);

% Set weights
ssimWeight = 0.8;
psnrWeight = 0.1;
cnnWeight = 0.1;

% Load pre-trained CNN
net = resnet50;
binaryFeatures = activations(net, resized_img, 'fc1000', 'OutputAs', 'rows');

% Calculate similarities
for i = 1:numImages
    colorImg = imread(imageFiles{i});
    colorImg = imresize(colorImg, [224, 224]);
    
    ssimValues(i) = ssim(colorImg, resized_img);
    psnrValues(i) = psnr(colorImg, resized_img);
    
    colorFeatures = activations(net, colorImg, 'fc1000', 'OutputAs', 'rows');
    cosineSimilarity = dot(binaryFeatures, colorFeatures) / (norm(binaryFeatures) * norm(colorFeatures));
    cnnSimilarities(i) = cosineSimilarity;
    
    similarityScores(i) = ssimWeight * ssimValues(i) + psnrWeight * (psnrValues(i) / 100) + cnnWeight * cnnSimilarities(i);
    similarityScores(i) = similarityScores(i) * 100;
    
    fprintf('Image %s - SSIM: %.4f, PSNR: %.2f, CNN Similarity: %.4f, Combined Score: %.4f\n', ...
            imageFiles{i}, ssimValues(i), psnrValues(i), cnnSimilarities(i), similarityScores(i));
end

% Find best match
[~, bestMatchIndex] = max(similarityScores);
bestMatchImage = imread(imageFiles{bestMatchIndex});
bestMatchImage = imresize(bestMatchImage, [224, 224]);

% Calculate BER
resized_img_binary = imbinarize(rgb2gray(resized_img));
bestMatchImage_binary = imbinarize(rgb2gray(bestMatchImage));
totalBits = numel(resized_img_binary);
bitErrors = sum(resized_img_binary(:) ~= bestMatchImage_binary(:));
ber = bitErrors / totalBits;

fprintf('Bit Error Rate (BER): %.6f\n', ber);

% Display results
subplot(1, 3, 2);
imshow(resized_img);
title('Upscaled Image');

subplot(1, 3, 3);
imshow(bestMatchImage);
title(sprintf('SSIM: %.4f\nPSNR: %.2f\nCNN: %.4f\nScore: %.1f점\nBER: %.6f', ...
                  ssimValues(bestMatchIndex), psnrValues(bestMatchIndex), cnnSimilarities(bestMatchIndex), similarityScores(bestMatchIndex),ber));
% title(sprintf('Reference Image\nScore: %.1f점\nBER: %.6f', similarityScores(bestMatchIndex), ber));

% Save results
save('kmeans_decimal_data.mat', 'indices_dec', 'centroids_dec');