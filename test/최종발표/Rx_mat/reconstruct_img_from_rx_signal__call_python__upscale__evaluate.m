clc
clear
close all

%% Part 1: Image Reconstruction from Binary Data

% Read the binary file
load('C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\Rx_signal.mat');

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
imwrite(uint8(quantized_img), 'Lena_restored_kmeans_a.png');

% Display initial image
subplot(1, 3, 1);
imshow(uint8(quantized_img));
title('Initial K-Means Quantized Image');

%% Part 2: Call Python for Upscaling
% 현재 MATLAB 작업 디렉토리 저장
current_dir = pwd;
%% Python 환경 설정 부분 (스크립트 시작 부분에 추가)
% 아나콘다 resolution 환경의 Python 실행 경로 설정
pyenv('Version', 'C:\Users\okmun\anaconda3\envs\resolution\python.exe');

%% 기존 코드에서 Python 호출 부분을 다음과 같이 수정
% upscale.py를 통한 이미지 처리
cd('C:\Users\okmun\image_upscaler');  % 작업 디렉토리 변경
system('C:\Users\okmun\anaconda3\envs\resolution\python.exe upscale.py');

% 파일이 저장될 때까지 대기
pause(2);

% 업스케일된 이미지를 MATLAB 작업 디렉토리로 복사
copyfile('upscaled_Lena.png', fullfile(current_dir, 'upscaled_Lena.png'));

% 원래 MATLAB 작업 디렉토리로 복귀
cd(current_dir);

%% Part 3: Image Quality Assessment

% Load upscaled image
img = imread('upscaled_Lena.png');
resized_img = imresize(img, [224, 224]);
gray_img = rgb2gray(resized_img);
binaryImg = imbinarize(gray_img);

% Load comparison images
imageFiles = {'Lena_color.png'};
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
title(sprintf('Reference Image\nScore: %.1f점\nBER: %.6f', similarityScores(bestMatchIndex), ber));

% Save results
save('kmeans_decimal_data.mat', 'indices_dec', 'centroids_dec');