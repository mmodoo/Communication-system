clc;
clear;
close all;

% 1. 바이너리 이미지 로드 및 이진화
% 이미지 로드 및 전처리
img = imread('Lena_restored_kmeans_a_upscayl_2x_realesrgan-x4plus-anime.png'); % 입력 이미지 경로
resized_img = imresize(img, [224, 224]);
gray_img = rgb2gray(resized_img);
binaryImg = imbinarize(gray_img);

% 2. 비교할 컬러 이미지 목록 로드
imageFiles = {'test1.jpg', 'test1_256X256.png', 'Lena_color.png'}; % 비교할 이미지 파일 이름 리스트
numImages = length(imageFiles);

% 유사도 계산 결과 저장할 변수들
ssimValues = zeros(1, numImages);
psnrValues = zeros(1, numImages);
cnnSimilarities = zeros(1, numImages);
similarityScores = zeros(1, numImages);

% 가중치 설정 (SSIM, PSNR, CNN 유사도의 합이 1이 되도록 설정)
ssimWeight = 0.8;
psnrWeight = 0.1;
cnnWeight = 0.1;

% 3. 사전 학습된 CNN 모델 로드
net = resnet50;
binaryFeatures = activations(net, resized_img, 'fc1000', 'OutputAs', 'rows'); % CNN 특징 벡터 추출

% 4. 각 이미지에 대해 SSIM, PSNR 및 CNN 유사도 계산
for i = 1:numImages
    % 컬러 이미지 로드 및 전처리
    colorImg = imread(imageFiles{i});
    colorImg = imresize(colorImg, [224, 224]);

    % SSIM 및 PSNR 계산
    ssimValues(i) = ssim(colorImg, resized_img);
    psnrValues(i) = psnr(colorImg, resized_img);

    % CNN 특징 벡터 추출 및 코사인 유사도 계산
    colorFeatures = activations(net, colorImg, 'fc1000', 'OutputAs', 'rows');
    cosineSimilarity = dot(binaryFeatures, colorFeatures) / (norm(binaryFeatures) * norm(colorFeatures));
    cnnSimilarities(i) = cosineSimilarity;

    % 최종 유사도 점수 계산
    similarityScores(i) = ssimWeight * ssimValues(i) + psnrWeight * (psnrValues(i) / 100) + cnnWeight * cnnSimilarities(i);
    similarityScores(i) = similarityScores(i)*100
    fprintf('Image %s - SSIM: %.4f, PSNR: %.2f, CNN Similarity: %.4f, Combined Score: %.4f\n', ...
            imageFiles{i}, ssimValues(i), psnrValues(i), cnnSimilarities(i), similarityScores(i));
end

% 5. 가장 유사한 이미지 선택
[~, bestMatchIndex] = max(similarityScores); % 가장 높은 유사도 인덱스 추출
bestMatchImage = imread(imageFiles{bestMatchIndex}); % 최종 선택된 이미지 로드
bestMatchImage = imresize(bestMatchImage, [224, 224]); % 크기 맞추기

% 6. BER 계산
% 이진화된 두 이미지를 비교
resized_img_binary = imbinarize(rgb2gray(resized_img));
bestMatchImage_binary = imbinarize(rgb2gray(bestMatchImage));

% BER 계산
totalBits = numel(resized_img_binary);
bitErrors = sum(resized_img_binary(:) ~= bestMatchImage_binary(:));
ber = bitErrors / totalBits;

fprintf('Bit Error Rate (BER) between resized image and best match: %.6f\n', ber);

% 7. 결과 시각화
figure;
subplot(1, numImages+1, 1);
imshow(resized_img);
title('Compressed Image');

for i = 1:numImages
    subplot(1, numImages+1, i+1);
    imshow(imread(imageFiles{i}));
    title(sprintf('SSIM: %.4f\nPSNR: %.2f\nCNN: %.4f\nScore: %.1f점', ...
                  ssimValues(i), psnrValues(i), cnnSimilarities(i), similarityScores(i)));
end

figure;
imshow(bestMatchImage);
title(sprintf('Best Match Image: %s\nScore: %.4f\nBER: %.6f', ...
              imageFiles{bestMatchIndex}, similarityScores(bestMatchIndex), ber));
