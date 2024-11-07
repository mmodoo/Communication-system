clc;
clear;
close all;

% 1. 바이너리 이미지 로드 및 이진화
% 이미지 로드 및 전처리
img = imread('Lena_color.png'); % 입력 이미지 경로
resized_img = imresize(img, [640 640]);

% disp('image를 gray-scale로 바꿉니다.');
gray_img = rgb2gray(resized_img);

% disp('image를 monochrome으로 바꿉니다.');
binaryImg = imbinarize(gray_img);

% 2. 비교할 컬러 이미지 목록 로드
imageFiles = {'test (2).jpg', 'test.jpg', 'Lena_color.png'}; % 비교할 이미지 파일 이름 리스트
numImages = length(imageFiles);

% 유사도 계산 결과 저장할 변수들
ssimValues = zeros(1, numImages);
psnrValues = zeros(1, numImages);
similarityScores = zeros(1, numImages);

% 가중치 설정 (가중치는 SSIM과 PSNR 합이 1이 되도록 설정)
ssimWeight = 0.5;
psnrWeight = 0.5;

% 3. 각 이미지에 대해 SSIM 및 PSNR 계산
for i = 1:numImages
    % 컬러 이미지 로드
    colorImg = imread(imageFiles{i});
    colorImg = imresize(colorImg, [640, 640]); % 크기 조정

    % 바이너리 이미지를 3채널로 변환하여 비교 (컬러 이미지와 동일한 형식으로 맞춤)
    binaryImg3Channel = uint8(repmat(binaryImg, [1, 1, 3]) * 255);

    % SSIM 및 PSNR 계산
    ssimValues(i) = ssim(colorImg, binaryImg3Channel);
    psnrValues(i) = psnr(colorImg, binaryImg3Channel);

    % SSIM과 PSNR의 가중 합 계산
    similarityScores(i) = ssimWeight * ssimValues(i) + psnrWeight * (psnrValues(i) / 100); % PSNR은 0-1로 정규화
    fprintf('Image %s - SSIM: %.4f, PSNR: %.2f, Combined Score: %.4f\n', imageFiles{i}, ssimValues(i), psnrValues(i), similarityScores(i));
end

% 4. 가장 유사한 이미지 선택
[~, bestMatchIndex] = max(similarityScores); % 가장 높은 유사도 인덱스 추출
bestMatchImage = imread(imageFiles{bestMatchIndex}); % 최종 선택된 이미지 로드

% 5. 결과 시각화
figure;
subplot(1, numImages+1, 1);
imshow(binaryImg);
title('Binary Image');

for i = 1:numImages
    subplot(1, numImages+1, i+1);
    imshow(imread(imageFiles{i}));
    title(sprintf('SSIM: %.4f\nPSNR: %.2f\nScore: %.4f', ssimValues(i), psnrValues(i), similarityScores(i)));
end

figure;
imshow(bestMatchImage);
title(sprintf('Best Match Image: %s\nScore: %.4f', imageFiles{bestMatchIndex}, similarityScores(bestMatchIndex)));


resized_bestMatchImage = imresize(bestMatchImage, [640 640]);

% disp('image를 gray-scale로 바꿉니다.');
gray_resized_bestMatchImage = rgb2gray(resized_bestMatchImage);

% disp('image를 monochrome으로 바꿉니다.');
binary_bestMatchImage = imbinarize(gray_resized_bestMatchImage);

bits = binaryImg(:);
bestMatch_bits = binary_bestMatchImage(:);

[~, BER] = biterr(bits, bestMatch_bits);
disp(BER);
