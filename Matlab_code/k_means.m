clc;
clear;
close all;

% Step 1: Read and preprocess image
original_img = imread('Lena_color.png');
[h, w, ~] = size(original_img);

% Step 2: Resize image to be divisible by block size
block_size = 4;
h_trimmed = floor(h / block_size) * block_size;
w_trimmed = floor(w / block_size) * block_size;
img_trimmed = original_img(1:h_trimmed, 1:w_trimmed, :);

% Step 3: Divide image into blocks and reshape
num_blocks = (h_trimmed / block_size) * (w_trimmed / block_size);
blocks = zeros(num_blocks, block_size * block_size * 3); % Store blocks as rows

block_idx = 1;
for i = 1:block_size:h_trimmed
    for j = 1:block_size:w_trimmed
        block = img_trimmed(i:i+block_size-1, j:j+block_size-1, :);
        blocks(block_idx, :) = reshape(block, 1, []); % Flatten each block
        block_idx = block_idx + 1;
    end
end

% Step 4: Apply K-Means clustering
K = 16; % Number of clusters (adjust for quality/compression tradeoff)
[indices, centroids] = kmeans(double(blocks), K);

% Step 5: Quantize image
quantized_indices = reshape(indices, h_trimmed / block_size, w_trimmed / block_size);

% Step 6: Decode the quantized image
reconstructed_img = zeros(h_trimmed, w_trimmed, 3);
block_idx = 1;
for i = 1:block_size:h_trimmed
    for j = 1:block_size:w_trimmed
        % Retrieve the centroid for the block
        centroid = reshape(centroids(quantized_indices(block_idx), :), block_size, block_size, 3);
        reconstructed_img(i:i+block_size-1, j:j+block_size-1, :) = centroid;
        block_idx = block_idx + 1;
    end
end

% Step 7: Display and save results
imwrite(uint8(reconstructed_img), 'Lena_color_vq.png');

figure;
subplot(1, 2, 1); imshow(original_img); title('Original Image');
subplot(1, 2, 2); imshow(uint8(reconstructed_img)); title('VQ Compressed Image');

% Step 8: Calculate Compression Ratio
original_size = numel(original_img); % Size in bytes
quantized_size = numel(quantized_indices) + numel(centroids) * 3; % Indices + centroids
compression_ratio = (1 - quantized_size / original_size) * 100;
fprintf('Compression ratio: %.2f%%\n', compression_ratio);
