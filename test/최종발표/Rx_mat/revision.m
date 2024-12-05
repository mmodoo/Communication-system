clc
clear

%% Verification (Optional)

% To verify, let's read the binary file and reconstruct 'indices' and 'centroids'

% Read the binary file
load('Rx_signal (1)');

% Step 13: 'indices'와 'centroids'를 8비트씩 묶어 10진수로 변환

% 가정: combined_logical_data는 1x132608 논리 배열
% combined_logical_data는 이전 단계에서 생성된 것으로 가정

% 예시: combined_logical_data 생성 (실제 데이터로 대체)
% combined_logical_data = [indices_bits, centroids_bits]; % 1x132608 논리 배열

% **주의**: 실제 combined_logical_data를 사용해야 합니다.

% Step 13.1: 비트 배열을 Indices와 Centroids로 분할
indices_bits = decoded_bits(1:98304);       % 비트 1부터 131072까지
centroids_bits = decoded_bits(98305:99840); % 비트 131073부터 132608까지

% Step 13.2: 8비트씩 그룹화하여 10진수로 변환하는 함수 정의
function decimal_values = bits_to_decimal(bits)
    % bits: 1xN 논리 배열
    % N은 8의 배수여야 함

    % N이 8의 배수인지 확인
    if mod(length(bits), 8) ~= 0
        error('비트 수는 8의 배수여야 합니다.');
    end

    % 8비트씩 그룹화
    % reshape을 사용하여 8행 N/8열 행렬로 변환 후 전치
    bytes_matrix = reshape(bits, 8, []).'; % Mx8 행렬 (M = N/8)

    % 논리 값을 더블 형으로 변환
    bytes_matrix = double(bytes_matrix);

    % 8비트에서 10진수로 변환 (MSB부터)
    weights = [128; 64; 32; 16; 8; 4; 2; 1]; % 각 비트의 가중치
    decimal_values = bytes_matrix * weights; % Mx1 벡터
end


% 8비트씩 그룹화
% reshape을 사용하여 6행 N/6열 행렬로 변환 후 전치
indices_bits_mat = reshape(indices_bits, 6, []).'; % Mx6 행렬 (M = N/6)

% 논리 값을 더블 형으로 변환
indices_bits_mat = double(indices_bits_mat);

% 8비트에서 10진수로 변환 (MSB부터)
weights = [32; 16; 8; 4; 2; 1]; % 각 비트의 가중치
indices_dec = indices_bits_mat * weights; % Mx1 벡터


% Step 13.3: Indices 비트 변환
% indices_dec = bits6_to_decimal(indices_bits); % 16384 x 1 10진수 배열
indices_dec = indices_dec+1;

% Step 13.4: Centroids 비트 변환
centroids_dec = bits_to_decimal(centroids_bits); % 192 x 1 10진수 배열
centroids_dec = reshape(centroids_dec,[64 3]);

% Step 13.5: 변환된 10진수 배열 확인 (옵션)
disp(['Indices 변환된 10진수 값의 개수: ', num2str(length(indices_dec))]);
disp(['Centroids 변환된 10진수 값의 개수: ', num2str(length(centroids_dec))]);

% Step 8: Reconstruct the quantized image (optional verification)
quantized_img = reshape(centroids_dec(indices_dec, :), 128, 128, 3);
quantized_img = uint8(quantized_img);
imwrite(uint8(quantized_img), 'Lena_restored_kmeans_a.png');

subplot(1, 2, 1);imshow(uint8(quantized_img)); title('K-Means Quantized Image');

smoothed_img = imbilatfilt(quantized_img);

subplot(1, 2, 2); imshow(smoothed_img); title('After Bilateral Filtering');



% Step 13.6: 필요에 따라 10진수 데이터를 저장 (예: .mat 파일)
save('kmeans_decimal_data.mat', 'indices_dec', 'centroids_dec');

% **참고**: 'bits_to_decimal' 함수는 같은 스크립트 내에 정의되어야 합니다.
% 또는 별도의 함수 파일로 저장하여 사용할 수 있습니다.

% MATLAB 코드 종료