% 통합된 코드 (Tx와 Rx 기능을 통합하고, 파일 입출력 방식으로 변경)
TX_RX_WITHOUT_SETTING = true; % 최초 실행 or parameter 변경시 무조건 얘는 주석 아래는 주석해제
% TX_RX_WITHOUT_SETTING = false; % SETTING 한 번 이상 실행했고 para 변경 없으면 얘 주석 위 주석해제

N = 16;
if TX_RX_WITHOUT_SETTING == false
    clear all;
    close all;
    clc;
    TX_RX_WITHOUT_SETTING = false;
    N = 16;
    disp('### SETTING을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');

    % 주요 parameter setting
    
    sampling_freq = 40000; % Nyquist freq = sampling_freq / 2 임을 잊지 말아야
    max_subcarrier_freq = 6000; % 최대 서브캐리어 주파수 설정
    
    WHETHER_REPETITION_CODING = true; % repetition coding을 하려면 아래 코드를 주석하고 얘를 주석 해제
    % WHETHER_REPETITION_CODING = false; % repetition coding X면 위를 주석, 얘는 주석 해제
    repetition_factor = 3;
    
    % IFFT의 N을 세팅. sub-carrier 수는 N/2 - 1
    % N = 128;
    delta_subcarrier_freq = sampling_freq / N;
    
    % 사용할 서브캐리어 인덱스 결정 (0 ~ max_subcarrier_freq 이하의 서브캐리어만 사용)
    valid_subcarrier_indices = find((0:N/2-1) * delta_subcarrier_freq <= max_subcarrier_freq);
    
    % cyclic_prefix 길이
    N_cp = N / 4;
    
    % 이미지 경로
    img_path = 'Lena_color.png'; % Lena_color.png는 코드와 같은 경로에 있어야 함
    
    % 이미지 사이즈를 어떤 크기로 고정할 것인가
    fixed_img_size = [128 128];
    
    % Preamlbe 길이 (단위: sample)
    Tp = 1000;
    
    % Preamble Chirp 계수 설정
    omega = 10;
    mu = 0.1;
    tp = (1:Tp).';
    preamble = cos(omega * tp + mu * tp.^2 / 2);

    % 주요 parameter setting 부분 끝.

    disp('### tx_signal을 만들겠습니다.');
    % imshow(img_path); % 상대 경로에 있는 이미지 잘 불러오는지 테스트

    disp('image를 읽어옵니다.');
    img = imread(img_path);

    disp('image를 고정된 크기로 줄입니다.');
    resized_img = imresize(img, fixed_img_size);

    disp('image를 gray-scale로 바꿉니다.');
    gray_img = rgb2gray(resized_img);

    disp('image를 monochrome으로 바꿉니다.');
    binarised_img = imbinarize(gray_img);

    disp('bit로 전송하기 위해, 128x128x을 단일 Column vector로 형식을 바꿉니다.');
    binarised_img_column_vector = binarised_img(:);

    if WHETHER_REPETITION_CODING == true
        disp('repetition coding을 적용합니다.');
        binarised_img_column_vector = repelem(binarised_img_column_vector, repetition_factor);
    end

    disp('PSK modulation을 진행합니다.');
    bpsk_modulated_symbols = 2 * binarised_img_column_vector - 1;

    disp('전체 symbol의 개수를 구합니다.');
    symbol_length = length(bpsk_modulated_symbols);

    disp('ofdm_symbols 개수를 구합니다.');
    num_ofdm_symbols = symbol_length / length(valid_subcarrier_indices);

    disp('전체 OFDM 블록 개수를 구합니다. 이것은 num_ofdm_symbols에, pilot 신호의 개수를 더한 것입니다.');
    % pilot 신호는 4번마다 나오기 때문에 num_ofdm_symbols / 4
    total_OFDM_block_number = num_ofdm_symbols + (num_ofdm_symbols / 4);
    
    disp('이제 psk_modulated_symbols를 각 subcarrier 주파수의 진폭에 싣습니다.');
    symbols_mapped_to_subcarrier_freq = {};
    for i = 1:num_ofdm_symbols
        subcarrier_mapping = zeros(N, 1);
        subcarrier_mapping(valid_subcarrier_indices + 1) = bpsk_modulated_symbols(length(valid_subcarrier_indices)*(i-1)+1 : length(valid_subcarrier_indices)*i);
        symbols_mapped_to_subcarrier_freq{end + 1} = [subcarrier_mapping; flip(conj(subcarrier_mapping(2:end-1)))];
    end

    disp('이제 subcarrier 주파수에 mapping된 신호를 time-domain으로 IFFT(IDFT 상위호환) 하겠습니다.');
    symbols_IFFTed_time = {};
    for i = 1:length(symbols_mapped_to_subcarrier_freq)
        symbols_IFFTed_time{end + 1} = ifft(symbols_mapped_to_subcarrier_freq{i}, N) * sqrt(N);
    end

    disp('cyclic prefix를 집어넣습니다.');
    for i = 1:length(symbols_IFFTed_time)
        symbols_IFFTed_time{i} = [symbols_IFFTed_time{i}(end - N_cp + 1 : end); symbols_IFFTed_time{i}];
    end

    disp('Pilot signal을 주파수 영역에서 생성한 후, IFFT를 통해 time-domain으로 옮깁니다.');
    pilot_freq = ones(N, 1);
    pilot_time = ifft(pilot_freq) * sqrt(N);
    pilot_time = [pilot_time(end - N_cp + 1 : end); pilot_time];

    disp('preamble, pilot_time, symbols_IFFTed_time 순으로 tx_signal을 구성합니다.');
    tx_signal = [preamble; pilot_time];
    for i = 1 : length(symbols_IFFTed_time)
        tx_signal = [tx_signal; symbols_IFFTed_time{i}];
        if rem(i, 4) == 0 && i ~= length(symbols_IFFTed_time)
            tx_signal = [tx_signal; pilot_time];
        end
    end

    disp('tx_signal을 파일로 저장합니다.');
    save('tx_signal.mat', 'tx_signal');

    disp('Tx SETTING이 완료되었습니다.');
    WHETHER_TX_SETTING_DONE = true;
end

if TX_RX_WITHOUT_SETTING == true && WHETHER_TX_SETTING_DONE == true
    clc;
    close all;

    disp('### 만들어져 있는 tx_signal을 처리하겠습니다.');

    % tx_signal.mat 파일 로드
    load('tx_signal.mat', 'tx_signal');

    % 주요 parameter setting
    sampling_freq = 40000;
    % N = 128;
    N_cp = N / 4;
    Tp = 1000;
    fixed_img_size = [128 128];
    repetition_factor = 3;
    img_path = 'Lena_color.png';
    WHETHER_REPETITION_CODING = true;
    max_subcarrier_freq = 6000;
    delta_subcarrier_freq = sampling_freq / N;
    valid_subcarrier_indices = find((0:N/2-1) * delta_subcarrier_freq <= max_subcarrier_freq);

    % Preamble 설정
    omega = 10;
    mu = 0.1;
    tp = (1:Tp).';
    preamble = cos(omega * tp + mu * tp.^2 / 2);

    % Time Synchronisation
    [xC, lags] = xcorr(tx_signal, preamble);
    [~, idx] = max(xC);
    start_pt = lags(idx);

    % preamble 검출 후 신호 추출
    rx_signal = tx_signal(start_pt + Tp + 1 : end);

    disp('수신 신호 분석 중입니다');
    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['시작 시각: ', char(start_time)]);

    % Serial to Parallel
    symbol_length = fixed_img_size(1) * fixed_img_size(2);
    if WHETHER_REPETITION_CODING == true
        symbol_length = symbol_length * repetition_factor;
    end
    num_ofdm_symbols = symbol_length / length(valid_subcarrier_indices);
    total_OFDM_block_number = num_ofdm_symbols + (num_ofdm_symbols / 4);

    OFDM_blks = {};
    for i = 1 : total_OFDM_block_number
        OFDM_blks{end + 1} = rx_signal(N_cp + 1 : N + N_cp);
        rx_signal = rx_signal(N_cp + N + 1 : end);
    end

    % Discrete Fourier Transform (DFT)
    demode_OFDM_blks = {};
    for i = 1 : length(OFDM_blks)
        demode_OFDM_blks{end + 1} = fft(OFDM_blks{i} / sqrt(N));
    end

    % Channel Estimation & Equalisation
    symbols_eq = {};
    for i = 1 : length(demode_OFDM_blks)
        if rem(i, 5) == 1
            channel = demode_OFDM_blks{i} ./ ones(N, 1);
        else
            symbols_eq{end + 1} = demode_OFDM_blks{i} ./ channel;
        end
    end

    % Detection
    symbols_detect = {};
    for i = 1 : length(symbols_eq)
        symbols_detect{end + 1} = sign(real(symbols_eq{i}));
    end

    % Demodulation
    symbols_est = [];
    for i = 1 : length(symbols_detect)
        symbols_est = [symbols_est; symbols_detect{i}(valid_subcarrier_indices + 1)];
    end

    % BPSK Demodulation
    decoded_bits = (symbols_est + 1) / 2;

    % Repetition Decoding
    decoded_bits_reshaped = reshape(decoded_bits, repetition_factor, []);
    sums = sum(decoded_bits_reshaped);
    repetition_decoded_bits = (sums > (repetition_factor / 2)).';

    % Source Decoding & Show img
    img_size = sqrt(length(repetition_decoded_bits));
    estimated_img = reshape(repetition_decoded_bits, [img_size, img_size]); 
    resized_estimated_img = imresize(estimated_img, fixed_img_size);
    imshow(resized_estimated_img);

    disp('Communication Tool box에 있는 biterr 함수를 사용하여 Bit Error Rate를 구합니다.');
    img = imread(img_path);
    resized_img = imresize(img, fixed_img_size);
    gray_img = rgb2gray(resized_img);
    binarised_img = imbinarize(gray_img);

    [~, BER] = biterr(binarised_img, estimated_img);
    BER

    % 완료 시각 출력
    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['완료 시각: ', char(end_time)]);
    elapsed_time = end_time - start_time;
    disp(['수신 신호 분석 경과 시간[s]: ', char(elapsed_time)]);
end