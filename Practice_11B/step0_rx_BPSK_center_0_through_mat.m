RX_WITHOUT_SETTING = true; % 최초 실행 or parameter 변경시 무조건 얘는 주석 아래는 주석해제
% RX_WITHOUT_SETTING = false; % SETTING 한 번 이상 실행했고 para 변경 없으면 얘 주석 위 주석해제

if RX_WITHOUT_SETTING == false
    clear all;
    close all;
    clc;
    RX_WITHOUT_SETTING = false;
    % TX_THROUGH_MAT = true;
    TX_THROUGH_MAT = false;
    % WHETHER_BASIC_PILOT = true;
    WHETHER_BASIC_PILOT = false;
    WHETHER_INTERLEAVING = true;
    % WHETHER_INTERLEAVING = false;
    disp('### SETTING을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');

    % 주요 parameter setting
    
    sampling_freq = 40000; % Nyquist freq = sampling_freq / 2 임을 잊지 말아야
    
    % WHETHER_REPETITION_CODING = true; % repetition coding을 하려면 아래 코드를 주석하고 얘를 주석 해제
    WHETHER_REPETITION_CODING = false; % repetition coding X면 위를 주석, 얘는 주석 해제
    repetition_factor = 3;
    
    % IFFT의 N을 세팅. sub-carrier 수는 N/2 - 1
    N = 256;
    
    % cyclic_prefix 길이
    N_cp = N / 4;
    
    % 이미지 경로
    img_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\실습 수업 매트랩 코드\week11B_HW\Lena_color.png';
    save_load_path_and_file_name = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\실습 수업 매트랩 코드\week11B_HW\tx_signal.mat';
    
    % 이미지 사이즈를 어떤 크기로 고정할 것인가
    fixed_img_size = [128 128];
    
    % Preamlbe 길이 (단위: sample)
    Tp = 1000;
    
    % Preamble Chirp 계수 설정
    omega = 10;
    mu = 0.1;
    tp = (1:Tp).';
    preamble = cos(omega * tp + mu * tp.^2 / 2);
    
    % RX_WITHOUT_SETTING == false라면, 즉 SETTING을 실행하면 recording time이 아래 부분에서 계산된다
    buffer_for_recording_time_sec = 5;

    disp('전체 symbol의 개수를 구합니다.');
    symbol_length = fixed_img_size(1) * fixed_img_size(2);
    if WHETHER_REPETITION_CODING == true
        symbol_length = symbol_length * repetition_factor;
    end
    disp('ofdm_symbols 개수를 구합니다.');
    num_ofdm_symbols = symbol_length / (N / 2);
    disp('전체 OFDM 블록 개수를 구합니다. 이것은 num_ofdm_symbols에, pilot 신호의 개수를 더한 것입니다.');
    % pilot 신호는 4번마다 나오기 때문에 num_ofdm_symbols / 4
    total_OFDM_block_number = num_ofdm_symbols + (num_ofdm_symbols / 4);

    disp('recording_time을 계산합니다 (+buffer).');
    recording_time_sec = ((Tp + (total_OFDM_block_number) * (N + N_cp)) / sampling_freq) + buffer_for_recording_time_sec
end

if RX_WITHOUT_SETTING == true
    clc;
    close all;

    if TX_THROUGH_MAT == true
        % tx_signal.mat 파일 로드
        disp('tx_signal 파일을 읽어옵니다.');
        load(save_load_path_and_file_name, 'tx_signal');
        rx_signal = tx_signal;
        figure;
        % plot(rx_signal);
        rx_signal_archieve = rx_signal;
        % plot(rx_signal_archieve);
        plot(tx_signal);
    else
        % 수신 시작
        devicereader = audioDeviceReader(sampling_freq);
        setup(devicereader);
        disp(['총 녹음 시간: ', num2str(recording_time_sec), '초']);
        rx_signal = [];
        elapsed = 0;
        tic;
        
        while toc < recording_time_sec  % recording_time_sec 동안 녹음
            acquiredAudio = devicereader();
            rx_signal = [rx_signal; acquiredAudio];
            
            % 1초마다 진행 상태 업데이트
            if floor(toc) > elapsed
                elapsed = floor(toc);
                fprintf('\r녹음 진행: %d초 / %d초 (%.1f%%)', ...
                    elapsed, ceil(recording_time_sec), (elapsed/recording_time_sec)*100);
            end
        end
        fprintf('\n');  % 줄바꿈 추가
        disp('Recording Completed')
    end
    
    
    % Time Synchronisation
    [xC, lags] = xcorr(rx_signal, preamble);
    [~, idx] = max(xC);
    start_pt = lags(idx);
    
    rx_signal = rx_signal(start_pt + Tp + 1 : end);
    
    disp('수신 신호 분석 중입니다');
    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['시작 시각: ', char(start_time)]);

    % Serial to Parallel
    OFDM_blks = {};
    for i = 1 : total_OFDM_block_number
        OFDM_blks{end + 1} = rx_signal(N_cp + 1 : (N + 2) + N_cp);
        rx_signal = rx_signal(N_cp + (N + 2) + 1 : end);
    end
    
    % Discrete Fourier Transform (DFT)
    demode_OFDM_blks = {};
    for i = 1 : length(OFDM_blks)
        demode_OFDM_blks{end + 1} = fft(OFDM_blks{i} / sqrt(N + 2));
    end
    
    % Channel Estimation & Equalisation. Pilot 여기서 활용됨
    symbols_eq = {};
    if WHETHER_BASIC_PILOT == true
        pilot_freq = ones(N + 2, 1);
    else
        rng('default');
        temp_pilot = 2 * (randi([0, 1], N/2, 1) - 0.5);  % N/2 개만 생성
        pilot_freq = [0; temp_pilot; 0; flip(conj(temp_pilot))];  % DC, 중간, 허미션 처리
    end
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
        symbols_detect{end + 1} = sign(real(symbols_eq{i}));
    end
    
    % Demodulation
    symbols_est = [];
    for i = 1 : length(symbols_detect)
        symbols_est = [symbols_est; symbols_detect{i}(2 : N / 2 + 1)];
    end
    
    % BPSK Demodulation
    decoded_bits = (symbols_est + 1) / 2;
    
    % 동일한 interleaver_order 생성
    rng('default'); % 동일한 시드로 초기화하여 Tx와 같은 순서 생성
    interleaver_order = randperm(length(decoded_bits));
    
    % Disinterleaving
    [~, deinterleaver_order] = sort(interleaver_order); % 원래 순서를 찾기 위해 역순서를 계산
    decoded_bits = decoded_bits(deinterleaver_order); % 원래 순서로 복원

    if WHETHER_REPETITION_CODING == true
        % Repetition Decoding
        % 3개씩 묶어서 reshape
        decoded_bits_reshaped = reshape(decoded_bits, repetition_factor, []);
        
        % 각 열의 합을 계산 (다수결 원칙 적용)
        sums = sum(decoded_bits_reshaped);
        
        % 합이 2 이상이면 1, 미만이면 0으로 판정
        repetition_decoded_bits = (sums > (repetition_factor / 2)).';
    end
    
    % Source Decoding & Show img
    % 이미지 크기 계산 시 repetition decoding으로 인한 크기 변화 고려
    if WHETHER_REPETITION_CODING == true
        img_size = sqrt(length(repetition_decoded_bits));
        estimated_img = reshape(repetition_decoded_bits, [img_size, img_size]);
    else
        img_size = sqrt(length(decoded_bits));
        estimated_img = reshape(decoded_bits, [img_size, img_size]);
    end
    resized_estimated_img = imresize(estimated_img, fixed_img_size);
    figure;
    imshow(resized_estimated_img);

    disp('Communication Tool box에 있는 biterr 함수를 사용하여 Bit Error Rate를 구합니다.');
        disp('image 경로를 지정해줍니다.');
    % img_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\실습 수업 매트랩 코드\week9_HW';
    disp(img_path);
    
    % disp('image를 읽어옵니다.');
    img = imread(img_path);
    
    % disp('image를 factor 비율만큼 줄입니다.');
    resized_img = imresize(img, fixed_img_size);
    
    % disp('image를 gray-scale로 바꿉니다.');
    gray_img = rgb2gray(resized_img);
    
    % disp('image를 monochrome으로 바꿉니다.');
    binarised_img = imbinarize(gray_img);

    [~, BER] = biterr(binarised_img, estimated_img);
    BER

    % 완료 시각 출력
    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['완료 시각: ', char(end_time)]);
    % 걸린 시간 계산 및 출력
    elapsed_time = end_time - start_time;
    disp(['수신 신호 분석 경과 시간[s]: ', char(elapsed_time)]);
end