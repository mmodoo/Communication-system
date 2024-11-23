function Estimated_Img = Rx_Step_2_Get_Estimated_Img(Rx_Setting_MAT_path_and_File_Name, Tx_signal)

    load(Rx_Setting_MAT_path_and_File_Name);

    disp('## Tx_signal 분석 중입니다');
    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['## 시작 시각: ', char(start_time)]);

    [xC, lags] = xcorr(Tx_signal, Preamble);
    [~, idx] = max(xC);
    start_pt = lags(idx);

    Tx_signal = Tx_signal(start_pt + T_p__that_is_preamble_1_length_Unit_is_Sample + 1 : end);

    % Serial to Parallel
    OFDM_blks = {};
    for i = 1 : Total_OFDM_symbols_Number_that_is_including_Pilot
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
    if Whether_Basic_Pilot__OR__PAPR_improved_Pilot == true
        pilot_freq = ones(N + 2, 1);
    else
        rng('default');
        if Whether_Pilot_Use_all_freq__OR__High_freq_only == true
            temp_pilot = 2 * (randi([0, 1], N/2, 1) - 0.5);  % N/2 개만 생성
        else
            % temp_pilot_1 = zeros(3*N/8, 1);
            % temp_pilot_2 = 2 * (randi([0, 1], N/8, 1) - 0.5);
            % temp_pilot_1 = zeros((Subcarrier_Freq_Divided_by-1)*N/(2*Subcarrier_Freq_Divided_by), 1);
            % temp_pilot_2 = 2 * (randi([0, 1], N/(2*Subcarrier_Freq_Divided_by), 1) - 0.5);
            % temp_pilot = [temp_pilot_1; temp_pilot_2];
            temp_pilot_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
            temp_pilot_2 = 2 * (randi([0, 1], floor(N/Subcarrier_Freq_Divided_by)/2, 1) - 0.5);
            temp_pilot_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
            temp_pilot = [temp_pilot_1; temp_pilot_2; temp_pilot_3];
        end
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
    
    % Source Decoding & Show img
    % 이미지 크기 계산 시 repetition decoding으로 인한 크기 변화 고려
    if Whether_NOT_Repetition_coding__OR__Repetition_How_Many == 1
        img_size = sqrt(length(decoded_bits));
        Estimated_Img = reshape(decoded_bits, [Fixed_Img_Size(1), Fixed_Img_Size(2)]);
    else
        img_size = sqrt(length(decoded_bits));
        Estimated_Img = reshape(decoded_bits, [Fixed_Img_Size(1), Fixed_Img_Size(2)]);
    end
    Estimated_Img = imresize(Estimated_Img, Fixed_Img_Size);
    figure;
    imshow(Estimated_Img);

    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['## Tx_signal 분석 완료 시각: ', char(end_time)]);
    % 걸린 시간 계산 및 출력
    elapsed_time = end_time - start_time;
    disp(['## Tx_signal 분석 경과 시간[s]: ', char(elapsed_time)]);
end