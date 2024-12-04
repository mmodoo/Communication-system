% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 1;
% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 2;
Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 3;
% Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening = 4;


clearvars -except Whether_Load_MAT__OR__WAV__OR__just_Rx__OR__Listening;
clc;
close all;

Original_MAT_path_and_file_name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\kmeans_logical_data.mat";
Tx_Setting_MAT_path_and_file_name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\Tx_Setting.mat";
Tx_signal_MAT_path_and_file_name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\Tx_signal.mat";
Tx_signal_WAV_path_and_file_name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\Tx_signal.wav";
Rx_signal_MAT_path_and_file_name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\최종발표\Rx_signal.mat";

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

load(Original_MAT_path_and_file_name);

different_bits = (combined_logical_data ~= decoded_bits);

different_bits_Number = sum(different_bits)

BER = different_bits_Number / length(combined_logical_data)