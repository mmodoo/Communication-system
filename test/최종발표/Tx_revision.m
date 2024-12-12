Whether_Load_MAT_and_Setting__OR__just_Tx = true;
% Whether_Load_MAT_and_Setting__OR__just_Tx = false;

if Whether_Load_MAT_and_Setting__OR__just_Tx == true
    clc;
    close all;
    clear all;

    Whether_just_save_Tx_Setting_MAT = true;
    % Whether_just_save_Tx_Setting_MAT = false;

    % Step 1: Read and preprocess the image
    original_img = imread('test3_256X256.png');
    
    % Step 2: Resize the image to 64x64
    resized_img = imresize(original_img, 0.5);
    
    % Step 3: Apply Bilateral Filtering for smoother transitions
    smoothed_img = imbilatfilt(resized_img);
    
    % Step 4: Histogram Matching to align color distribution
    matched_img = imhistmatch(smoothed_img, resized_img);
    
    % % Step 5: Edge refinement using Gaussian filter and edge detection
    % edges = edge(rgb2gray(matched_img), 'Canny');
    % refined_img = imgaussfilt(double(matched_img), 1); % Gaussian smoothing
    % refined_img(edges) = double(resized_img(edges)); % Blend original edges
    
    % Step 6: Apply K-Means Clustering for compression
    [h, w, ~] = size(matched_img);
    num_pixels = h * w;
    blocks = reshape(double(matched_img), num_pixels, 3); % Each pixel as a row (RGB vector)
    
    K = 64; % Number of clusters
    [indices, centroids] = kmeans(blocks, K);
    
    % Step 7: Save indices and centroids
    save('kmeans_data.mat', 'indices', 'centroids', 'h', 'w');
    
    % Step 8: Reconstruct the quantized image (optional verification)
    quantized_img = reshape(centroids(indices, :), 128, 128, 3);

    num_bits_indices = 6;
    indices = indices-1;
    binary_indices = dec2bin(indices, num_bits_indices); % Convert to char array of '0's and '1's
    binary_indices = binary_indices'; % Transpose to make it easier to handle
    binary_indices = binary_indices(:)'; % Convert to a single row vector
    
    % Convert 'centroids' to binary
    % Option 1: Convert to uint8 if centroids are in the range [0, 255]
    % Option 2: Convert to single precision (32 bits) if you need higher precision
    % Here, we'll use uint8 for compactness
    
    % First, ensure centroids are in uint8 format
    centroids_uint8 = uint8(centroids);
    
    % Flatten the centroids matrix
    centroids_flat = centroids_uint8(:); % Column-wise
    
    % Convert each byte to its 8-bit binary representation
    binary_centroids = dec2bin(centroids_flat, 8);
    binary_centroids = binary_centroids'; % Transpose
    binary_centroids = binary_centroids(:)'; % Convert to a single row vector
    
    % Concatenate binary_indices and binary_centroids
    binary_combined = [binary_indices, binary_centroids];
    combined_logical_data = imbinarize(uint8(binary_combined));
    save('kmeans_logical_data.mat','combined_logical_data');


    Original_MAT_path_and_file_name = 'kmeans_logical_data.mat';
    % Original_MAT_path_and_file_name = 'C:\Users\user\Desktop\졸업드가자\통신시스템설계\최종발표\test.mat';
    load(Original_MAT_path_and_file_name);
    Original_MAT_data = combined_logical_data;
    Repeated_MAT_data = Original_MAT_data.';
    Tx_Setting_MAT_path_and_file_name = "Tx_Setting.mat";
    Tx_signal_MAT_path_and_file_name = "Tx_signal.mat";
    Tx_signal_WAV_path_and_file_name = "Tx_signal.wav";

    Whether_NOT_Repetition_coding__OR__Repetition_How_Many = 3;

    Modulation_Number = 4;
    N = 288;
    Subcarrier_Freq_Divided_by = 6;

    N_cp__that_is__Cyclic_Prefix_Length = N / 4;

    Sampling_Freq = 48000;
    Cut_off_Freq = 16000;
    Cut_off_Freq_normalised = Cut_off_Freq / (Sampling_Freq / 2);

    T_preamble_sec = 2;
    % omega = 10;
    % mu = 0.1;
    t_preamble_vector = (0:1/Sampling_Freq:T_preamble_sec).';
    Freq_Preamble_Start = 18000;
    Freq_Preamble_End = 19000;
    Phase_Preamble = 2 * pi * (Freq_Preamble_Start * t_preamble_vector + ((Freq_Preamble_End - Freq_Preamble_Start)/(2*T_preamble_sec))*t_preamble_vector.^2);
    Preamble = cos(Phase_Preamble);

    Whether_Plot__OR__Not = true;
    % Whether_Plot__OR__Not = false;

    if Whether_NOT_Repetition_coding__OR__Repetition_How_Many > 1
        Repeated_MAT_data = repelem(Repeated_MAT_data, Whether_NOT_Repetition_coding__OR__Repetition_How_Many);
    end

    rng('default');
    inter_leaver_order = randperm(length(Repeated_MAT_data));
    Inter_Left_Repeated_MAT_data = Repeated_MAT_data(inter_leaver_order);

    Symbolised_Inter_Left_Repeated_MAT_data = bit2int(Inter_Left_Repeated_MAT_data, 2);
    PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data = pskmod(Symbolised_Inter_Left_Repeated_MAT_data, Modulation_Number, pi / 4);
    Symbol_Length = length(PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data);
    OFDM_Symbol_Number = Symbol_Length / (N / (2 * Subcarrier_Freq_Divided_by));
    Rest_Num_PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data = 0;
    if isinteger(OFDM_Symbol_Number) == false
        OFDM_Symbol_Number = floor(Symbol_Length / (N / (2 * Subcarrier_Freq_Divided_by)));
        Rest_Num_PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data = mod(Symbol_Length,  (N / (2 * Subcarrier_Freq_Divided_by)));
    end
    Total_OFDM_Symbol_Number_including_Pilot = OFDM_Symbol_Number + (OFDM_Symbol_Number / 4);

    Symbols_mapped_to_Subcarrier_Freq = {};
    for i = 1:OFDM_Symbol_Number
        symbols_temp_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
        symbols_temp_2 = PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data(1:floor(N/Subcarrier_Freq_Divided_by)/2);
        symbols_temp_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
        PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data = PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data(1+floor(N/Subcarrier_Freq_Divided_by)/2:end);
        Symbols_mapped_to_Subcarrier_Freq{end+1} = [symbols_temp_1; symbols_temp_2; symbols_temp_3];
        Symbols_mapped_to_Subcarrier_Freq{end} = [0; Symbols_mapped_to_Subcarrier_Freq{end}; 0; flip(conj(Symbols_mapped_to_Subcarrier_Freq{end}))];
    end
    
    if Rest_Num_PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data ~= 0
        symbols_temp_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
        symbols_temp_2 = PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data(1:end);
        symbols_temp_2 = [symbols_temp_2; zeros(floor(N/Subcarrier_Freq_Divided_by/2) - Rest_Num_PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data)];
        symbols_temp_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
        Symbols_mapped_to_Subcarrier_Freq{end+1} = [symbols_temp_1; symbols_temp_2; symbols_temp_3];
        Symbols_mapped_to_Subcarrier_Freq{end} = [0; Symbols_mapped_to_Subcarrier_Freq{end}; 0; flip(conj(Symbols_mapped_to_Subcarrier_Freq{end}))];
    end

    Symbols_IFFTed_at_Time = {};
    for i = 1:length(Symbols_mapped_to_Subcarrier_Freq)
        Symbols_IFFTed_at_Time{end + 1} = ifft(Symbols_mapped_to_Subcarrier_Freq{i}, N+2) * sqrt(N+2);
    end

    for i = 1:length(Symbols_IFFTed_at_Time)
        Symbols_IFFTed_at_Time{i} = [Symbols_IFFTed_at_Time{i}(end - N_cp__that_is__Cyclic_Prefix_Length + 1 : end); Symbols_IFFTed_at_Time{i}];
    end

    rng('default');
    temp_pilot_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
    temp_pilot_2 = 2 * (randi([0, 1], floor(N/Subcarrier_Freq_Divided_by)/2, 1) - 0.5);
    temp_pilot_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
    temp_pilot = [temp_pilot_1; temp_pilot_2; temp_pilot_3];
    Pilot_freq = [0; temp_pilot; 0; flip(conj(temp_pilot))];  % DC, 중간, 허미션 처리
    Pilot_at_Time = ifft(Pilot_freq) * sqrt(N+2);
    Pilot_at_Time = [Pilot_at_Time(end - N_cp__that_is__Cyclic_Prefix_Length + 1 : end); Pilot_at_Time];

    % disp('a');
    Tx_signal = [Pilot_at_Time];
    for i = 1 : length(Symbols_IFFTed_at_Time)
        Tx_signal = [Tx_signal; Symbols_IFFTed_at_Time{i}];
        if rem(i, 4) == 0 && i ~= length(Symbols_IFFTed_at_Time)
            Tx_signal = [Tx_signal; Pilot_at_Time];
        end
    end

    % disp('b');
    % 필터 차수 설정 (예: 100차)
    filter_order = 4;
    
    % FIR 고역 통과 필터 설계
    % b = fir1(order, cut_off_freq_normalised, 'high');
    [b, a] = butter(filter_order, Cut_off_Freq_normalised, 'high');
    
    % 필터의 주파수 응답 확인 (선택 사항)
    % fvtool(b, 1);
    
    % Tx_signal에 필터 적용
    % filtered_signal = filter(b, 1, Tx_signal);
    % Tx_signal = filter(b, 1, Tx_signal);
    Tx_signal = filtfilt(b, a, Tx_signal);

    % Tx_signal_archieved = Tx_signal;
    Tx_signal = [Preamble; Tx_signal];
    % temp = zeros(5, 1);
    % Tx_signal = [temp; Tx_signal];
    Tx_signal = normalize(Tx_signal, 'range', [-1 1]);

    Recording_Time_Sec = length(Tx_signal) / Sampling_Freq

    audiowrite(Tx_signal_WAV_path_and_file_name, Tx_signal, Sampling_Freq);
    save(Tx_signal_MAT_path_and_file_name, 'Tx_signal');

    save(Tx_Setting_MAT_path_and_file_name, ...
        'Whether_NOT_Repetition_coding__OR__Repetition_How_Many', ...
        'Modulation_Number', ...
        'N', ...
        'Subcarrier_Freq_Divided_by', ...
        'N_cp__that_is__Cyclic_Prefix_Length', ...
        'Sampling_Freq', ...
        'Cut_off_Freq', ...
        'Cut_off_Freq_normalised', ...
        'T_preamble_sec', ...
        't_preamble_vector', ...
        'Freq_Preamble_Start', ...
        'Freq_Preamble_End', ...
        'Phase_Preamble', ...
        'Preamble', ...
        'OFDM_Symbol_Number', ...
        'Rest_Num_PSK_modulated_Symbolised_Inter_Left_Repeated_MAT_data', ...
        'Total_OFDM_Symbol_Number_including_Pilot', ...
        'Recording_Time_Sec');

    if Whether_just_save_Tx_Setting_MAT == true
        return;
    end

    if Whether_Plot__OR__Not == true
        figure;
        plot(Tx_signal);
    
        % 시간 축을 초 단위로 변환하여 Tx_signal을 플로팅합니다.
        t = (0:length(Tx_signal)-1) / Sampling_Freq; % 시간 축 생성
        figure;
        plot(t, Tx_signal);
        xlabel('시간 (초)');
        ylabel('신호 진폭');
        title('Tx\_signal의 시간 영역 파형');
    
    
        % Tx_signal의 주파수 스펙트럼을 계산하고 플로팅합니다.
        N_fft = length(Tx_signal);
        f = (-N_fft/2:N_fft/2-1)*(Sampling_Freq/N_fft); % 주파수 축 생성
        Tx_signal_fft = fftshift(fft(Tx_signal));
        
        figure;
        plot(f, abs(Tx_signal_fft));
        xlabel('주파수 (Hz)');
        ylabel('스펙트럼 진폭');
        title('tx\_signal의 주파수 스펙트럼');
        xlim([0, Sampling_Freq/2]); % 양의 주파수 성분만 표시
    
    
        figure;
        spectrogram(Tx_signal, 256, 128, 256, Sampling_Freq, 'yaxis');
    end
    
elseif Whether_Load_MAT_and_Setting__OR__just_Tx == false

end