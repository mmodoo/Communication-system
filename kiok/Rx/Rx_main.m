Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 1;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 2;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 3;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 4;
Whether_there_was_an_error = false;
% 컴퓨터 바뀔 때마다 체크해야할 부분 == CTRL + F "여기 반드시 확인"
% 여기 반드시 확인
Save_and_Load_Tx_signal_MAT_Path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Tx\Tx_signal.MAT";
Tx_Signal_Wav_Rx_Path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Tx\Tx_signal.WAV";
Rx_Setting_MAT_path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Rx\Rx_Setting.MAT";

try
    load(Rx_Setting_MAT_path_and_File_Name);

    vars = who;
    required_vars = {'Whether_NOT_Repetition_coding__OR__Repetition_How_Many', ...
        'Sampling_Freq', ...
        'Recording_Time_Sec', ...
        'Preamble', ...
        'OFDM_symbols_Number', ...
        'Total_OFDM_symbols_Number_that_is_including_Pilot', ...
        'Fixed_Img_Size', ...
        'Whether_PAPR_improved_inter_leaving__OR__NOT', ...
        'Modulation_Number', ...
        'N', ...
        'Subcarrier_Freq_Divided_by', ...
        'Whether_Basic_Pilot__OR__PAPR_improved_Pilot', ...
        'N_cp__that_is__Cyclic_Prefix_Length', ...
        'Whether_Pilot_Use_all_freq__OR__High_freq_only', ...
        'T_p__that_is_preamble_1_length_Unit_is_Sample'};
    for i = 1:length(required_vars)
        if ~ismember(required_vars{i}, vars)
            disp('Rx_Setting_MAT 파일에 변수가 누락되어 있습니다.');
            disp('# Setting 부분으로 갑니다.');
            Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 2;
            Whether_there_was_an_error = true;
            break;
        end
    end
catch ME
    fprintf('파일 로드 실패: %s\n', ME.message);
    disp('# Setting 부분으로 갑니다.');
    Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 2;
    Whether_there_was_an_error = true;
end

if Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 2
    clearvars -except Tx_Signal_Wav_Rx_Path_and_File_Name ...
        Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV ...
        Whether_there_was_an_error ...
        Rx_Setting_MAT_path_and_File_Name ...
        Save_and_Load_Tx_signal_MAT_Path_and_File_Name;
    close all;
    if Whether_there_was_an_error == false
        clc;
    end

    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['# Setting 시작 시각: ', char(start_time)]);
    disp('# Setting을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');

    disp('## Parameter들을 설정하겠습니다.');

    % Whether_NOT_Repetition_coding__OR__Repetition_How_Many = 1;
    Whether_NOT_Repetition_coding__OR__Repetition_How_Many = 3;
    Fixed_Img_Size = [128 128];

    Whether_PAPR_improved_inter_leaving__OR__NOT = true;
    % Whether_PAPR_improved_inter_leaving__OR__NOT = false;

    Modulation_Number = 4;
    N = 288; % 6의 배수
    Subcarrier_Freq_Divided_by = 6;

    [OFDM_symbols_Number, Total_OFDM_symbols_Number_that_is_including_Pilot] ...
        = Rx_Step_1_Calculate_OFDM_symbols_Number_and_Total_Number(...
        N, ...
        Whether_NOT_Repetition_coding__OR__Repetition_How_Many, ...
        Fixed_Img_Size, ...
        Modulation_Number, ...
        Subcarrier_Freq_Divided_by);

    % Whether_Basic_Pilot__OR__PAPR_improved_Pilot = true;
    Whether_Basic_Pilot__OR__PAPR_improved_Pilot = false;
    N_cp__that_is__Cyclic_Prefix_Length = N / 1;

    % Whether_Pilot_Use_all_freq__OR__High_freq_only = true;
    Whether_Pilot_Use_all_freq__OR__High_freq_only = false;

    Sampling_Freq = 48000;

    Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 = true;
    % Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 = false;
    T_p__that_is_preamble_1_length_Unit_is_Sample = 1000;
    omega = 10;
    mu = 0.1;
    tp = (1:T_p__that_is_preamble_1_length_Unit_is_Sample).';
    Preamble_1_Chirp = cos(omega * tp + (mu * tp.^2 / 2));
    if Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 == true
        Preamble = Preamble_1_Chirp;
    end

    buffer_for_recording_time_sec = 10;
    Recording_Time_Sec = ((T_p__that_is_preamble_1_length_Unit_is_Sample + (Total_OFDM_symbols_Number_that_is_including_Pilot) * (N + N_cp__that_is__Cyclic_Prefix_Length)) / Sampling_Freq) * (1) + buffer_for_recording_time_sec;
    fprintf('Recording_Time_Sec: %s\n', num2str(Recording_Time_Sec));

    save(Rx_Setting_MAT_path_and_File_Name, ...
        'Whether_NOT_Repetition_coding__OR__Repetition_How_Many', ...
        'Sampling_Freq', ...
        'Recording_Time_Sec', ...
        'Preamble', ...
        'OFDM_symbols_Number', ...
        'Total_OFDM_symbols_Number_that_is_including_Pilot', ...
        'Fixed_Img_Size', ...
        'Whether_PAPR_improved_inter_leaving__OR__NOT', ...
        'Modulation_Number', ...
        'N', ...
        'Subcarrier_Freq_Divided_by', ...
        'Whether_Basic_Pilot__OR__PAPR_improved_Pilot', ...
        'N_cp__that_is__Cyclic_Prefix_Length', ...
        'Whether_Pilot_Use_all_freq__OR__High_freq_only', ...
        'T_p__that_is_preamble_1_length_Unit_is_Sample');

    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['# Setting 종료 시각: ', char(end_time)]);
    elapsed_time = end_time - start_time;
    disp(['# Setting 중 총 경과 시간[s]: ', char(elapsed_time)]);    
elseif Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 3 ...
        || Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 4 ...
        || Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 1
    close all;
    clear Estimated_Img;

    if Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 3
        start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# MAT에서 Rx 시작 시각: ', char(start_time)]);
        disp('# MAT에서 Rx을 실행합니다.');
        try
            load(Save_and_Load_Tx_signal_MAT_Path_and_File_Name, 'Tx_signal');
        catch ME
            fprintf('Tx_signal.MAT 로드 실패: %s\n', ME.message);
        end
        end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# MAT에서 Rx 종료 시각: ', char(end_time)]);
        elapsed_time = end_time - start_time;
        disp(['# MAT에서 Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
    elseif Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 4
        start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# WAV에서 Rx 시작 시각: ', char(start_time)]);
        disp('# WAV에서 Rx을 실행합니다.');
        try
            Tx_signal = audioread(Tx_Signal_Wav_Rx_Path_and_File_Name);
        catch ME
            fprintf('Tx_signal.WAV 로드 실패: %s\n', ME.message);
        end
        end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# WAV에서 Rx 종료 시각: ', char(end_time)]);
        elapsed_time = end_time - start_time;
        disp(['# WAV에서 Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
    elseif Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 1
        start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Rx 시작 시각: ', char(start_time)]);
        disp('# Rx을 실행합니다.');

        % 수신 시작
        release(devicereader);
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

        end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Rx 종료 시각: ', char(end_time)]);
        elapsed_time = end_time - start_time;
        disp(['# Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
    end
    Estimated_Img = Rx_Step_2_Get_Estimated_Img(Rx_Setting_MAT_path_and_File_Name, Tx_signal);
end