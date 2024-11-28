% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 1;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 2;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 3;
% Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 4;
Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV = 5; % Listening state 
Whether_there_was_an_error = false;
% Whether_OCR = true;
Whether_OCR = false;
% 컴퓨터 바뀔 때마다 체크해야할 부분 == CTRL + F "여기 반드시 확인"
% 여기 반드시 확인 
Save_and_Load_Tx_signal_MAT_Path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\8QAM\Tx\Tx_signal.MAT";
Tx_Signal_Wav_Rx_Path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\8QAM\Tx\Tx_signal.WAV";
Rx_Setting_MAT_path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\8QAM\Rx\Rx_Setting.MAT";
Base_WAV_Path_and_File_Name = "C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\최종프로젝트\8QAM\Base_WAV\Base_1.WAV";
Listening_State_time_Sec = 1;

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
        'T_p__that_is_preamble_1_length_Unit_is_Sample', ...
        'Whether_median_filter', ...
        'Whether_Use_Base_WAV_Changing_through_minute'};
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
        Save_and_Load_Tx_signal_MAT_Path_and_File_Name ...
        Base_WAV_Path_and_File_Name;
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

    Modulation_Number = 8;
    N = 384; % 6의 배수
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
    T_p__that_is_preamble_1_length_Unit_is_Sample = 10000;
    omega = 10;
    mu = 0.1;
    % tp = (1:T_p__that_is_preamble_1_length_Unit_is_Sample).';
    f_preamble = 18000;
    T_p_Sec = 2;
    tp = (0:1/Sampling_Freq:T_p_Sec).';
    % Preamble_1_Chirp = cos(omega * tp + (mu * tp.^2 / 2));
    Preamble_1_Chirp = sin(2 * pi * f_preamble * tp);

    % 파라미터 설정
    f_start = 18000;        % 시작 주파수 (Hz)
    f_end = 19000;         % 종료 주파수 (Hz)
    T = 2;                 % 지속 시간 (초)
    fs = Sampling_Freq;    % 샘플링 주파수 (Hz)
    
    % 시간 벡터 생성
    t = (0:1/fs:T).';
    
    % Chirp 신호 생성
    % 순간 주파수 = f_start + (f_end-f_start)t/T
    % 위상 = 2π * ∫f(t)dt = 2π * (f_start*t + (f_end-f_start)t^2/(2T))
    phase = 2*pi * (f_start*t + ((f_end-f_start)/(2*T))*t.^2);
    Preamble_1_Chirp = cos(phase);
    if Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 == true
        Preamble = Preamble_1_Chirp;
    end

    buffer_for_recording_time_sec = 5;
    Recording_Time_Sec = ((T_p__that_is_preamble_1_length_Unit_is_Sample + (Total_OFDM_symbols_Number_that_is_including_Pilot) * (N + N_cp__that_is__Cyclic_Prefix_Length)) / Sampling_Freq) * (1) + buffer_for_recording_time_sec;
    fprintf('Recording_Time_Sec: %s\n', num2str(Recording_Time_Sec));

    Whether_Use_Base_WAV_Changing_through_minute = true;
    % Whether_Use_Base_WAV_Changing_through_minute = false;

    % Whether_median_filter = true;
    Whether_median_filter = false;

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
        'T_p__that_is_preamble_1_length_Unit_is_Sample', ...
        'Whether_median_filter', ...
        'Whether_Use_Base_WAV_Changing_through_minute');

    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['# Setting 종료 시각: ', char(end_time)]);
    elapsed_time = end_time - start_time;
    disp(['# Setting 중 총 경과 시간[s]: ', char(elapsed_time)]);    
elseif Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 3 ...
        || Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 4 ...
        || Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 1 ...
        || Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 5
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
    else
        % 수신 시작
        % release(devicereader);
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
        
        if Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 1
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
    
            end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
            disp(['# Rx 종료 시각: ', char(end_time)]);
            elapsed_time = end_time - start_time;
            disp(['# Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
        elseif Whether_Only_Rx__OR__Set__OR__Load_Rx_signal_MAT__OR__WAV == 5
            t = 0:(1/Sampling_Freq):Listening_State_time_Sec;
            F_wake_up_signal = 19000;
            Wake_up_signal = sin(2 * pi * F_wake_up_signal * t);
            % Wake_up_threshold = max(xcorr(Wake_up_signal, Wake_up_signal)) / 2;
            Wake_up_threshold = 50;
            start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
            disp(['# Listening State 시작 시각: ', char(start_time)]);
            while true
                start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
                disp(['## ', char(start_time), ' Wake_up_signal이 있는지 듣고 있습니다.']);
                devicereader = audioDeviceReader(Sampling_Freq);
                setup(devicereader);
                Tx_signal = [];
                tic;
                while toc < Listening_State_time_Sec
                    elapsed = 0;
                    acquiredAudio = devicereader();
                    Tx_signal = [Tx_signal; acquiredAudio];
                    
                    
                    % % 1초마다 진행 상태 업데이트
                    % if floor(toc) > elapsed
                    %     elapsed = floor(toc);
                    %     fprintf('\r녹음 진행: %d초 / %d초 (%.1f%%)', ...
                    %     elapsed, ceil(Listening_State_time_Sec), (elapsed/Listening_State_time_Sec)*100);
                    % end
                end
                Xcorr_for_Checking_Whether_threshold = max(xcorr(Tx_signal, Wake_up_signal));
                Xcorr_for_Checking_Whether_threshold
                Wake_up_threshold
                if Xcorr_for_Checking_Whether_threshold > Wake_up_threshold
                    % disp('아 잘 잤다.');
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
                    try
                        Estimated_Img = Rx_Step_2_Get_Estimated_Img(Rx_Setting_MAT_path_and_File_Name, Tx_signal, Base_WAV_Path_and_File_Name, Whether_median_filter, Whether_OCR, Whether_Use_Base_WAV_Changing_through_minute, Preamble);
                    catch ME
                        fprintf('이미지 추정에 실패하였습니다. 신호가 오긴 한 건가요?: %s\n', ME.message);
                    end
            
                    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
                    disp(['# Rx 종료 시각: ', char(end_time)]);
                    elapsed_time = end_time - start_time;
                    disp(['# Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
                end
            end
            % start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
            % disp(['# Listening State 시작 시각: ', char(start_time)]);
            % devicereader = audioDeviceReader(Sampling_Freq);
            % setup(devicereader);
            % disp(['총 녹음 시간: ', num2str(Recording_Time_Sec), '초']);
            % Tx_signal = [];
            % elapsed = 0;
            % tic;
            % 
            % while toc < Recording_Time_Sec  % recording_time_sec 동안 녹음
            %     acquiredAudio = devicereader();
            %     Tx_signal = [Tx_signal; acquiredAudio];
            % 
            %     % 1초마다 진행 상태 업데이트
            %     if floor(toc) > elapsed
            %         elapsed = floor(toc);
            %         fprintf('\r녹음 진행: %d초 / %d초 (%.1f%%)', ...
            %             elapsed, ceil(Recording_Time_Sec), (elapsed/Recording_Time_Sec)*100);
            %     end
            % end
            % fprintf('\n');  % 줄바꿈 추가
            % disp('Recording Completed');
            % 
            % end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
            % disp(['# Rx 종료 시각: ', char(end_time)]);
            % elapsed_time = end_time - start_time;
            % disp(['# Rx 중 총 경과 시간[s]: ', char(elapsed_time)]); 
        end
    end
    Estimated_Img = Rx_Step_2_Get_Estimated_Img(Rx_Setting_MAT_path_and_File_Name, Tx_signal, Base_WAV_Path_and_File_Name, Whether_median_filter, Whether_OCR, Whether_Use_Base_WAV_Changing_through_minute, Preamble);
end

