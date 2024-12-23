% function Tx_main(Whether_Only_Tx__OR__Set_Save_Tx_signal_MAT_and_Tx_signal_WAV)
    Whether_Only_Tx__OR__Set_Save_Tx_signal_MAT_and_Tx_signal_WAV = true;
    % Whether_Only_Tx__OR__Set_Save_Tx_signal_MAT_and_Tx_signal_WAV = false;
    % 컴퓨터 바뀔 때마다 체크해야할 부분 == CTRL + F "여기 반드시 확인"
    % 여기 반드시 확인
    Save_and_Load_WAV_Path_and_File_Name = "C:\Users\user\Desktop\졸업드가자\통신시스템설계\8QAM\Tx\Tx_signal.WAV";
    
    if Whether_Only_Tx__OR__Set_Save_Tx_signal_MAT_and_Tx_signal_WAV == false
        clearvars -except Save_and_Load_WAV_Path_and_File_Name Whether_Only_Tx__OR__Set_Save_Tx_signal_MAT_and_Tx_signal_WAV;
        % close all;
        clc;
        
        start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Set_Save_Tx_signal_MAT_and_Tx_signal_WAV 시작 시각: ', char(start_time)]);
        disp('# SETTING, MAT 및 WAV 파일 제조 및 저장을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');
    
        disp('## Parameter들을 설정하겠습니다.');
    
        % Tx_Step_1_imread_resize_gray_mono_repetition
        % Whether_NOT_Repetition_coding__OR__Repetition_How_Many = 1;
        Whether_NOT_Repetition_coding__OR__Repetition_How_Many = 3;
        % 여기 반드시 확인
        Img_Path_and_File_Name = "C:\Users\user\Desktop\졸업드가자\통신시스템설계\8QAM\Tx\IMG.PNG";
        Fixed_Img_Size = [128 128];
        
        Whether_PAPR_improved_inter_leaving__OR__NOT = true;
        % Whether_PAPR_improved_inter_leaving__OR__NOT = false;
    
        % Tx_Step_2_PSK_modulation
        Modulation_Number = 8;
        % N = 288; % 6의 배수
        N = 384 * 2;
        Subcarrier_Freq_Divided_by = 6;
    
        % Tx_Step_3_setting_for_IFFT_and_IFFT
    
        % Tx_Step_4_Add_Cyclic_Prefix_and_Pilot
        % Whether_Basic_Pilot__OR__PAPR_improved_Pilot = true;
        Whether_Basic_Pilot__OR__PAPR_improved_Pilot = false;
        N_cp__that_is__Cyclic_Prefix_Length = N / 1;
    
        % Whether_Pilot_Use_all_freq__OR__High_freq_only = true;
        Whether_Pilot_Use_all_freq__OR__High_freq_only = false;
        
        % Tx_Step_5_High_pass_filter
        Whether_High_pass_filter__OR__NOT = true;
        % Whether_High_pass_filter__OR__NOT = false;
        Sampling_Freq = 48000;
        Cut_off_Freq = (Sampling_Freq / 2) * (1 - 2 * (1/Subcarrier_Freq_Divided_by)); % 48000Hz, 6 경우에 16000HZ
        Cut_off_Freq_normalised = Cut_off_Freq / (Sampling_Freq / 2);
    
        % Tx_Step_6_Add_Preamble
        Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 = true;
        % Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 = false;
        % T_p__that_is_preamble_1_length_Unit_is_Sample = 1000;
        % T_p__that_is_preamble_1_length_Unit_is_Sample = 0;
        T_p_Sec = 1;
        omega = 10;
        mu = 0.1;
        % tp = (1:T_p__that_is_preamble_1_length_Unit_is_Sample).';
        tp = (0:1/Sampling_Freq:T_p_Sec).';
        % Preamble_1_Chirp = cos(omega * tp + (mu * tp.^2 / 2));
        f_preamble = 18000;
        % Preamble_1_Chirp = sin(2 * pi * f_preamble * tp);
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
    
        % Tx_Step_6_1_Add_Preamble_sin_wave   @ Step_9
        Whether_Add_Preamble_sine_wave = true;
        % Whether_Add_Preamble_sine_wave = false;
        duration = 2;
        f_Preamble_sine_wave = 19000;
        t_Preamble_sine_wave = (0:1/Sampling_Freq:duration).';
        Preamble_sine_wave = sin(2 * pi * f_Preamble_sine_wave * t_Preamble_sine_wave);
       
        % Tx_Step_7_Calculate_Recording_time_Sec
    
        % Tx_Step_8_Plot_Tx_signal_in_many_ways
        % Whether_Plot_Tx_signal__OR__NOT = true;
        Whether_Plot_Tx_signal__OR__NOT = false;
    
        % Tx_Step_9_Save_Tx_signal_MAT_and_WAV
        % 여기 반드시 확인
        % Whether_Use_Base_WAV__OR__NOT = true;
        Whether_Use_Base_WAV__OR__NOT = false;
        Save_and_Load_Tx_signal_MAT_Path_and_File_Name = "C:\Users\user\Desktop\졸업드가자\통신시스템설계\8QAM\Tx\Tx_signal.MAT";
        Base_WAV_Path_and_File_Name = "C:\Users\user\Desktop\졸업드가자\통신시스템설계\8QAM\Tx\Base_WAV\Base_6.WAV";
        Amplitude_ratio_Base_WAV_over_Tx_signal_WAV = 1;
        Whether_Use_Base_WAV_Changing_through_minute = true;
        % Whether_Use_Base_WAV_Changing_through_minute = false;
        if Whether_Use_Base_WAV_Changing_through_minute == true
            % 현재 시간의 '분'을 숫자로 추출
            current_minute = minute(start_time);  % 53이 저장됨
            
            % 20으로 나눈 나머지를 숫자로 저장
            remainder_num = mod(current_minute, 20) + 1;  % 13이 저장됨
            
            % 나머지를 문자열로 변환
            remainder_str = num2str(remainder_num);   % '13'이 저장됨
            
            % 파일 경로에 나머지 값 추가
            Base_WAV_Path_and_File_Name = "C:\Users\user\Desktop\졸업드가자\종합설계\테스트\Base_WAV\Base_" + remainder_str + ".WAV";
            % 결과: "C:\Users\user\Desktop\졸업드가자\종합설계\테스트\Base_WAV\Base13.WAV"
        end

        Whether_median_filter = true;
        Whether_median_filter = false;
        
        
    
        
        % 이제 Step_1부터 본격적으로 시작
    
        Binarised_img_column_vector = Tx_Step_1_imread_resize_gray_mono_repetition(Img_Path_and_File_Name, Fixed_Img_Size, Whether_NOT_Repetition_coding__OR__Repetition_How_Many, Whether_PAPR_improved_inter_leaving__OR__NOT);
    
        [PSK_modulated_symbols, OFDM_symbols_Number, Total_OFDM_symbols_Number_that_is_including_Pilot] = Tx_Step_2_PSK_modulation(Binarised_img_column_vector, Modulation_Number, N, Subcarrier_Freq_Divided_by);
    
        Symbols_IFFTed_at_time = Tx_Step_3_setting_for_IFFT_and_IFFT(PSK_modulated_symbols, N, Subcarrier_Freq_Divided_by, OFDM_symbols_Number);
    
        Tx_signal = Tx_Step_4_Add_Cyclic_Prefix_and_Pilot(Symbols_IFFTed_at_time, N, N_cp__that_is__Cyclic_Prefix_Length, Subcarrier_Freq_Divided_by, Whether_Basic_Pilot__OR__PAPR_improved_Pilot, Whether_Pilot_Use_all_freq__OR__High_freq_only);
    
        if Whether_High_pass_filter__OR__NOT == true
            Tx_signal = Tx_Step_5_High_pass_filter(Tx_signal, Cut_off_Freq_normalised);
        end
    
        Tx_signal = Tx_Step_6_Add_Preamble(Tx_signal, Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2, Preamble_1_Chirp);
    
        Recording_time_Sec = Tx_Step_7_Calculate_Recording_time_Sec(Tx_signal, Sampling_Freq);
        disp(['### Recording_time_Sec: ' num2str(Recording_time_Sec) ' 초']);
    
        if Whether_Plot_Tx_signal__OR__NOT == true
            Tx_Step_8_Plot_Tx_signal_in_many_ways(Tx_signal, Sampling_Freq);
        end
    
        Tx_Step_9_Save_Tx_signal_MAT_and_WAV(Tx_signal, Sampling_Freq, Save_and_Load_Tx_signal_MAT_Path_and_File_Name, Save_and_Load_WAV_Path_and_File_Name, Whether_Use_Base_WAV__OR__NOT, Base_WAV_Path_and_File_Name, Amplitude_ratio_Base_WAV_over_Tx_signal_WAV, Whether_Add_Preamble_sine_wave, Preamble_sine_wave);
    
        end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Set_Save_Tx_signal_MAT_and_Tx_signal_WAV 종료 시각: ', char(end_time)]);
        elapsed_time = end_time - start_time;
        disp(['# Set_Save_Tx_signal_MAT_and_Tx_signal_WAV 중 총 경과 시간[s]: ', char(elapsed_time)]);
        
    else
        % close all;
        clc;
        start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Only_Tx 시작 시각: ', char(start_time)]);
        disp('# Tx_signal.WAV를 재생하여 통신합니다.');
        [Tx_signal, Sampling_Freq, Recording_time_Sec] = Tx_Step_10_Calculate_Recording_time_Sec_from_Tx_signal_WAV(Save_and_Load_WAV_Path_and_File_Name);
        disp(['### Recording_time_Sec: ' num2str(Recording_time_Sec) ' 초']);
    
        Tx_Step_11_play_WAV(Tx_signal, Sampling_Freq, Recording_time_Sec);
    
        end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
        disp(['# Only_Tx 종료 시각: ', char(end_time)]);
        elapsed_time = end_time - start_time;
        disp(['# Only_Tx 중 총 경과 시간[s]: ', char(elapsed_time)]);
    end
% end