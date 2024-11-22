function Tx_Step_9_Save_Tx_signal_MAT_and_WAV(Tx_signal, Sampling_Freq, Save_and_Load_Tx_signal_MAT_Path_and_File_Name, Save_and_Load_Wav_Path_and_File_Name, Whether_Use_Base_WAV__OR__NOT, Base_WAV_Path_and_File_Name, Amplitude_ratio_Base_WAV_over_Tx_signal_WAV)

    if Whether_Use_Base_WAV__OR__NOT == true
        [Base_WAV_siganl, Sampling_Freq_Previous_of_BASE_WAV] = audioread(Base_WAV_Path_and_File_Name);
        if size(Base_WAV_siganl, 2) > 1
            Base_WAV_siganl_mono = mean(Base_WAV_siganl, 2);
        else
            Base_WAV_siganl_mono = Base_WAV_siganl;
        end
        Base_WAV_siganl_mono_resampled = resample(Base_WAV_siganl_mono, Sampling_Freq, Sampling_Freq_Previous_of_BASE_WAV);
        Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) = Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) + (Tx_signal ./ Amplitude_ratio_Base_WAV_over_Tx_signal_WAV);
        Tx_signal = Base_WAV_siganl_mono_resampled;
    else
        Tx_signal = normalize(Tx_signal, 'range', [-1 1]);
    end
    audiowrite(Save_and_Load_Wav_Path_and_File_Name, Tx_signal, Sampling_Freq);
    save(Save_and_Load_Tx_signal_MAT_Path_and_File_Name, 'Tx_signal');
end