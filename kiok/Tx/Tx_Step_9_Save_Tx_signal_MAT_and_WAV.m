function Tx_Step_9_Save_Tx_signal_MAT_and_WAV(Tx_signal, Sampling_Freq, Save_and_Load_Tx_signal_MAT_Path_and_File_Name, Save_and_Load_WAV_Path_and_File_Name, Whether_Use_Base_WAV__OR__NOT, Base_WAV_Path_and_File_Name, Amplitude_ratio_Base_WAV_over_Tx_signal_WAV)

    if Whether_Use_Base_WAV__OR__NOT == true
        [Base_WAV_siganl, Sampling_Freq_Previous_of_BASE_WAV] = audioread(Base_WAV_Path_and_File_Name);
        if size(Base_WAV_siganl, 2) > 1
            Base_WAV_siganl_mono = mean(Base_WAV_siganl, 2);
        else
            Base_WAV_siganl_mono = Base_WAV_siganl;
        end
        Base_WAV_siganl_mono_resampled = resample(Base_WAV_siganl_mono, Sampling_Freq, Sampling_Freq_Previous_of_BASE_WAV);
        % figure;
        % plot(Base_WAV_siganl_mono_resampled);
        % figure;
        % plot(Tx_signal);
        % Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled .* 2;
        % figure;
        % plot(Base_WAV_siganl_mono_resampled);

        % Base_WAV_range = max(abs(Base_WAV_siganl_mono));
        % Tx_signal_range = max(abs(Tx_signal));
        % desired_Base_WAV_range = Tx_signal_range * Amplitude_ratio_Base_WAV_over_Tx_signal_WAV;
        % scaling_factor = desired_Base_WAV_range / Base_WAV_range;
        % Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled * scaling_factor;
        % Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) = Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) + (Tx_signal ./ Amplitude_ratio_Base_WAV_over_Tx_signal_WAV);
        % Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled ./ max(abs(Base_WAV_siganl_mono_resampled));

        Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled ./ (2 * max(abs(Base_WAV_siganl_mono_resampled)));
        Tx_signal = Tx_signal ./ (2 * max(abs(Tx_signal)));
        Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) = Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) + (Tx_signal ./ Amplitude_ratio_Base_WAV_over_Tx_signal_WAV);
        Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled ./ (max(abs(Base_WAV_siganl_mono_resampled)));
        % Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) = Base_WAV_siganl_mono_resampled(1:length(Tx_signal)) + (Tx_signal ./ max(abs(Tx_signal)));
        % Base_WAV_siganl_mono_resampled = Base_WAV_siganl_mono_resampled ./ max(abs(Base_WAV_siganl_mono_resampled));
        Tx_signal = Base_WAV_siganl_mono_resampled;
    else
        Tx_signal = normalize(Tx_signal, 'range', [-1 1]);
    end
    audiowrite(Save_and_Load_WAV_Path_and_File_Name, Tx_signal, Sampling_Freq);
    save(Save_and_Load_Tx_signal_MAT_Path_and_File_Name, 'Tx_signal');
end