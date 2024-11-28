function [Tx_signal, Sampling_Freq, Recording_time_Sec] = Tx_Step_10_Calculate_Recording_time_Sec_from_Tx_signal_WAV(Save_and_Load_WAV_Path_and_File_Name)

    [Tx_signal, Sampling_Freq] = audioread(Save_and_Load_WAV_Path_and_File_Name);
    % Tx_signal = normalize(Tx_signal, 'range', [-2 2]);
    Recording_time_Sec = length(Tx_signal) / Sampling_Freq;
end