function Tx_Step_9_Save_Tx_signal_MAT_and_WAV(Tx_signal, Sampling_Freq, Save_and_Load_Tx_signal_MAT_Path_and_File_Name, Save_and_Load_Wav_Path_and_File_Name, Wav_Path_and_File_Name)

    save(Save_and_Load_Tx_signal_MAT_Path_and_File_Name, 'Tx_signal');

    Tx_signal = normalize(Tx_signal, 'range', [-1 1]);
    audiowrite(Save_and_Load_Wav_Path_and_File_Name, Tx_signal, Sampling_Freq);

    % figure;
    % plot(Tx_signal);
end