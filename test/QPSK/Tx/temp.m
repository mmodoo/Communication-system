% function Recording_time_Sec = Tx_Step_7_Calculate_Recording_time_Sec_and_Save_MAT(Tx_signal, Sampling_Freq, Save_and_Load_Recording_time_Sec_MAT_Path_and_File_Name)
function Recording_time_Sec = Tx_Step_7_Calculate_Recording_time_Sec(Tx_signal, Sampling_Freq)

    Recording_time_Sec = length(Tx_signal / Sampling_Freq);
    % save(Save_and_Load_Recording_time_Sec_MAT_Path_and_File_Name, 'Recording_time_Sec');
    
end