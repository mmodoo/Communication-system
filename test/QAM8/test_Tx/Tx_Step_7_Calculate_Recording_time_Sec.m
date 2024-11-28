function Recording_time_Sec = Tx_Step_7_Calculate_Recording_time_Sec(Tx_signal, Sampling_Freq)

    Recording_time_Sec = length(Tx_signal) / Sampling_Freq;
    
end