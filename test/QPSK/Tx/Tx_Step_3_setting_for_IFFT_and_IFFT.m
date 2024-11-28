function Symbols_IFFTed_at_time = Tx_Step_3_setting_for_IFFT_and_IFFT(PSK_modulated_symbols, N, Subcarrier_Freq_Divided_by, OFDM_symbols_Number)

    Symbols_mapped_to_Subcarrier_Freq = {};
    for i = 1:OFDM_symbols_Number
        symbols_temp_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
        symbols_temp_2 = PSK_modulated_symbols(1:floor(N/Subcarrier_Freq_Divided_by)/2);
        symbols_temp_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
        PSK_modulated_symbols = PSK_modulated_symbols(1+floor(N/Subcarrier_Freq_Divided_by)/2:end);
        Symbols_mapped_to_Subcarrier_Freq{end+1} = [symbols_temp_1; symbols_temp_2; symbols_temp_3];
        % Symbols_mapped_to_Subcarrier_Freq{end + 1} = [zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by))), 1); PSK_modulated_symbols(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by))*(i-1)+1 : N*(1-(1/Subcarrier_Freq_Divided_by))*i)];
        Symbols_mapped_to_Subcarrier_Freq{end} = [0; Symbols_mapped_to_Subcarrier_Freq{end}; 0; flip(conj(Symbols_mapped_to_Subcarrier_Freq{end}))];
    end

    Symbols_IFFTed_at_time = {};
    for i = 1:length(Symbols_mapped_to_Subcarrier_Freq)
        Symbols_IFFTed_at_time{end + 1} = ifft(Symbols_mapped_to_Subcarrier_Freq{i}, N+2) * sqrt(N+2);
    end
end