function [PSK_modulated_symbols, OFDM_symbols_Number, Total_OFDM_symbols_Number_that_is_including_Pilot] = Tx_Step_2_PSK_modulation(Binarised_img_column_vector, Modulation_Number, N, Subcarrier_Freq_Divided_by)

    int_symbol = bit2int(Binarised_img_column_vector, 2);
    PSK_modulated_symbols = pskmod(int_symbol, Modulation_Number, pi/4);

    symbol_length = length(PSK_modulated_symbols);

    OFDM_symbols_Number = symbol_length / (N / (2 * Subcarrier_Freq_Divided_by));

    Total_OFDM_symbols_Number_that_is_including_Pilot = OFDM_symbols_Number + (OFDM_symbols_Number / 4);
end