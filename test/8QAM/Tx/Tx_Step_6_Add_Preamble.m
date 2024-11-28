function Tx_signal = Tx_Step_6_Add_Preamble(Tx_signal, Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2, Preamble_1_Chirp)

    if Whther_Only_Preamble_1_Chirp__OR__plus_Preamble_2 == true
        Preamble = Preamble_1_Chirp;
    else
    end

    Tx_signal = [Preamble; Tx_signal];
end