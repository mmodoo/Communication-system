% function [Tx_signal, Symbols_IFFTed_at_time]  = Tx_Step_4_Add_Cyclic_Prefix_and_Pilot(Symbols_IFFTed_at_time, N, N_cp__that_is__Cyclic_Prefix_Length, Subcarrier_Freq_Divided_by, Whether_Basic_Pilot__OR__PAPR_improved_Pilot, Whether_Pilot_Use_all_freq__OR__High_freq_only)
function Tx_signal = Tx_Step_4_Add_Cyclic_Prefix_and_Pilot(Symbols_IFFTed_at_time, N, N_cp__that_is__Cyclic_Prefix_Length, Subcarrier_Freq_Divided_by, Whether_Basic_Pilot__OR__PAPR_improved_Pilot, Whether_Pilot_Use_all_freq__OR__High_freq_only)
    
    for i = 1:length(Symbols_IFFTed_at_time)
        Symbols_IFFTed_at_time{i} = [Symbols_IFFTed_at_time{i}(end - N_cp__that_is__Cyclic_Prefix_Length + 1 : end); Symbols_IFFTed_at_time{i}];
    end

    if Whether_Basic_Pilot__OR__PAPR_improved_Pilot == true
        Pilot_freq = ones(N+2, 1);
    else
        rng('default');
        if Whether_Pilot_Use_all_freq__OR__High_freq_only == true
            temp_pilot = 2 * (randi([0, 1], N/2, 1) - 0.5);  % N/2 개만 생성
        else
            % temp_pilot_1 = zeros(3*N/8, 1);
            % temp_pilot_2 = 2 * (randi([0, 1], N/8, 1) - 0.5);
            % temp_pilot_1 = zeros((Subcarrier_Freq_Divided_by-1)*N/(2*Subcarrier_Freq_Divided_by), 1);
            % temp_pilot_2 = 2 * (randi([0, 1], N/(2*Subcarrier_Freq_Divided_by), 1) - 0.5);
            % temp_pilot_2 = [temp_pilot_1; temp_pilot_2];
            temp_pilot_1 = zeros(floor(N*(1-2*(1/Subcarrier_Freq_Divided_by)))/2, 1);
            temp_pilot_2 = 2 * (randi([0, 1], floor(N/Subcarrier_Freq_Divided_by)/2, 1) - 0.5);
            temp_pilot_3 = zeros(floor(N/Subcarrier_Freq_Divided_by/2), 1);
            temp_pilot = [temp_pilot_1; temp_pilot_2; temp_pilot_3];
        end
        Pilot_freq = [0; temp_pilot; 0; flip(conj(temp_pilot))];  % DC, 중간, 허미션 처리
    end
    
    Pilot_at_time = ifft(Pilot_freq) * sqrt(N+2);
    Pilot_at_time = [Pilot_at_time(end - N_cp__that_is__Cyclic_Prefix_Length + 1 : end); Pilot_at_time];

    Tx_signal = [Pilot_at_time];
    for i = 1 : length(Symbols_IFFTed_at_time)
        Tx_signal = [Tx_signal; Symbols_IFFTed_at_time{i}];
        if rem(i, 4) == 0 && i ~= length(Symbols_IFFTed_at_time)
            Tx_signal = [Tx_signal; Pilot_at_time];
        end
    end
end