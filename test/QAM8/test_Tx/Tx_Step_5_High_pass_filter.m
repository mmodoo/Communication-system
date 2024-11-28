function Tx_signal = Tx_Step_5_High_pass_filter(Tx_signal, Cut_off_Freq_normalised)
    
    % 필터 차수 설정 (예: 100차)
    filter_order = 4;
    
    % FIR 고역 통과 필터 설계
    % b = fir1(order, cut_off_freq_normalised, 'high');
    [b, a] = butter(filter_order, Cut_off_Freq_normalised, 'high');
    
    % 필터의 주파수 응답 확인 (선택 사항)
    % fvtool(b, 1);
    
    % Tx_signal에 필터 적용
    % filtered_signal = filter(b, 1, Tx_signal);
    % Tx_signal = filter(b, 1, Tx_signal);
    Tx_signal = filtfilt(b, a, Tx_signal);
end