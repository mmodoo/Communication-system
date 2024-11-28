function Tx_Step_8_Plot_Tx_signal_in_many_ways(Tx_signal, Sampling_Freq)

    figure;
    plot(Tx_signal);

    % 시간 축을 초 단위로 변환하여 Tx_signal을 플로팅합니다.
    t = (0:length(Tx_signal)-1) / Sampling_Freq; % 시간 축 생성
    figure;
    plot(t, Tx_signal);
    xlabel('시간 (초)');
    ylabel('신호 진폭');
    title('Tx\_signal의 시간 영역 파형');


    % Tx_signal의 주파수 스펙트럼을 계산하고 플로팅합니다.
    N_fft = length(Tx_signal);
    f = (-N_fft/2:N_fft/2-1)*(Sampling_Freq/N_fft); % 주파수 축 생성
    Tx_signal_fft = fftshift(fft(Tx_signal));
    
    figure;
    plot(f, abs(Tx_signal_fft));
    xlabel('주파수 (Hz)');
    ylabel('스펙트럼 진폭');
    title('tx\_signal의 주파수 스펙트럼');
    xlim([0, Sampling_Freq/2]); % 양의 주파수 성분만 표시


    figure;
    spectrogram(Tx_signal, 256, 128, 256, Sampling_Freq, 'yaxis');
end