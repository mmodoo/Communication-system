TX_WITHOUT_SETTING = true; % 최초 실행 or parameter 변경시 무조건 얘는 주석 아래는 주석해제
TX_WITHOUT_SETTING = false; % SETTING 한 번 이상 실행했고 para 변경 없으면 얘 주석 위 주석해제

if TX_WITHOUT_SETTING == false
    clear all;
    close all;
    clc;
    TX_WITHOUT_SETTING = false;
    disp('### SETTING을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');

    % 주요 parameter setting
    
    sampling_freq = 40000; % Nyquist freq = sampling_freq /x` 2 임을 잊지 말아야
    
    WHETHER_REPETITION_CODING = true; % repetition coding을 하려면 아래 코드를 주석하고 얘를 주석 해제
    % WHETHER_REPETITION_CODING = false; % repetition coding X면 위를 주석, 얘는 주석 해제
    repetition_factor = 3;
    
    % IFFT의 N을 세팅. sub-carrier 수는 N/2 - 1
    N = 128;
    
    % cyclic_prefix 길이
    N_cp = N / 4;
    
    % 이미지 경로
    img_path = '\Lena_color.png'; % 상대경로로서, 이 코드(main_tx.m)의 상위 폴더에 있는 Lena_color.png
    
    % 이미지 사이즈를 어떤 크기로 고정할 것인가
    fixed_img_size = [128 128];
    
    % Preamlbe 길이 (단위: sample)
    Tp = 1000;
    
    % Preamble Chirp 계수 설정
    omega = 10;
    mu = 0.1;
    tp = (1:Tp).';
    preamble = cos(omega * tp + mu * tp.^2 / 2);
    
    % TX_WITHOUT_SETTING == false라면, 즉 SETTING을 실행하면 recording time이 아래 부분에서 계산된다

    % 주요 parameter setting 부분 끝.

    disp('### tx_signal을 만들겠습니다.');
    % imshow(img_path); % 상대 경로에 있는 이미지 잘 불러오는지 테스트

    disp('image를 읽어옵니다.');
    img = imread(img_path);

    disp('image를 고정된 크기로 줄입니다.');
    resized_img = imresize(img, fixed_img_size);

    disp('image를 gray-scale로 바꿉니다.');
    gray_img = rgb2gray(resized_img);

    disp('image를 monochrome으로 바꿉니다.');
    binarised_img = imbinarize(gray_img);

    disp('bit로 전송하기 위해, 128x128x을 단일 Column vector로 형식을 바꿉니다.');
    binarised_img_column_vector = binarised_img(:);

    if WHETHER_REPETITION_CODING == true
        disp('repetition coding을 적용합니다.');
        binarised_img_column_vector = repelem(binarised_img_column_vector, repetition_factor);
    end

    disp('PSK modulation을 진행합니다.');
    bpsk_modulated_symbols = 2 * binarised_img_column_vector - 1;

    disp('전체 symbol의 개수를 구합니다.');
    symbol_length = length(bpsk_modulated_symbols);

    disp('ofdm_symbols 개수를 구합니다.');
    num_ofdm_symbols = symbol_length / (N / 2);

    disp('전체 OFDM 블록 개수를 구합니다. 이것은 num_ofdm_symbols에, pilot 신호의 개수를 더한 것입니다.');
    % pilot 신호는 4번마다 나오기 때문에 num_ofdm_symbols / 4
    total_OFDM_block_number = num_ofdm_symbols + (num_ofdm_symbols / 4);
    
    disp('이제 psk_modulated_symbols를 각 subcarrier 주파수의 진폭에 싣습니다.');
    symbols_mapped_to_subcarrier_freq = {};
    for i = 1:num_ofdm_symbols
        symbols_mapped_to_subcarrier_freq{end + 1} = [0; bpsk_modulated_symbols(N/2*(i-1)+1 : N/2*i)];
        symbols_mapped_to_subcarrier_freq{end} = [symbols_mapped_to_subcarrier_freq{end}; flip(symbols_mapped_to_subcarrier_freq{end}(2:end-1))];
    end

    disp('이제 subcarrier 주파수에 mapping된 신호를 time-domain으로 IFFT(IDFT 상위호환) 하겠습니다.');
    symbols_IFFTed_time = {};
    for i = 1:length(symbols_mapped_to_subcarrier_freq)
        symbols_IFFTed_time{end + 1} = ifft(symbols_mapped_to_subcarrier_freq{i}, N) * sqrt(N);
    end

    disp('cyclic prefix를 집어넣습니다.');
    for i = 1:length(symbols_IFFTed_time)
        symbols_IFFTed_time{i} = [symbols_IFFTed_time{i}(end - N_cp + 1 : end); symbols_IFFTed_time{i}];
    end

    disp('Pilot signal을 주파수 영역에서 생성한 후, IFFT를 통해 time-domain으로 옮깁니다.');
    pilot_freq = ones(N, 1);
    pilot_time = ifft(pilot_freq) * sqrt(N);
    pilot_time = [pilot_time(end - N_cp + 1 : end); pilot_time];

    disp('preamble, pilot_time, symbols_IFFTed_time 순으로 tx_signal을 구성합니다.');
    tx_signal = [preamble; pilot_time];
    for i = 1 : length(symbols_IFFTed_time)
        tx_signal = [tx_signal; symbols_IFFTed_time{i}];
        if rem(i, 4) == 0 && i ~= length(symbols_IFFTed_time)
            tx_signal = [tx_signal; pilot_time];
        end
    end

    disp('recording_time을 계산합니다.');
    recording_time_sec = (Tp + (total_OFDM_block_number) * (N + N_cp)) / sampling_freq

    disp('Tx SETTING이 완료되었습니다.');
    WHETHER_TX_SETTING_DONE = true;

    figure
    plot([0:1:length(tx_signal)-1], tx_signal, LineWidth=1.5);
    xlim([0 length(tx_signal) - 1]);
    ylim([-20 20]);
    xlabel('Time');
    ylabel('Amplitude');
    grid on;

    max_pwr = max(abs(tx_signal)) * max(abs(tx_signal))
    avg_pwr = mean(abs(tx_signal)) * mean(abs(tx_signal))

    PAPR = max_pwr / avg_pwr
end

% 키 입력 콜백 함수. 스피커 사용 도중 스페이스를 누르면 종료하기 위함입니다.
function keyPressCallback(evt, player)
    if strcmp(evt.Key, 'space')  % 스페이스바가 눌렸을 때
        if isplaying(player)
            stop(player);
            fprintf('\n재생이 중지되었습니다.\n');
        end
    end
end

if TX_WITHOUT_SETTING == true && WHETHER_TX_SETTING_DONE == true
    close all;
    clc;

    disp('### 만들어져 있는 tx_signal을 보내겠습니다.');

    % audioplayer 객체 생성
    player = audioplayer(tx_signal, sampling_freq);    

    % 키보드 입력 처리를 위한 figure 생성
    fig = figure('KeyPressFcn', @(src, evt)keyPressCallback(evt, player));
    set(fig, 'Name', 'Press Space to Stop Playback', 'NumberTitle', 'off');

    % 재생 시작
    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['tx 시작 시각: ', char(start_time)]);
    play(player);

    % 재생 상태 모니터링
    disp(['총 재생 시간: ', num2str(recording_time_sec), '초 (스페이스바를 누르면 중지됩니다)']);
    elapsed = 0;
    while isplaying(player)
        pause(1);
        elapsed = elapsed + 1;
        fprintf('\r진행 시간: %d초 / %d초 (%.1f%%)', ...
            elapsed, ceil(recording_time_sec), (elapsed/recording_time_sec)*100);

        if ~ishandle(fig)  % figure가 닫혔는지 확인
            stop(player);
            break;
        end
    end
    fprintf('\n');  % 줄바꿈을 추가하여 다음 출력이 깔끔하게 보이도록 함

    % 재생 종료 처리
    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp('.');
    disp(['tx 종료 시각: ', char(end_time)]);
    elapsed_time = end_time - start_time;
    disp(['tx 중 총 경과 시간[s]: ', char(elapsed_time)]);

    % figure 정리
    if ishandle(fig)
        close(fig);
    end
end

