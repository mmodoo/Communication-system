TX_WITHOUT_SETTING = true; % 최초 실행 or parameter 변경시 무조건 얘는 주석 아래는 주석해제
% TX_WITHOUT_SETTING = false; % SETTING 한 번 이상 실행했고 para 변경 없으면 얘 주석 위 주석해제

if TX_WITHOUT_SETTING == false
    clear all;
    close all;
    clc;
    TX_WITHOUT_SETTING = false;
    % TX_THROUGH_MAT = true;
    TX_THROUGH_MAT = false;
    % WHETHER_BASIC_PILOT = true;
    WHETHER_BASIC_PILOT = false;
    WHETHER_INTERLEAVING = true;
    % WHETHER_INTERLEAVING = false;
    disp('### SETTING을 실행합니다. 기존에 저장된 모든 변수는 삭제되었습니다.');

    % 주요 parameter setting
    
    sampling_freq = 10000; % Nyquist freq = sampling_freq / 2 임을 잊지 말아야
    
    % WHETHER_REPETITION_CODING = true; % repetition coding을 하려면 아래 코드를 주석하고 얘를 주석 해제
    WHETHER_REPETITION_CODING = false; % repetition coding X면 위를 주석, 얘는 주석 해제
    repetition_factor = 3;
    
    % IFFT의 N을 세팅. sub-carrier 수는 N/2 - 1
    N = 16;
    
    % cyclic_prefix 길이
    N_cp = N / 4;
    
    % 이미지 경로
    img_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\실습 수업 매트랩 코드\week11B_HW\Lena_color.png';
    save_load_path_and_file_name = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\통신시스템설계 (신원재 교수님)\실습 수업 매트랩 코드\week11B_HW\tx_signal.mat';
    
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

    % Interleaving
    if WHETHER_INTERLEAVING == true
        rng('default'); % 재현성을 위해 난수 시드 초기화
        interleaver_order = randperm(length(binarised_img_column_vector)); % 길이만큼의 난수 순서 생성
        binarised_img_column_vector = binarised_img_column_vector(interleaver_order); % 무작위 순서로 비트 재배열
    end

    disp('QPSK modulation을 진행합니다.');
    % bpsk_modulated_symbols = 2 * binarised_img_column_vector - 1;
    int_symbol = bit2int(binarised_img_column_vector, 2);
    qpsk_modulated_symbols = pskmod(int_symbol, 4, pi/4);

    disp('전체 symbol의 개수를 구합니다.');
    symbol_length = length(qpsk_modulated_symbols);

    disp('ofdm_symbols 개수를 구합니다.');
    num_ofdm_symbols = symbol_length / (N / 4);

    disp('전체 OFDM 블록 개수를 구합니다. 이것은 num_ofdm_symbols에, pilot 신호의 개수를 더한 것입니다.');
    % pilot 신호는 4번마다 나오기 때문에 num_ofdm_symbols / 4
    total_OFDM_block_number = num_ofdm_symbols + (num_ofdm_symbols / 4);
    
    disp('이제 psk_modulated_symbols를 각 subcarrier 주파수의 진폭에 싣습니다.');
    symbols_mapped_to_subcarrier_freq = {};
    for i = 1:num_ofdm_symbols
        symbols_mapped_to_subcarrier_freq{end + 1} = [zeros(N/4, 1); qpsk_modulated_symbols(N/4*(i-1)+1 : N/4*i)];
        symbols_mapped_to_subcarrier_freq{end} = [0; symbols_mapped_to_subcarrier_freq{end}; 0; flip(conj(symbols_mapped_to_subcarrier_freq{end}))];
    end

    disp('이제 subcarrier 주파수에 mapping된 신호를 time-domain으로 IFFT(IDFT 상위호환) 하겠습니다.');
    symbols_IFFTed_time = {};
    for i = 1:length(symbols_mapped_to_subcarrier_freq)
        symbols_IFFTed_time{end + 1} = ifft(symbols_mapped_to_subcarrier_freq{i}, N+2) * sqrt(N+2);
    end
    symbols_IFFTed_time_archieved = symbols_IFFTed_time;

    disp('cyclic prefix를 집어넣습니다.');
    for i = 1:length(symbols_IFFTed_time)
        symbols_IFFTed_time{i} = [symbols_IFFTed_time{i}(end - N_cp + 1 : end); symbols_IFFTed_time{i}];
    end

    disp('Pilot signal을 주파수 영역에서 생성한 후, IFFT를 통해 time-domain으로 옮깁니다.');
    if WHETHER_BASIC_PILOT == true
        pilot_freq = ones(N + 2, 1);
    else
        rng('default');
        temp_pilot = 2 * (randi([0, 1], N/2, 1) - 0.5);  % N/2 개만 생성
        pilot_freq = [0; temp_pilot; 0; flip(conj(temp_pilot))];  % DC, 중간, 허미션 처리
    end
    pilot_time = ifft(pilot_freq) * sqrt(N + 2);
    pilot_time = [pilot_time(end - N_cp + 1 : end); pilot_time];

    disp('preamble, pilot_time, symbols_IFFTed_time 순으로 tx_signal을 구성합니다.');
    tx_signal = [preamble; pilot_time];
    for i = 1 : length(symbols_IFFTed_time)
        tx_signal = [tx_signal; symbols_IFFTed_time{i}];
        if rem(i, 4) == 0 && i ~= length(symbols_IFFTed_time)
            tx_signal = [tx_signal; pilot_time];
        end
    end

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

    figure;
    plot(tx_signal);

    % 시간 축을 초 단위로 변환하여 tx_signal을 플로팅합니다.
    t = (0:length(tx_signal)-1) / sampling_freq; % 시간 축 생성
    figure;
    plot(t, tx_signal);
    xlabel('시간 (초)');
    ylabel('신호 진폭');
    title('tx\_signal의 시간 영역 파형');
    
    % tx_signal의 주파수 스펙트럼을 계산하고 플로팅합니다.
    N_fft = length(tx_signal);
    f = (-N_fft/2:N_fft/2-1)*(sampling_freq/N_fft); % 주파수 축 생성
    tx_signal_fft = fftshift(fft(tx_signal));
    
    figure;
    plot(f, abs(tx_signal_fft));
    xlabel('주파수 (Hz)');
    ylabel('스펙트럼 진폭');
    title('tx\_signal의 주파수 스펙트럼');
    xlim([0, sampling_freq/2]); % 양의 주파수 성분만 표시
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

    if TX_THROUGH_MAT == true
        disp('tx_signal을 파일로 저장합니다.');
        save(save_load_path_and_file_name, 'tx_signal');
        figure;
        plot(tx_signal);
        disp('tx_signal을 파일로 저장하였습니다.');
    else
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
end

