function Tx_Step_11_play_WAV(Tx_signal, Sampling_Freq, Recording_time_Sec)
    
    % audioplayer 객체 생성
    player = audioplayer(Tx_signal, Sampling_Freq); 

    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp(['#### 재생 시작 시각: ', char(start_time)]);
    play(player);

    % 키보드 입력 처리를 위한 figure 생성
    fig = figure('KeyPressFcn', @(src, evt)keyPressCallback(evt, player));
    set(fig, 'Name', 'Press Space to Stop Playback', 'NumberTitle', 'off');

    % 재생 상태 모니터링
    disp(['#### 총 재생 시간: ', num2str(Recording_time_Sec), '초 (스페이스바를 누르면 중지됩니다)']);
    elapsed = 0;
    while isplaying(player)
        pause(1);
        elapsed = elapsed + 1;
        fprintf('\r##### 진행 시간: %d초 / %d초 (%.1f%%) (스페이스바를 누르면 중지됩니다)', ...
            elapsed, ceil(Recording_time_Sec), (elapsed/Recording_time_Sec)*100);

        if ~ishandle(fig)  % figure가 닫혔는지 확인
            stop(player);
            break;
        end
    end
    fprintf('\n');  % 줄바꿈을 추가하여 다음 출력이 깔끔하게 보이도록 함

    % 재생 종료 처리
    end_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    disp('.');
    disp(['#### 재생 종료 시각: ', char(end_time)]);
    elapsed_time = end_time - start_time;
    disp(['#### 재생 중 총 경과 시간[s]: ', char(elapsed_time)]);

    % figure 정리
    if ishandle(fig)
        close(fig);
    end
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