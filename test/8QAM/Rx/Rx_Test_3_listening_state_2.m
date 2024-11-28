try
    release(devicereader);
catch ME
    switch ME.identifier  % 에러 타입에 따라 다른 처리
        case 'MATLAB:UndefinedFunction'
            fprintf('devicereader가 정의되지 않았습니다.\n');
        case 'MATLAB:class:InvalidHandle'
            fprintf('devicereader가 유효하지 않은 핸들입니다.\n');
        otherwise
            fprintf('예상치 못한 오류 발생: %s\n', ME.message);
    end
    
    % 에러 로그 기록이나 추가 처리가 필요한 경우
    % disp(ME.stack);  % 에러가 발생한 위치 추적
end

% 오디오 입력 및 16kHz 순음 감지 시스템
function detectSineWave()
    % 상수 설정
    TARGET_FREQ = 16000;     % 목표 주파수 (Hz)
    SAMPLE_RATE = 44100;     % 샘플링 레이트 (Hz)
    WINDOW_SIZE = 4096;      % FFT 윈도우 크기
    THRESHOLD = 0.7;         % 감지 임계값
    DURATION_THRESHOLD = 3;  % 필요한 지속 시간 (초)
    
    % 오디오 입력 객체 생성
    audioObj = audioDeviceReader('SampleRate', SAMPLE_RATE, ...
                               'SamplesPerFrame', WINDOW_SIZE);
    
    % 주파수 빈 계산
    freqBins = (0:WINDOW_SIZE-1) * SAMPLE_RATE / WINDOW_SIZE;
    targetBin = round(TARGET_FREQ * WINDOW_SIZE / SAMPLE_RATE) + 1;
    
    % 변수 초기화
    detectionCount = 0;
    lastTime = 0;
    a = 0;
    
    fprintf('Listening for %d Hz tone...\n', TARGET_FREQ);
    
    try
        while true
            % 오디오 데이터 읽기
            audioData = audioObj();
            
            % FFT 수행
            spectrum = abs(fft(audioData));
            normalizedSpectrum = spectrum / max(spectrum);
            
            % 목표 주파수 주변 검사
            freqRange = targetBin + (-2:2);  % 주변 빈까지 검사
            freqRange = freqRange(freqRange > 0 & freqRange <= length(normalizedSpectrum));
            peakValue = max(normalizedSpectrum(freqRange));
            
            % 순음 감지
            if peakValue > THRESHOLD
                detectionCount = detectionCount + WINDOW_SIZE / SAMPLE_RATE;
                if detectionCount >= DURATION_THRESHOLD
                    fprintf('16kHz tone detected for %0.2f seconds!\n', detectionCount);
                    a = 3;
                    break;  % while 루프 종료
                end
            else
                % 감지가 끊기면 카운터 리셋
                detectionCount = 0;
            end
            
            % 현재 시간 표시 (1초마다)
            currentTime = floor(toc);
            if currentTime > lastTime
                fprintf('Listening... %d seconds\n', currentTime);
                lastTime = currentTime;
            end
        end
        
    catch ME
        % 에러 처리
        fprintf('Error occurred: %s\n', ME.message);
    end
    
    % 오디오 객체 정리
    release(audioObj);
end

% 메인 실행 루프
while true
    a = 0;
    tic;  % 타이머 시작
    
    % 순음 감지 함수 실행
    detectSineWave();
    
    % a가 3이 되면 실행할 코드
    if a == 3
        fprintf('Executing code for a = 3\n');
        % 여기에 필요한 코드 추가
        
        fprintf('Returning to listening mode...\n');
        pause(1);  % 잠시 대기
    end
end
