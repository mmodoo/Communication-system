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

% MATLAB 주파수 범위 검출 예제

% 1. 설정
fs = 48000;          % 샘플링 주파수 (Hz)
frameLength = 4800;  % 프레임 길이 (샘플 수)
deviceReader = audioDeviceReader('SampleRate', fs, 'SamplesPerFrame', frameLength);

% 리소스 해제를 위한 onCleanup 객체 설정
cleanupObj = onCleanup(@() release(deviceReader));

% 2. 주파수 범위 설정
freqMin = 14000;     % 최소 주파수 (Hz)
freqMax = 16000;     % 최대 주파수 (Hz)

% FFT 주파수 벡터 생성
NFFT = frameLength;
f = (0:NFFT-1)*(fs/NFFT); % 주파수 벡터

% 주파수 범위에 해당하는 인덱스 계산
idxMin = find(f >= freqMin, 1, 'first');
idxMax = find(f <= freqMax, 1, 'last');

% 창 함수 설정 (Hamming 창)
window = hamming(frameLength);

% 에너지 누적 시간 및 임계값 설정
detectionTime = 0;         % 검출된 누적 시간 (초)
threshold = 1e7;           % 주파수 범위 에너지 임계값 (실험적으로 설정 필요)
requiredDuration = 3;      % 검출을 위한 필요한 시간 (초)

% 변수 초기화
a = 0;

disp('Listening 상태 시작...');

% 3. 실시간 오디오 모니터링 루프
while true
    % 오디오 프레임 읽기
    audioFrame = deviceReader();
    
    % 스테레오인 경우 모노로 변환
    if size(audioFrame, 2) > 1
        audioFrame = mean(audioFrame, 2);
    end
    
    % 창 함수 적용
    audioWindowed = audioFrame .* window;
    
    % FFT 수행
    Y = fft(audioWindowed, NFFT);
    
    % 파워 스펙트럼 계산
    Y_power = abs(Y).^2;
    
    % 특정 주파수 범위의 에너지 합계
    energy = sum(Y_power(idxMin:idxMax));
    
    % 에너지 임계값 검증
    if energy > threshold
        % 검출 시간 누적
        detectionTime = detectionTime + (frameLength / fs);
        fprintf('14000Hz~16000Hz 주파수 범위 검출 중... 누적 시간: %.2f 초\n', detectionTime);
        
        % 3초 이상 검출되었는지 확인
        if detectionTime >= requiredDuration && a ~= 3
            a = 3;
            disp('14000Hz~16000Hz 주파수 범위가 3초 이상 검출되었습니다. a = 3으로 설정합니다.');
            
            % 여기에 a == 3인 경우 수행할 작업을 구현합니다.
            % 예시 작업:
            disp('a == 3인 경우의 작업을 수행합니다.');
            % pause(2); % 실제 작업으로 대체
            
            % 작업이 끝난 후 초기화
            a = 0;
            detectionTime = 0;
            disp('작업 완료. 다시 Listening 상태로 돌아갑니다.');
        end
    else
        % 에너지가 임계값 이하인 경우
        if detectionTime > 0
            disp('14000Hz~16000Hz 주파수 범위 검출이 중단되었습니다. 누적 시간을 초기화합니다.');
        end
        detectionTime = 0;
    end
end

% (이 코드는 무한 루프이므로 아래 코드는 실행되지 않습니다.)
% release(deviceReader);
