function recognizedText = performEasyOCR(img)
    try
        % GPU 메모리 관리를 위한 Reader 객체 재사용
        persistent reader
        if isempty(reader)
            reader = py.easyocr.Reader({'ko', 'en'}, gpu=false);  % CPU 모드 명시
        end
        
        % 이미지를 uint8로 변환 (0-255 범위)
        img = uint8(img * 255);  % logical을 uint8로 변환
        
        % 이미지가 2D(흑백)인 경우 3D(RGB)로 변환
        if ismatrix(img)
            img = repmat(img, [1, 1, 3]);  % RGB로 변환
        end
        
        % 이미지 임시 저장
        temp_img_path = 'temp_ocr_img.png';
        imwrite(img, temp_img_path);
        
        % OCR 실행
        results = reader.readtext(temp_img_path);
        
        % 결과 텍스트 추출 및 합치기
        all_text = '';
        for i = 1:length(results)
            text = char(results{i}{2});
            all_text = [all_text, ' ', text];
        end
        recognizedText = strtrim(all_text);
        
        % 임시 파일 삭제
        delete(temp_img_path);
        
    catch ME
        fprintf('OCR 에러: %s\n', ME.message);
        recognizedText = 'OCR 처리 중 에러 발생';
    end
end