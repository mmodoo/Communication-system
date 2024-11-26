function Binarised_img_column_vector = Tx_Step_1_imread_resize_gray_mono_repetition(Img_Path_and_File_Name, Fixed_Img_Size, Whether_NOT_Repetition_coding__OR__Repetition_How_many, Whether_PAPR_improved_inter_leaving__OR__NOT)
    
    Img = imread(Img_Path_and_File_Name);
    % figure;
    % imshow(Img);

    Img = imresize(Img, Fixed_Img_Size);
    % figure;
    % imshow(Resized_Img);

    [~, ~, numChannels] = size(Img)
    Whether_Logical = islogical(Img)

    % Img = rgb2gray(Img);
    % Img = imbinarize(Img);
    if numChannels == 3
        Img = rgb2gray(Img);
        % figure;
        % imshow(Gray_Img);
    % elseif numChannels == 1 && Whether_Logical == true
    %     Img = imbinarize(Img);
    % figure;
    % imshow(Binarised_img);
    end
    Img = imbinarize(Img);
    % size(Img)
    % Img(1, 1)

    Binarised_img_column_vector = Img(:);

    if Whether_NOT_Repetition_coding__OR__Repetition_How_many ~= 1
        Binarised_img_column_vector = repelem(Binarised_img_column_vector, Whether_NOT_Repetition_coding__OR__Repetition_How_many);
    end

    if Whether_PAPR_improved_inter_leaving__OR__NOT == true
        rng('default');
        interleaver_order = randperm(length(Binarised_img_column_vector));
        Binarised_img_column_vector = Binarised_img_column_vector(interleaver_order);
    end
end