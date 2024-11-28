clear;
clc;

Base_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\';

List_Base_WAV_path = cell(1, 20);
for i = 1:20
    List_Base_WAV_path{i} = sprintf('%sBase_%d.WAV', Base_path, i);
end

% WAV_1_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_1.WAV';
% WAV_2_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_2.WAV';
% WAV_3_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_3.WAV';
% WAV_4_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_4.WAV';
% WAV_5_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_5.WAV';
% WAV_6_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_6.WAV';
% WAV_7_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_7.WAV';
% WAV_8_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_8.WAV';
% WAV_9_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_9.WAV';
% WAV_10_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_10.WAV';
% WAV_11_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_11.WAV';
% WAV_12_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_12.WAV';
% WAV_13_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_13.WAV';
% WAV_14_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_14.WAV';
% WAV_15_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_15.WAV';
% WAV_16_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_16.WAV';
% WAV_17_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_17.WAV';
% WAV_18_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_18.WAV';
% WAV_19_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_19.WAV';
% WAV_20_path = 'C:\Users\okmun\OneDrive\대외 공개 가능\고려대학교 전기전자공학부\24_2\종합설계II (신원재 교수님)\Base_WAV\Base_20.WAV';

% clc;
% [WAV_signal_1, Fs_1] = audioread(WAV_1_path);
% [WAV_signal_2, Fs_2] = audioread(WAV_2_path);
% 
% if size(WAV_signal_1, 2) > 1
%     WAV_signal_1 = mean(WAV_signal_1, 2);
% end

List_WAV_signal = cell(1, length(List_Base_WAV_path));
List_Fs = cell(1, length(List_Base_WAV_path));

for i = 1:length(List_Base_WAV_path)
    [List_WAV_signal{i}, List_Fs{i}] = audioread(List_Base_WAV_path{i});
    % if size(List_WAV_signal{i}, 2) > 1
    %     List_WAV_signal{i} = mean(List_WAV_signal{i}, 2);
    % end
    % audiowrite(List_Base_WAV_path{i}, List_WAV_signal{i}, List_Fs{i});
end

length(List_WAV_signal{1})
a = List_WAV_signal{1}(1:1000);
[xC_1, lags] = xcorr(List_WAV_signal{1}, a);
[xC_1, lags] = xcorr(List_WAV_signal{1}, a);
[max_1, idx_1] = max(xC_1);

[xC_2, lags] = xcorr(List_WAV_signal{2}, a);
[max_2, idx_2] = max(xC_2);

% max = 0;
% idx = 0;
% for i = 1:20
%     % [xC_1, lags] = xcorr(List_WAV_signal{1}, a);
%     [xC_1, lags] = xcorr(List_WAV_signal{i}, a);
%     % [max_1, idx_1] = max(xC_1);
%     [~, idx] = max(xC_1);
% end

max_xcorr = 0;
idx = 0;
i_save = 0;
for i = 1:length(List_WAV_signal)
    [xC, lags] = xcorr(List_WAV_signal{1}, a);
    % i
    [max_temp, idx_temp] = max(xC);
    if max_temp > max_xcorr
        max_xcorr = max_temp;
        idx = idx_temp;
        i_save = i;
    end
end