%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function behave_analysis_block1_230125_lick_all4_2_2_all_ver20230602

current_folder = pwd;

cd('e:\Tone_discri1\all_mouse_behavior\lick_data_20180312_4_2_2_no_lick\ver_20230125')

% [filename1, pathname1,findex]=uigetfile('*.mat','Lick ver 20230125','Multiselect','on');
% filename1
filename1 = dir('*.mat'); %Only one 
if length(filename1) ~= 6,
    filename1
    hoge
end

stim_lick_count = [];
rew_lick_count = [];

stim_lick_all = [];
rew_lick_all = [];

stim_lick_all2 = [];
rew_lick_all2 = [];

for filecount = 1:length(filename1)
%     temp_filename = filename1(filecount) 
%     temp_filename = cell2mat(temp_filename);
%     temp_path = pathname1;
%     fpath = fullfile(temp_path, temp_filename);
%     load(fpath);
    temp_filename = filename1(filecount).name;
    load(temp_filename);
    
    stim_lick_count = [stim_lick_count; stim.lick_on_off];
    rew_lick_count = [rew_lick_count; rew.lick_on_off];
    
    temp1 = stim.lick_on_off;
    temp1_sum = sum(temp1,2);
    temp1_sum = [temp1_sum, temp1_sum];
    temp1 = temp1 ./ temp1_sum;
    
    temp2 = rew.lick_on_off;
    temp2_sum = sum(temp2,2);
    temp2_sum = [temp2_sum, temp2_sum];
    temp2 = temp2 ./ temp2_sum;
    
    stim_lick_all = [stim_lick_all; temp1];
    rew_lick_all = [rew_lick_all; temp2];

    temp_stim = stim.lick_sound_trial;
    temp_rew = rew.lick_sound_trial;
    
    for j = 1:length(temp_stim)
        temp = temp_stim(j).matrix;
        temp = sum(temp) ./ length(temp);
        stim_lick_all2 = [stim_lick_all2; temp];
    end
    for j = 1:length(temp_rew)
        temp = temp_rew(j).matrix;
        temp = sum(temp) ./ length(temp);
        rew_lick_all2 = [rew_lick_all2; temp];
    end
end

stim_lick_all
rew_lick_all

figure
subplot(1,2,1)
boxplot(stim_lick_all)
subplot(1,2,2)
boxplot(rew_lick_all)

figure
subplot(1,2,1)
boxplot(stim_lick_all2)
subplot(1,2,2)
boxplot(rew_lick_all2)

cd(current_folder)


