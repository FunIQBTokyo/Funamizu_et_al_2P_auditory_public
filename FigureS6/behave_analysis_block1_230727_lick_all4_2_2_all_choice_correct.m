
function behave_analysis_block1_230727_lick_all4_2_2_all_choice_correct

current_folder = pwd;
[stim_lick_all, rew_lick_all, stim_lick_trial, rew_lick_trial] = ...
    get_behavior_lick_data(current_folder);


%Correct rate data
use_frame = 27;
[stim_correct_trial, rew_correct_trial] = get_decode_correct_data(use_frame, current_folder);
 
size(stim_lick_trial)
size(stim_correct_trial)

use_scale = [0.1 0.9];

[stim_p, stim_p_choice,length_session(1)] = combine_lick_choice_decode(stim_lick_trial, stim_correct_trial,use_scale,1);
[rew_p, rew_p_choice,length_session(2)] = combine_lick_choice_decode(rew_lick_trial, rew_correct_trial,use_scale,1);

stim_p
stim_p_choice
stim_p_choice(1)
stim_p_choice(3)
rew_p
rew_p_choice
rew_p_choice(1)
rew_p_choice(3)

length_session


for i = 1:60
    [stim_correct_trial, rew_correct_trial, stim_sound_all(:,i), rew_sound_all(:,i), ...
        stim_choice_all(:,i), rew_choice_all(:,i)] = get_decode_correct_data(i, current_folder);
    [~,~,~,temp_correct_stim,temp_correct_stim2] = combine_lick_choice_decode(stim_lick_trial, stim_correct_trial,use_scale,0);
    [~,~,~,temp_correct_rew,temp_correct_rew2] = combine_lick_choice_decode(rew_lick_trial, rew_correct_trial,use_scale,0);
    correct_stim1(:,i) = temp_correct_stim(:,1);
    correct_stim2(:,i) = temp_correct_stim(:,2);
    correct_stim3(:,i) = temp_correct_stim2(:,1);
    correct_stim4(:,i) = temp_correct_stim2(:,2);
    correct_rew1(:,i) = temp_correct_rew(:,1);
    correct_rew2(:,i) = temp_correct_rew(:,2);
    correct_rew3(:,i) = temp_correct_rew2(:,1);
    correct_rew4(:,i) = temp_correct_rew2(:,2);
end
figure
subplot(2,1,1)
%plot_mean_se_moto(correct_stim1,[0 0 1],2)
plot_mean_se_moto(correct_stim1,[1 0 0],2)
hold on
%plot_mean_se_moto(correct_stim2,[1 0 0],2)
plot_mean_se_moto(correct_stim2,[1 0 1],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.65])
subplot(2,1,2)
%plot_mean_se_moto(correct_rew1,[0 0 1],2)
plot_mean_se_moto(correct_rew1,[1 0 0],2)
hold on
%plot_mean_se_moto(correct_rew2,[1 0 0],2)
plot_mean_se_moto(correct_rew2,[1 0 1],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.65])

figure
subplot(2,1,1)
plot_mean_se_moto(correct_stim3,[0 0 1],2)
hold on
plot_mean_se_moto(correct_stim4,[1 0 0],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.7])
subplot(2,1,2)
plot_mean_se_moto(correct_rew3,[0 0 1],2)
hold on
plot_mean_se_moto(correct_rew4,[1 0 0],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.7])

figure
subplot(2,1,1)
plot_mean_se_moto(stim_sound_all,[0 0 1],2)
hold on
plot_mean_se_moto(stim_choice_all,[1 0 0],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.85])
subplot(2,1,2)
plot_mean_se_moto(rew_sound_all,[0 0 1],2)
hold on
plot_mean_se_moto(rew_choice_all,[1 0 0],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
set(gca,'ylim', [0.45 0.85])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p, p_choice,length_session,median_decode_prop,median_decode_prop2] = combine_lick_choice_decode(stim_lick_trial, stim_correct_trial,use_scale,make_fig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if length(stim_lick_trial) ~= length(stim_correct_trial)
    hoge
end

for i = 1:length(stim_lick_trial)
    temp_lick =   stim_lick_trial(i).matrix;
    temp_decode = stim_correct_trial(i).matrix;
%     temp_decode
%     temp_lick
%     hoge
    
    if length(temp_lick)~=length(temp_decode)
        boge
    end
    temp0 = find(temp_lick == 0); %no side lick
    temp1 = find(temp_lick == 1); %side lick
    %This is still the matrix, not vector
%     temp_decode0 = temp_decode(temp0);
%     temp_decode1 = temp_decode(temp1);
%     size(temp_decode)
    temp_decode0 = temp_decode(temp0,:);
    temp_decode1 = temp_decode(temp1,:);
    temp_decode0 = nanmean(temp_decode0,1);
    temp_decode1 = nanmean(temp_decode1,1);
%    size(temp_decode0)
%    size(temp_decode1)
    
    lick_prop(i,1) = length(temp0) ./ length(temp_lick);
    decode_prop(i,:) = [nanmean(temp_decode0), nanmean(temp_decode1)];
end
%lick_prop

%median_decode_prop = median(decode_prop);
median_decode_prop = decode_prop;
%decode_prop
%use_trial = find(lick_prop > 0.2 & lick_prop < 0.8);
use_trial = find(lick_prop > use_scale(1) & lick_prop < use_scale(2));

length_session = length(use_trial);

decode_prop = double(decode_prop);

if make_fig
    figure
    subplot(1,2,1)
    boxplot(decode_prop)
    hold on
    plot(decode_prop')
    set(gca,'xlim',[0 3],'ylim',[0.3 0.8])
end
median(decode_prop);
p(1) = signrank(decode_prop(:,1),decode_prop(:,2));

p_choice(1,1) = signrank(decode_prop(:,1),0.5);
p_choice(1,2) = signrank(decode_prop(:,2),0.5);

decode_prop = decode_prop(use_trial,:);
if make_fig
    subplot(1,2,2)
    boxplot(decode_prop)
    hold on
    plot(decode_prop')
    set(gca,'xlim',[0 3],'ylim',[0.3 0.8])
end

%median_decode_prop2 = median(decode_prop)
median_decode_prop2 = decode_prop;
p(2) = signrank(decode_prop(:,1),decode_prop(:,2));
p_choice(2,1) = signrank(decode_prop(:,1),0.5);
p_choice(2,2) = signrank(decode_prop(:,2),0.5);

% p
% p_choice
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [stim_correct_trial, rew_correct_trial, stim_sound_all, rew_sound_all, stim_choice_all, rew_choice_all] = ...
    get_decode_correct_data(use_frame, current_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd('E:\Tone_discri1\all_mouse\single_decode\separate4\population_190717_time_seq\ver20230127')

filename1 = dir('*.mat'); %Only one 
if length(filename1) ~= 6,
    filename1
    hoge
end

count = 0;
for filecount = 1:length(filename1)
    temp_filename = filename1(filecount).name;
    load(temp_filename);

    for j = 1:length(stim)
       count = count + 1;
       stim_correct_trial(count).matrix = stim(j).matrix.distri_choice(use_frame).matrix.correct_choice_trial_moto;
       rew_correct_trial(count).matrix = rew(j).matrix.distri_choice(use_frame).matrix.correct_choice_trial_moto;
%        stim_correct_trial(filecount,j).matrix = stim(j).matrix.distri_choice(use_frame).matrix.correct_choice_trial_moto;
%        rew_correct_trial(filecount,j).matrix = rew(j).matrix.distri_choice(use_frame).matrix.correct_choice_trial_moto;

       stim_sound_all(count,1) = stim(j).matrix.distri_sound(use_frame).matrix.correct_sound_moto;
       rew_sound_all(count,1) = rew(j).matrix.distri_sound(use_frame).matrix.correct_sound_moto;
       stim_choice_all(count,1) = stim(j).matrix.distri_choice(use_frame).matrix.correct_choice_moto;
       rew_choice_all(count,1) = rew(j).matrix.distri_choice(use_frame).matrix.correct_choice_moto;
    end
end    

cd(current_folder)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [stim_lick_all, rew_lick_all, stim_lick_trial, rew_lick_trial] = get_behavior_lick_data(current_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

count_stim = 0;
count_rew = 0;
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
        
        count_stim = count_stim + 1;
        %stim_lick_trial(filecount,j).matrix = temp_stim(j).matrix;
        stim_lick_trial(count_stim).matrix = temp_stim(j).matrix;
    end
    for j = 1:length(temp_rew)
        temp = temp_rew(j).matrix;
        temp = sum(temp) ./ length(temp);
        rew_lick_all2 = [rew_lick_all2; temp];
        
        count_rew = count_rew + 1;
        %rew_lick_trial(filecount,j).matrix = temp_rew(j).matrix;
        rew_lick_trial(count_rew).matrix = temp_rew(j).matrix;
    end
end

stim_lick_all
rew_lick_all

% figure
% subplot(1,2,1)
% boxplot(stim_lick_all)
% subplot(1,2,2)
% boxplot(rew_lick_all)
% 
% figure
% subplot(1,2,1)
% boxplot(stim_lick_all2)
% subplot(1,2,2)
% boxplot(rew_lick_all2)

cd(current_folder)
return


