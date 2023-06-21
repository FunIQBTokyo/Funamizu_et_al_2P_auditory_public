%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuning_curve_230531_sound_only_compare_block3_prefer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
%analysis_folder = 'stimulus';
analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

%[filename1, pathname1,findex]=uigetfile('*.mat','Sound_file','Multiselect','on');
pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

cd(default_folder)

clear neuron_number
all_sig_sound_or = [];
all_sig_sound_only = [];

all_median_all = [];
all_median_correct = [];
all_median_error = [];
all_BF_neuron = [];
all_p_RE = [];
all_p_block = [];
all_p_block_correct = [];


all_median_block_L = [];
all_median_block_R = [];
all_median_block_L_correct = [];
all_median_block_R_correct = [];

all_p_kruskal_stim = [];

all_p_block_20230531 = [];
all_activity_block_20230531 = [];

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, ...
 moto_sig_kruskal_both, moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all

for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    %'stim_sound','stim_norm','stim_task','stim_order_block','stim_mean','stim_std',
    %'rew_sound','rew_norm','rew_task','rew_order_block','rew_mean','rew_std'
    if length(analysis_folder) == 8, %Stimulus
        %'stim_sound','stim_baseline','stim_task','stim_order_block','rew_sound','rew_baseline','rew_task','rew_order_block'
        stim_sound = data.stim_sound; %df/f
        %stim_baseline = data.stim_baseline;
        stim_task = data.stim_task; %[Sound, reward, choice, Evidence, Block];
    elseif length(analysis_folder) == 6, %reward
        stim_sound = data.rew_sound;
        %rew_baseline = data.rew_baseline;
        stim_task = data.rew_task; %[Sound, reward, choice, Evidence, Block];
    else
        hoge
    end

    length_session = length(stim_sound);
    for j = 1:length_session,
        temp_stim = stim_sound(j).matrix; %activity
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);

        %Each tone cloud
        stim_evi   = temp_stim_task(:,4);
        stim_reward = temp_stim_task(:,2);
        stim_block = temp_stim_task(:,5);
        stim_correct = find(stim_reward == 1);
        stim_error   = find(stim_reward == 0);
        stim_block_L = find(stim_block == 0);
        stim_block_R  = find(stim_block == 1);
        
        %Activity depended on the block
        clear p_block_0531 median_block_0531
        for l = 1:size_neuron
            %size(temp_stim(stim_block_L,l))
            p_block_0531(l,1) = ranksum(temp_stim(stim_block_L,l),temp_stim(stim_block_R,l));
            median_block_0531(l,:) = [median(temp_stim(stim_block_L,l)), median(temp_stim(stim_block_R,l))];
        end
        
        %BF is determined with activity during reward
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear median_block_L_correct median_block_R_correct
        clear p_kruskal_stim BF_neuron p_RE_neuron p_block_neuron p_block_correct
        for k = 1:6,
            temp_evi = find(stim_evi == k);
            temp_correct = intersect(temp_evi, stim_correct);
            temp_error = intersect(temp_evi, stim_error);
            temp_block_L = intersect(temp_evi, stim_block_L);
            temp_block_R = intersect(temp_evi, stim_block_R);
            temp_block_L_correct = intersect(temp_correct, temp_block_L);
            temp_block_R_correct = intersect(temp_correct, temp_block_R);
            
            median_all(:,k) = median(temp_stim(temp_evi,:),1)';
            median_correct(:,k) = median(temp_stim(temp_correct,:),1)';
            if length(temp_error) ~= 0,
                median_error(:,k) =  median(temp_stim(temp_error,:),1)';
                for l = 1:size_neuron,
                    p_RE_neuron(l,k) = ranksum(temp_stim(temp_correct,l),temp_stim(temp_error,l));
                end
            else
                median_error(:,k) =  nan(size_neuron,1);
                p_RE_neuron(:,k) = nan(size_neuron,1);
            end
            
            for l = 1:size_neuron,
                p_block_neuron(l,k) = ranksum(temp_stim(temp_block_L,l),temp_stim(temp_block_R,l));
                if length(temp_block_L_correct) ~= 0 & length(temp_block_R_correct) ~= 0
                    p_block_correct(l,k) = ranksum(temp_stim(temp_block_L_correct,l),temp_stim(temp_block_R_correct,l));
                else
                    p_block_correct(l,k) = nan;
                end
            end
            
            %get the significant test value
            median_block_L(:,k) = median(temp_stim(temp_block_L,:),1)';
            median_block_R(:,k) = median(temp_stim(temp_block_R,:),1)';
            median_block_L_correct(:,k) = median(temp_stim(temp_block_L_correct,:),1)';
            median_block_R_correct(:,k) = median(temp_stim(temp_block_R_correct,:),1)';
        end
        %Detect BF
        for l = 1:size_neuron,
            p_kruskal_stim(l,1) = kruskalwallis(temp_stim(:,l),stim_evi,'off');

            BF_neuron(l,1) = find(median_all(l,:) == max(median_all(l,:)),1);
            BF_neuron(l,2) = find(median_correct(l,:) == max(median_correct(l,:)),1);
            BF_neuron(l,3) = find(median_error(l,:) == max(median_error(l,:)),1);
        end

        all_p_RE = [all_p_RE; p_RE_neuron];
        all_p_block = [all_p_block; p_block_neuron];
        all_p_block_correct = [all_p_block_correct; p_block_correct];
        all_p_kruskal_stim = [all_p_kruskal_stim; p_kruskal_stim];
        all_median_all = [all_median_all; median_all];
        all_median_correct = [all_median_correct; median_correct];
        all_median_error = [all_median_error; median_error];
        all_median_block_L = [all_median_block_L; median_block_L];
        all_median_block_R = [all_median_block_R; median_block_R];
        all_median_block_L_correct = [all_median_block_L_correct; median_block_L_correct];
        all_median_block_R_correct = [all_median_block_R_correct; median_block_R_correct];
        all_BF_neuron = [all_BF_neuron; BF_neuron];
        
        all_p_block_20230531 = [all_p_block_20230531; p_block_0531];
        all_activity_block_20230531 = [all_activity_block_20230531; median_block_0531];
    end    

    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2, sig_sound_ore];
    %Pick only sig_sound
    length_session = length(data2.stim_box);
    for j = 1:length_session,
        temp_stim = data2.stim_box(j).matrix;
        temp_rew  = data2.rew_box(j).matrix;
        temp_sound = [temp_stim(:,2), temp_rew(:,2)];
        all_sig_sound_only = [all_sig_sound_only; temp_sound];
    end
    
    %sig_roi_overlap_matrix = [sig_roi_overlap; sig_roi_overlap_S; sig_roi_overlap_L; sig_roi_overlap_R];
     all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
end

cd(currentFolder)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%all_sig_sound_or: number of overlapping neurons
task_relevant = find(all_sig_sound_or == 1);

%This is based on the overlapping neurons
check_neuron_number

length_important_neurons = ...
[length(all_sig_sound_or),length(task_relevant),length(moto_sig_kruskal_stim), length(moto_sig_kruskal_rew), length(moto_sig_kruskal_both), length(moto_sig_sound_timing)]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Remove the sound responsive neurons (1735) and focus on the task relevant
%neurons (13581), how was the activity changed by the stimulus probability or
%reward amount.
task_relevant_no_sound = setdiff(task_relevant, sig_kruskal_sound);
length(task_relevant_no_sound) %13581-1735 = 11846

size(all_activity_block_20230531) %same as the overlapping neurons: 17523
all_activity_block_20230531 = double(all_activity_block_20230531);

%Plot the activity change between the blocks
block_hist_x = [-inf,-0.2:0.02:0.2,inf];
block_hist_abs = [0:0.01:0.2,inf];

[hist_no_sound,abs_hist_no_sound,distri_no_sound,abs_sabun_no_sound] = plot_block_different_activity(all_activity_block_20230531, all_p_block_20230531, task_relevant_no_sound, block_hist_x,block_hist_abs);

[hist_sound,abs_hist_sound,distri_sound,abs_sabun_sound] = plot_block_different_activity(all_activity_block_20230531, all_p_block_20230531, sig_kruskal_sound, block_hist_x,block_hist_abs);

figure
subplot(1,3,1)
plot(hist_no_sound,'k')
hold on
plot(hist_sound,'b')
set(gca,'xlim',[0 length(block_hist_x)])

subplot(1,3,2)
plot(abs_hist_no_sound,'k')
hold on
plot(abs_hist_sound,'b')
set(gca,'xlim',[0 length(block_hist_abs)])

subplot(1,3,3)
plot(distri_no_sound(:,1),distri_no_sound(:,2),'k')
hold on
plot(distri_sound(:,1),distri_sound(:,2),'b')

%all_mouse_number; mouse_number, session_number
data1 = [abs_sabun_sound; abs_sabun_no_sound];
group_number = zeros(length(abs_sabun_sound),1);
group_number1 = ones(length(abs_sabun_no_sound),1);
data2 = [group_number; group_number1];
data3 = [all_mouse_number(sig_kruskal_sound,:); all_mouse_number(task_relevant_no_sound,:)];

[length(abs_sabun_no_sound), length(abs_sabun_sound), length(abs_sabun_no_sound)+length(abs_sabun_sound)]

ranksum(abs_sabun_no_sound, abs_sabun_sound)

lme = fitlme_analysis_20210520_2_ranksum([data1,data2,data3]);
for i = 1:length(lme)
    %lme(i).lme
    AIC_model(i) = lme(i).lme.ModelCriterion.AIC;
    BIC_model(i) = lme(i).lme.ModelCriterion.BIC;
end
AIC_model = find(AIC_model == min(AIC_model));
BIC_model = find(BIC_model == min(BIC_model));
lme(AIC_model).lme
lme(BIC_model).lme

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Another way to test the distribution
%Task-relevant neurons > sound_sig_neurons > sound_responsive neurons
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
task_relevant_no_sound = setdiff(task_relevant, moto_sig_sound_timing);
sound_timing_neuron = setdiff(moto_sig_sound_timing, sig_kruskal_sound);
% test = length(task_relevant_no_sound)+length(sound_timing_neuron)+length(sig_kruskal_sound); %13581
% [length(task_relevant_no_sound),length(sound_timing_neuron),length(sig_kruskal_sound),test]

size(all_activity_block_20230531) %same as the overlapping neurons: 17523
all_activity_block_20230531 = double(all_activity_block_20230531);

%Plot the activity change between the blocks
block_hist_x = [-inf,-0.2:0.02:0.2,inf];
block_hist_abs = [0:0.01:0.2,inf];

[hist_no_sound,abs_hist_no_sound,distri_no_sound,abs_sabun_no_sound] = plot_block_different_activity(all_activity_block_20230531, all_p_block_20230531, task_relevant_no_sound, block_hist_x,block_hist_abs);
[hist_time_sound,abs_hist_time_sound,distri_time_sound,abs_sabun_time_sound] = plot_block_different_activity(all_activity_block_20230531, all_p_block_20230531, sound_timing_neuron, block_hist_x,block_hist_abs);
[hist_sound,abs_hist_sound,distri_sound,abs_sabun_sound] = plot_block_different_activity(all_activity_block_20230531, all_p_block_20230531, sig_kruskal_sound, block_hist_x,block_hist_abs);

figure
subplot(1,3,1)
plot(hist_no_sound,'k')
hold on
plot(hist_time_sound,'g')
hold on
plot(hist_sound,'b')
set(gca,'xlim',[0 length(block_hist_x)])

subplot(1,3,2)
plot(abs_hist_no_sound,'k')
hold on
plot(abs_hist_time_sound,'g')
hold on
plot(abs_hist_sound,'b')
set(gca,'xlim',[0 length(block_hist_abs)])

subplot(1,3,3)
plot(distri_no_sound(:,1),distri_no_sound(:,2),'k')
hold on
plot(distri_time_sound(:,1),distri_time_sound(:,2),'g')
hold on
plot(distri_sound(:,1),distri_sound(:,2),'b')

[length(abs_sabun_no_sound),length(abs_sabun_time_sound),length(abs_sabun_sound),length(abs_sabun_no_sound)+length(abs_sabun_time_sound)+length(abs_sabun_sound)]
ranksum(abs_sabun_no_sound, abs_sabun_time_sound)
ranksum(abs_sabun_sound, abs_sabun_no_sound)
ranksum(abs_sabun_sound, abs_sabun_time_sound)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot the activity change between the blocks
function [hist_sabun,abs_hist_sabun,plot_distri,abs_sabun] = plot_block_different_activity(block_activity, block_p, use_neuron, hist_x,hist_abs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

block_activity = block_activity(use_neuron,:);
block_p = block_p(use_neuron,:);

sabun_activity = block_activity(:,2) - block_activity(:,1);
abs_sabun = abs(sabun_activity);
sort_abs_sabun = sort(abs_sabun);
use_y = [1/length(sort_abs_sabun):1/length(sort_abs_sabun):1];
use_y = use_y';

hist_sabun = histcounts(sabun_activity, hist_x);
abs_hist_sabun = histcounts(abs_sabun, hist_abs);

figure
subplot(2,2,1)
plot(block_activity(:,1), block_activity(:,2), 'k.')
subplot(2,2,2)
plot(hist_sabun,'b')
subplot(2,2,3)
plot(abs_hist_sabun,'b')
subplot(2,2,4)
plot(sort_abs_sabun,use_y,'b')

hist_sabun = hist_sabun ./ sum(hist_sabun);
abs_hist_sabun = abs_hist_sabun ./ sum(abs_hist_sabun);
plot_distri = [sort_abs_sabun, use_y];

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sabun_all, median_sabun, p, length_neuron] = plot_activity_each_neuron(all_median_block_L, all_median_block_R, BF_freq, all_p_block, neuron_number, tone_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%neuron number: 1-3
%tone number: 1-3

low_prefer = find(all_median_block_L(:,tone_number) > all_median_block_R(:,tone_number));
high_prefer = find(all_median_block_R(:,7-tone_number) > all_median_block_L(:,7-tone_number));
p_low  = find(all_p_block(:,tone_number) < 0.05);
p_high = find(all_p_block(:,7-tone_number) < 0.05);
p_low_prefer  = intersect(p_low,low_prefer);
p_high_prefer = intersect(p_high,high_prefer);
p_low_non_prefer = setdiff(p_low, p_low_prefer);
p_high_non_prefer = setdiff(p_high, p_high_prefer);
    
p_non_low  = setdiff([1:length(all_p_block)], p_low);
p_non_high = setdiff([1:length(all_p_block)], p_high);

figure
%subplot(1,2,1)
%non sig neurons
plot_raster_each_BF_neuron(tone_number,neuron_number,all_median_block_L,all_median_block_R,BF_freq,p_non_low,p_non_high,[0 0 0]); %black
hold on
%sig but opposite side
plot_raster_each_BF_neuron(tone_number,neuron_number,all_median_block_L,all_median_block_R,BF_freq,p_low_non_prefer,p_high_non_prefer,[0 0 1]); %blue
hold on
%sig neurons
plot_raster_each_BF_neuron(tone_number,neuron_number,all_median_block_L,all_median_block_R,BF_freq,p_low_prefer,p_high_prefer,[1 0 0]); %red
hold on
plot([-1 8],[-1 8],'k')
set(gca,'xlim',[-1 8],'ylim',[-1 8])

%Get the values
sabun1 = all_median_block_L(BF_freq(neuron_number).matrix,tone_number)-all_median_block_R(BF_freq(neuron_number).matrix,tone_number);
sabun2 = all_median_block_R(BF_freq(7-neuron_number).matrix,7-tone_number)-all_median_block_L(BF_freq(7-neuron_number).matrix,7-tone_number);
sabun_all = [sabun1; sabun2];

sabun_all = double(sabun_all);
p = signrank(sabun_all);
median_sabun = median(sabun_all);
length_neuron = length(BF_freq(neuron_number).matrix) + length(BF_freq(7-neuron_number).matrix);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_raster_each_BF_neuron(tone_number,neuron_number,all_median_block_L,all_median_block_R,BF_freq,sig_low,sig_high,use_color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    BF_freq(neuron_number).matrix = intersect(BF_freq(neuron_number).matrix,sig_low);
    BF_freq(7-neuron_number).matrix = intersect(BF_freq(7-neuron_number).matrix,sig_high);
    
    if length(sig_low) ~= 0,
        if neuron_number == 1;
            plot(all_median_block_L(BF_freq(neuron_number).matrix,tone_number),all_median_block_R(BF_freq(neuron_number).matrix,tone_number),'o','color',use_color)
        elseif neuron_number == 2,
            plot(all_median_block_L(BF_freq(neuron_number).matrix,tone_number),all_median_block_R(BF_freq(neuron_number).matrix,tone_number),'d','color',use_color)
        elseif neuron_number == 3,
            plot(all_median_block_L(BF_freq(neuron_number).matrix,tone_number),all_median_block_R(BF_freq(neuron_number).matrix,tone_number),'+','color',use_color)
        end
        hold on
    end
    if length(sig_high) ~= 0,
        if neuron_number == 1;
            plot(all_median_block_R(BF_freq(7-neuron_number).matrix,7-tone_number),all_median_block_L(BF_freq(7-neuron_number).matrix,7-tone_number),'o','color',use_color)
        elseif neuron_number == 2,
            plot(all_median_block_R(BF_freq(7-neuron_number).matrix,7-tone_number),all_median_block_L(BF_freq(7-neuron_number).matrix,7-tone_number),'d','color',use_color)
        elseif neuron_number == 3,
            plot(all_median_block_R(BF_freq(7-neuron_number).matrix,7-tone_number),all_median_block_L(BF_freq(7-neuron_number).matrix,7-tone_number),'+','color',use_color)
        end
    end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function make_tuning_curve_block_evi(all_median_L,all_median_R,all_BF_neuron,sig_sound_only)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];

all_median_L = double(all_median_L);
all_median_R = double(all_median_R);

all_median_L = all_median_L(sig_sound_only,:);
all_median_R = all_median_R(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);

all_prefer = [];
all_non = [];
for j = 1:3
    temp_neuron1 = find(all_BF_neuron == j);
    temp_neuron2 = find(all_BF_neuron == 7-j);

    temp_median_prefer1 = fliplr(all_median_L(temp_neuron1,:));
    temp_median_non1 = fliplr(all_median_R(temp_neuron1,:));
    temp_median_prefer2 = all_median_R(temp_neuron2,:);
    temp_median_non2 = all_median_L(temp_neuron2,:);

    temp_prefer = [temp_median_prefer1; temp_median_prefer2];
    temp_non  = [temp_median_non1; temp_median_non2];
    
    figure
    %plot_median_se_moto(temp_prefer,[255 102 21]./255,2)
    %plot_median_se_moto_x_axis(temp_prefer,freq_x,[0 141 203]./255,2)
    plot_median_se_moto_x_axis(temp_prefer,freq_x,[1 0 0],2)
    hold on
    %plot_median_se_moto(temp_non,[105 105 105]./255,2)
    %plot_median_se_moto_x_axis(temp_non,freq_x,[0 0 0],2)
    plot_median_se_moto_x_axis(temp_non,freq_x,[0 0 1],2)
    set(gca,'xlim',[-0.1 1.1])
    set(gca,'ylim',[0 0.2])
    
    all_prefer = [all_prefer; temp_prefer];
    all_non = [all_non; temp_non];
    
    length_BF_neuron(j,1) = size(temp_prefer,1);
    length_BF_neuron(j,2) = size(temp_non,1);
    
    for k = 1:6,
        p_block(j,k) = signrank(temp_prefer(:,k),temp_non(:,k));
    end
end
length_BF_neuron(4,1) = size(all_prefer,1);
length_BF_neuron(4,2) = size(all_non,1);

figure
%plot_median_se_moto(temp_prefer,[255 102 21]./255,2)
plot_median_se_moto_x_axis(all_prefer,freq_x,[1 0 0],2)
hold on
%plot_median_se_moto(temp_non,[105 105 105]./255,2)
plot_median_se_moto_x_axis(all_non,freq_x,[0 0 1],2)
set(gca,'xlim',[-0.1 1.1])

for k = 1:6,
    p_block(4,k) = signrank(all_prefer(:,k),all_non(:,k));
end

length_BF_neuron
p_block

return

