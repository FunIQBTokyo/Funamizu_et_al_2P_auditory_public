%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_200403_decoder_single_neuron_ver230601
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%From: tuning_curve_190801_decoder_single_neuron

currentFolder = pwd;

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';
%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');
%all_sound all_sound_L all_sound_R all_sound_category all_sound_max_time 
%all_sig_sound all_sig_sound_S all_sig_sound_L all_sig_sound_R 
%all_block_L all_block_R 
%all_block_LL all_block_LR all_block_RL all_block_RR 
%all_block_category_L all_block_category_R all_block_max_time
%all_roi_overlap

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');
% [filename2, pathname2,findex]=uigetfile('*.mat','Overlap_file','Multiselect','on');
% filename2

pathname3 = strcat('e:/Tone_discri1/all_mouse/single_decode/separate4/single_190925/');
cd(pathname3)
filename3 = dir('*.mat');

%Behavior data
pathname4 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731/');
cd(pathname4)
filename4 = dir('*.mat');

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, moto_sig_kruskal_both,...
 moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all
%Based on all_mouse_number make the session data
max_session = max(all_mouse_number);
max_session = max_session(2);
all_session_mouse = [];
for i = 1:max_session
    temp = find(all_mouse_number(:,2) == i);
    temp = all_mouse_number(temp,1);
    temp = unique(temp);
    if length(temp) ~= 1
        hoge
    else
        all_session_mouse(i,1) = temp;
    end
end
%all_session_mouse

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

all_BF_sound = [];
all_thre_decode = [];
all_reward_evi = [];
all_correct = [];
all_correct_evi = [];
all_p_prior = [];

all_correct_behave = [];
all_correct_evi_mix = [];
all_opt_curve = [];
all_max_neuron = [];
all_max_neurometric = [];
all_neurometric = [];

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
    clear mouse_p_kruskal_stim mouse_BF_stim mouse_sabun_block mouse_std_sound
    for j = 1:length_session,
        temp_stim = stim_sound(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);

        %Each tone cloud
        stim_evi   = temp_stim_task(:,4);
        stim_reward = temp_stim_task(:,2);
        stim_block = temp_stim_task(:,5);
        stim_category = temp_stim_task(:,1);
        stim_correct = find(stim_reward == 1);
        stim_error   = find(stim_reward == 0);
        stim_block_L = find(stim_block == 0);
        stim_block_R  = find(stim_block == 1);
        stim_category_L = find(stim_category == 0);
        stim_category_R = find(stim_category == 1);
        %BF is determined with activity during reward
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear median_block_L_correct median_block_R_correct
        clear p_kruskal_stim BF_neuron p_RE_neuron p_block_neuron p_block_correct
        clear sabun_block p_prior
        %Make new definition about block modulation
        %Within each tone cloud category, if the activity is different
        %block modulation
        block_category1L = intersect(stim_block_L, stim_category_L);
        block_category2L = intersect(stim_block_R, stim_category_L);
        block_category1R = intersect(stim_block_L, stim_category_R);
        block_category2R = intersect(stim_block_R, stim_category_R);
        for l = 1:size_neuron,
            p_prior(l,1) = ranksum(temp_stim(block_category1L,l),temp_stim(block_category2L,l));
            p_prior(l,2) = ranksum(temp_stim(block_category1R,l),temp_stim(block_category2R,l));
        end
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
            
            %Based on the BF_neuron(2), get the sabun block activity
            if BF_neuron(l,2) < 3.5,
                sabun_block(l,1) = median_block_L(l,BF_neuron(l,2)) - median_block_R(l,BF_neuron(l,2));
            else
                sabun_block(l,1) = median_block_R(l,BF_neuron(l,2)) - median_block_L(l,BF_neuron(l,2));
            end
        end

        all_p_prior = [all_p_prior; p_prior];
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
        
        mouse_p_kruskal_stim(j).matrix = p_kruskal_stim;
        mouse_BF_stim(j).matrix = BF_neuron(:,2); %reward only
        mouse_sabun_block(j).matrix = sabun_block;
        mouse_std_sound(j).matrix = std(temp_stim);
    end    
    
    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2, sig_sound_ore];
    %Pick only sig_sound
    clear mouse_sig_sound
    length_session = length(data2.stim_box);
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            temp_stim = data2.stim_box(j).matrix;
        elseif length(analysis_folder) == 6, %reward
            temp_stim  = data2.rew_box(j).matrix;
        end
        temp_sound = temp_stim(:,2);
        mouse_sig_sound(j).matrix = temp_sound;
        all_sig_sound_only = [all_sig_sound_only; temp_sound];
    end
    
     all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
    
    temp_filename = filename3(i).name 
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    data4 = load(fpath);

    clear temp_sound_decode_neuron temp_sound_neuron
    clear length_sig_sound session_sabun median_session_thre
    clear max_correct_neuron max_neurometric
    for j = 1:length_session,
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         correct_rate = [correct_rate_mix, correct_rate_easy, correct_rate_mid, correct_rate_dif, temp_correct2];
%         correct_rate2 = [mix_L0, mix_L1, mix_R0, mix_R1];
%         reward_rate = all easy mid diff
        %About decoding
        if length(analysis_folder) == 8, %Stimulus
            temp_correct = data4.stim(j).matrix.distri_sound.correct_rate; %same diff mix
            correct_sound2 = data4.stim(j).matrix.distri_sound.correct_rate2;
            reward_sound = data4.stim(j).matrix.distri_sound.reward_sound;
            temp_neurometric = data4.stim(j).matrix.distri_sound.opt_LR_mix;
        elseif length(analysis_folder) == 6, %reward
            temp_correct = data4.rew(j).matrix.distri_sound.correct_rate; %same diff mix
            correct_sound2 = data4.rew(j).matrix.distri_sound.correct_rate2;
            reward_sound = data4.rew(j).matrix.distri_sound.reward_sound;
            temp_neurometric = data4.rew(j).matrix.distri_sound.opt_LR_mix;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Find the best neuron
        temp_correct_mix = temp_correct(:,1);
        max_correct = find(temp_correct_mix == max(temp_correct_mix),1);
        max_correct_neuron(j,1) = temp_correct_mix(max_correct);
        max_neurometric(j,:) = temp_neurometric(max_correct,:);
        
        all_correct = [all_correct; temp_correct];
        all_correct_evi = [all_correct_evi; correct_sound2];
        all_reward_evi = [all_reward_evi; reward_sound];
        all_neurometric = [all_neurometric; temp_neurometric];
        %get median based session
    end
    all_max_neuron = [all_max_neuron; max_correct_neuron];
    all_max_neurometric = [all_max_neurometric; max_neurometric];
    
    temp_filename = filename4(i).name 
    temp_path = pathname4;
    fpath = fullfile(temp_path, temp_filename);
    data4 = load(fpath);
    clear correct_mix opt_curve
    for j = 1:length_session,
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %About decoding
        if length(analysis_folder) == 8, %Stimulus
            temp_correct = data4.stim_correct_sum;
            temp_correct_evi = data4.stim_correct_evi(j).matrix;
            temp_opt_curve = data4.stim_opt_curve(j).matrix; %left right mix
        elseif length(analysis_folder) == 6, %reward
            temp_correct = data4.rew_correct_sum;
            temp_correct_evi = data4.rew_correct_evi(j).matrix;
            temp_opt_curve = data4.rew_opt_curve(j).matrix; %left right mix
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        correct_mix(j,:) = temp_correct_evi(3,:);
        opt_curve(j,:) = temp_opt_curve(3,:);
    end
    all_correct_behave = [all_correct_behave; temp_correct];
    all_correct_evi_mix = [all_correct_evi_mix; correct_mix];
    all_opt_curve = [all_opt_curve; opt_curve];
end

% all_opt_curve
% figure
% plot(all_opt_curve')

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

all_BF_neuron = all_BF_neuron(:,2);
for i = 1:length(all_BF_neuron),
    all_p_BF(i,1) = all_p_block(i,all_BF_neuron(i));
end
thre = 0.05;
all_p_prior = all_p_BF; %Take the smaller value
%all_p_prior = min(all_p_prior,[],2); %Take the smaller value

%Pick up the sig_sound_neuron with overlap
%Use the or sig neurons
all_sig_sound_or = find(all_sig_sound_or(:,1) == 1);

[length(all_correct), length(all_sig_sound_or)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Get how much neurons had prior change
p_sig_neuron   = all_p_prior(all_sig_sound_or);
p_sound_neuron = all_p_prior(sig_kruskal_sound);

%thre = 0.01; 
p_sig_number   = find(p_sig_neuron < thre);
p_sound_number = find(p_sound_neuron < thre);

sound_neuron_modulated = sig_kruskal_sound(p_sound_number);
sound_neuron_non_modulated = setdiff(sig_kruskal_sound, sound_neuron_modulated);

length_neuron(1,:) = [length(sig_kruskal_sound), length(all_sig_sound_or)];
length_neuron(2,:) = [length(p_sound_number), length(p_sig_number)];
length_neuron(3,:) = [length(sound_neuron_modulated), length(sound_neuron_non_modulated)];
length_neuron

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compare the best performance neuron and behavior
figure
plot(all_correct_behave(:,3), all_max_neuron, 'k.')
hold on
plot([0.5 1], [0.5 1], 'k')
signrank(all_correct_behave(:,3), all_max_neuron)
%lme analysis
lme = fitlme_analysis_20210520_0(all_correct_behave(:,3)-all_max_neuron,all_session_mouse);
lme(2).lme

%Example sessions
good_session = find(all_max_neuron > all_correct_behave(:,3));
length(good_session)

if length(analysis_folder) == 8 %Stimulus
    good_session = good_session([11,12,15,16,20]); %Stimulus task
elseif length(analysis_folder) == 6 %reward
    good_session = good_session([6 10 14 15 16 17 18 25 33 34 37 40 43]); %reward task
else
    hoge
end

temp_x = [0:0.01:1];
for i = 1:length(good_session)
    if rem(i,6) == 1,
        figure
        count = 0;
    end
    count = count + 1;
    subplot(2,3,count)
    plot(temp_x, all_max_neurometric(good_session(i,:),:), 'r')
    hold on
    plot(temp_x, all_opt_curve(good_session(i,:),:), 'k')
    set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])
end

hist_x = [-inf,0.5 : 0.05 : 1,inf];
label_x = [0.45,0.5 : 0.05 : 1,1.05];
label_x = (label_x(1:length(label_x)-1) + label_x(2:length(label_x))) ./ 2;

plot_correct_rate_neurons(all_correct, all_correct_evi, all_neurometric, ...
    all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)

%plot_correct_rate_behavior(all_correct_behave, all_correct_evi_mix, all_opt_curve, [1:length(all_correct_behave)], hist_x, label_x)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_correct_rate_behavior(all_correct, all_correct_evi, all_neurometric, all_sig_sound_or, hist_x, label_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get correct rate for diff mid easy
all_correct_mix = all_correct(:,3);
all_correct_dif = (all_correct_evi(:,3) + all_correct_evi(:,4)) ./ 2;
all_correct_mid = (all_correct_evi(:,2) + all_correct_evi(:,5)) ./ 2;
all_correct_easy = (all_correct_evi(:,1) + all_correct_evi(:,6)) ./ 2;

all_right_rate = all_correct_evi;
all_right_rate(:,1) = 1 - all_right_rate(:,1);
all_right_rate(:,2) = 1 - all_right_rate(:,2);
all_right_rate(:,3) = 1 - all_right_rate(:,3);

ave_correct_mix = mean(all_correct_mix(all_sig_sound_or));
ave_correct_dif = mean(all_correct_dif(all_sig_sound_or));
ave_correct_mid = mean(all_correct_mid(all_sig_sound_or));
ave_correct_easy = mean(all_correct_easy(all_sig_sound_or));

%Correct rate histgram
%hist_x = [-inf,0.5 : 0.05 : 1,inf];

hist_correct_all = histcounts(all_correct_mix(all_sig_sound_or), hist_x);
hist_correct_dif = histcounts(all_correct_dif(all_sig_sound_or), hist_x);
hist_correct_mid = histcounts(all_correct_mid(all_sig_sound_or), hist_x);
hist_correct_easy = histcounts(all_correct_easy(all_sig_sound_or), hist_x);

prob_correct_all = hist_correct_all ./ sum(hist_correct_all);
prob_correct_dif = hist_correct_dif ./ sum(hist_correct_dif);
prob_correct_mid = hist_correct_mid ./ sum(hist_correct_mid);
prob_correct_easy = hist_correct_easy ./ sum(hist_correct_easy);

figure
subplot(2,2,1)
plot(label_x, prob_correct_all,'k')
hold on
plot(label_x, prob_correct_dif,'b')
hold on
plot(label_x, prob_correct_mid,'g')
hold on
plot(label_x, prob_correct_easy,'r')

hold on %Plot ave correct rate
plot([ave_correct_mix, ave_correct_mix], [0 0.5],'k')
hold on
plot([ave_correct_dif, ave_correct_dif], [0 0.5],'b')
hold on
plot([ave_correct_mid, ave_correct_mid], [0 0.5],'g')
hold on
plot([ave_correct_easy, ave_correct_easy], [0 0.5],'r')

set(gca,'ylim',[0 1])
set(gca,'xlim',[0.45 1.05])

subplot(2,2,2)
plot(label_x, hist_correct_all,'k')
hold on
plot(label_x, hist_correct_dif,'b')
hold on
plot(label_x, hist_correct_mid,'g')
hold on
plot(label_x, hist_correct_easy,'r')
set(gca,'ylim',[0 max(all_sig_sound_or)])
set(gca,'xlim',[0.45 1.05])

%Box plot
subplot(2,2,3)
correct_rate_boxplot(all_right_rate(all_sig_sound_or,:));

%Tuning curves
temp_x = [0:0.01:1];

subplot(2,2,4)
%All 83 lines, plot each function
[~,order_correct] = sort(all_correct_mix(all_sig_sound_or),'descend');
%all_correct_mix(order_correct)
all_neurometric = all_neurometric(order_correct,:);
length_plot = length(all_sig_sound_or);
%use_color = cool(length_plot);
use_color = summer(length_plot);
for i = 1:length_plot,
    plot(temp_x, all_neurometric(i,:), 'color',use_color(i,:))
    hold on
end
hold on
plot(temp_x, mean(all_neurometric), 'color',[0 0 0])
set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])
%ave_neuron_tuning_curve(all_correct_mix(all_sig_sound_or), all_neurometric(all_sig_sound_or,:))
size(all_neurometric)

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_correct_rate_neurons(all_correct, all_correct_evi, all_neurometric, ...
    all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated,hist_x, label_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get correct rate for diff mid easy
all_correct_mix = all_correct(:,1);
all_correct_easy = all_correct(:,2);
all_correct_mid = all_correct(:,3);
all_correct_dif = all_correct(:,4);

% all_right_rate = all_correct_evi;
% all_right_rate(:,1) = 1 - all_right_rate(:,1);
% all_right_rate(:,2) = 1 - all_right_rate(:,2);
% all_right_rate(:,3) = 1 - all_right_rate(:,3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fitting dy definied the moving bins

%Correct rate histgram
%hist_x = [-inf,0.5 : 0.05 : 1,inf];
hist_x_sabun = [-inf,0 : 0.003 : 0.03,inf];

hist_count_dif_neuron_group(all_correct_mix, all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)
hist_count_dif_neuron_group(all_correct_dif, all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)
hist_count_dif_neuron_group(all_correct_mid, all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)
hist_count_dif_neuron_group(all_correct_easy, all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)

% %Tuning curves
% figure
% subplot(2,2,1)
% ave_neuron_tuning_curve(all_correct_mix(all_sig_sound_or,:), all_neurometric(all_sig_sound_or,:))
% subplot(2,2,2)
% ave_neuron_tuning_curve(all_correct_mix(sig_kruskal_sound,:), all_neurometric(sig_kruskal_sound,:))
% subplot(2,2,3)
% ave_neuron_tuning_curve(all_correct_mix(sound_neuron_modulated,:), all_neurometric(sound_neuron_modulated,:))
% subplot(2,2,4)
% ave_neuron_tuning_curve(all_correct_mix(sound_neuron_non_modulated,:), all_neurometric(sound_neuron_non_modulated,:))

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ave_neuron_tuning_curve(all_correct_mix, all_neurometric)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp_x = [0:0.01:1];
set_neuron = 10;

[~,order_correct] = sort(all_correct_mix,'descend');
%all_correct_mix(order_correct)
all_neurometric = all_neurometric(order_correct,:);

length_neuron = length(all_correct_mix);

length_plot = ceil(length_neuron / set_neuron);
use_color = cool(length_plot);

count = 0;
for i = 1:set_neuron:length_neuron,
    count = count + 1;
    temp1 = i;
    temp2 = min(set_neuron * (i+1), length_neuron);
    
    temp_neurometric = all_neurometric([temp1:temp2],:);
    plot(temp_x, mean(temp_neurometric),'color',use_color(count,:))
    set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])
    hold on
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function correct_rate_boxplot(all_right_rate)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
length_neuron = length(all_right_rate);

for i = 1:6,
    rand_x = ones(length_neuron,1) .* i + 0.2 .* (rand(length_neuron,1) - 0.5);
    plot(rand_x, all_right_rate(:,i), '.', 'color', [0.7 0.7 0.7])
    hold on
end
boxplot(all_right_rate)
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hist_count_dif_neuron_group(all_correct, all_sig_sound_or, sig_kruskal_sound, sound_neuron_modulated, sound_neuron_non_modulated, hist_x, label_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hist_correct_all   = histcounts(all_correct(all_sig_sound_or), hist_x);
hist_correct_sound = histcounts(all_correct(sig_kruskal_sound), hist_x);
hist_correct_sound_mod = histcounts(all_correct(sound_neuron_modulated), hist_x);
hist_correct_sound_non = histcounts(all_correct(sound_neuron_non_modulated), hist_x);

prob_correct_all = hist_correct_all ./ sum(hist_correct_all);
prob_correct_sound = hist_correct_sound ./ sum(hist_correct_sound);
prob_correct_sound_mod = hist_correct_sound_mod ./ sum(hist_correct_sound_mod);
prob_correct_sound_non = hist_correct_sound_non ./ sum(hist_correct_sound_non);

figure
subplot(1,2,1)
plot(label_x, prob_correct_all,'color',[0.5 0.5 0.5])
hold on
%plot(prob_correct_sound,'r')
%subplot(1,2,2)
plot(label_x, prob_correct_sound_mod,'r')
hold on
plot(label_x, prob_correct_sound_non,'b')
set(gca,'ylim',[0 0.5])
set(gca,'xlim',[0.45 1.05])

subplot(1,2,2)
plot(label_x, hist_correct_all,'color',[0.5 0.5 0.5])
hold on
%plot(hist_correct_sound,'r')
%subplot(1,2,2)
plot(label_x, hist_correct_sound_mod,'r')
hold on
plot(label_x, hist_correct_sound_non,'b')
set(gca,'ylim',[0 9000])
set(gca,'xlim',[0.45 1.05])

return
