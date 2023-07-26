%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_230602_sound_only_compare_block3_prefer_correct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
analysis_folder = 'stimulus';
%analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

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
        temp_stim = stim_sound(j).matrix;
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

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Check the number of neurons
if length(all_BF_neuron) ~= length(all_mouse_number)
    hoge
end

[length(all_median_correct), length(all_sig_sound_only)]
if length(all_median_correct) ~= length(all_sig_sound_only),
    hoge
end

for i = 1:6,
    temp_BF_freq = find(all_BF_neuron(:,2) == i);
    BF_freq(i).matrix = intersect(temp_BF_freq,sig_kruskal_sound);
    %BF_freq is the neuron number
end
%hist_x = [-inf,-1:0.1:1,inf];
hist_x = [-inf,-0.2:0.02:0.2,inf];

[sabun_easy, median_sabun(1), p_BF_sabun(1), length_BF(1),AIC_model(1).matrix,BIC_model(1).matrix,p_AIC_BIC(1,:),easy_mouse] = ...
    plot_activity_each_neuron_230602(all_median_block_L_correct, all_median_block_R_correct, BF_freq, all_p_block, all_mouse_number, 1, 1);
[sabun_mid,  median_sabun(2), p_BF_sabun(2), length_BF(2),AIC_model(2).matrix,BIC_model(2).matrix,p_AIC_BIC(2,:),mid_mouse] = ...
    plot_activity_each_neuron_230602(all_median_block_L_correct, all_median_block_R_correct, BF_freq, all_p_block, all_mouse_number, 2, 2);
[sabun_dif,  median_sabun(3), p_BF_sabun(3), length_BF(3),AIC_model(3).matrix,BIC_model(3).matrix,p_AIC_BIC(3,:),dif_mouse] = ...
    plot_activity_each_neuron_230602(all_median_block_L_correct, all_median_block_R_correct, BF_freq, all_p_block, all_mouse_number, 3, 3);

length(sig_kruskal_sound)

%Tuning curve
%make_tuning_curve_reward_error_evi(all_median_correct,all_median_error,all_BF_neuron(:,1),sig_sound_only);
%make_tuning_curve_reward_error_evi(all_median_correct,all_median_error,all_BF_neuron(:,2),sig_kruskal_sound);
make_tuning_curve_block_evi_230602(all_median_block_L_correct,all_median_block_R_correct,all_BF_neuron(:,2),sig_kruskal_sound, all_mouse_number);
%make_tuning_curve_block_evi_230602(all_median_block_L_correct,all_median_block_R_correct,all_BF_neuron(:,2),sig_kruskal_sound);

figure
boxplot([sabun_dif; sabun_mid; sabun_easy],[zeros(length(sabun_dif),1); ones(length(sabun_mid),1); ones(length(sabun_easy),1)*2]);
set(gca,'ylim',[-0.3 0.3])

disp('standard analysis')
median_sabun
length_BF
p_BF_sabun
p_BF_sabun(1)
p_BF_sabun(2)
p_BF_sabun(3)

disp('for lme analysis')
%p_AIC_BIC
disp('AIC')
p_AIC_BIC(1,1)
p_AIC_BIC(2,1)
p_AIC_BIC(3,1)
disp('BIC')
p_AIC_BIC(1,2)
p_AIC_BIC(2,2)
p_AIC_BIC(3,2)
% disp('easy for lme analysis')
% AIC_model(1).matrix
% BIC_model(1).matrix
% disp('mid for lme analysis')
% AIC_model(2).matrix
% BIC_model(2).matrix
% disp('dif for lme analysis')
% AIC_model(3).matrix
% BIC_model(3).matrix

disp('Compare across different BF')
p_sabun(1) = ranksum(sabun_dif, sabun_mid);
p_sabun(2) = ranksum(sabun_dif, sabun_easy);
p_sabun(3) = ranksum(sabun_mid, sabun_easy);
p_sabun
p_sabun(1)
p_sabun(2)
p_sabun(3)

%Make lme for ranksum test
dif_number = zeros(length(sabun_dif),1);
mid_number = ones(length(sabun_mid),1);
easy_number = ones(length(sabun_easy),1)*2;

temp_data = [sabun_dif; sabun_mid];
temp_variable = [dif_number; mid_number];
temp_random = [dif_mouse; mid_mouse];
[lme,AIC_model,BIC_model,p_AIC_BIC(1,:)] = fitlme_analysis_20210520_2_ranksum([temp_data, temp_variable, temp_random]);

temp_data = [sabun_dif; sabun_easy];
temp_variable = [dif_number; easy_number];
temp_random = [dif_mouse; easy_mouse];
[lme,AIC_model,BIC_model,p_AIC_BIC(2,:)] = fitlme_analysis_20210520_2_ranksum([temp_data, temp_variable, temp_random]);

temp_data = [sabun_mid; sabun_easy];
temp_variable = [mid_number; easy_number];
temp_random = [mid_mouse; easy_mouse];
[lme,AIC_model,BIC_model,p_AIC_BIC(3,:)] = fitlme_analysis_20210520_2_ranksum([temp_data, temp_variable, temp_random]);

disp('AIC')
p_AIC_BIC(1,1)
p_AIC_BIC(2,1)
p_AIC_BIC(3,1)
disp('BIC')
p_AIC_BIC(1,2)
p_AIC_BIC(2,2)
p_AIC_BIC(3,2)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sabun_all, median_sabun, p, length_neuron,AIC_model,BIC_model,p_AIC_BIC,all_mouse] = ...
    plot_activity_each_neuron_230602(all_median_block_L, all_median_block_R, BF_freq, all_p_block, all_mouse_number, neuron_number, tone_number)
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

all_mouse1 = all_mouse_number(BF_freq(neuron_number).matrix,:);
all_mouse2 = all_mouse_number(BF_freq(7-neuron_number).matrix,:);
all_mouse = [all_mouse1; all_mouse2];

sabun_all = double(sabun_all);
p = signrank(sabun_all);
median_sabun = median(sabun_all);
length_neuron = length(BF_freq(neuron_number).matrix) + length(BF_freq(7-neuron_number).matrix);

%Lme test
[lme,AIC_model,BIC_model,p_AIC_BIC] = fitlme_analysis_20210520_1([sabun_all,all_mouse]);

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
function make_tuning_curve_block_evi_230602(all_median_L,all_median_R,all_BF_neuron,sig_sound_only,all_mouse_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];

all_median_L = double(all_median_L);
all_median_R = double(all_median_R);

all_median_L = all_median_L(sig_sound_only,:);
all_median_R = all_median_R(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);

all_mouse_number = all_mouse_number(sig_sound_only,:);

all_prefer = [];
all_non = [];
all_mouse = [];
for j = 1:3
    temp_neuron1 = find(all_BF_neuron == j);
    temp_neuron2 = find(all_BF_neuron == 7-j);

    temp_median_prefer1 = fliplr(all_median_L(temp_neuron1,:));
    temp_median_non1 = fliplr(all_median_R(temp_neuron1,:));
    temp_median_prefer2 = all_median_R(temp_neuron2,:);
    temp_median_non2 = all_median_L(temp_neuron2,:);

    temp_prefer = [temp_median_prefer1; temp_median_prefer2];
    temp_non  = [temp_median_non1; temp_median_non2];
    
    temp_mouse1 = all_mouse_number(temp_neuron1,:);
    temp_mouse2 = all_mouse_number(temp_neuron2,:);
    temp_mouse = [temp_mouse1; temp_mouse2];
    
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
    all_mouse = [all_mouse; temp_mouse];
    
    length_BF_neuron(j,1) = size(temp_prefer,1);
    length_BF_neuron(j,2) = size(temp_non,1);
    
    for k = 1:6,
        p_block(j,k) = signrank(temp_prefer(:,k),temp_non(:,k));
        
        temp_data = temp_prefer(:,k)-temp_non(:,k);
        [lme,AIC_model,BIC_model,p_AIC_BIC] = fitlme_analysis_20210520_1([temp_data,temp_mouse]);
        p_block_AIC(j,k) = p_AIC_BIC(1);
        p_block_BIC(j,k) = p_AIC_BIC(2);
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
    
    %lme
    temp_data = all_prefer(:,k)-all_non(:,k);
    [lme,AIC_model,BIC_model,p_AIC_BIC] = fitlme_analysis_20210520_1([temp_data,all_mouse]);
    p_block_AIC(4,k) = p_AIC_BIC(1);
    p_block_BIC(4,k) = p_AIC_BIC(2);
end

length_BF_neuron
p_block
p_block_AIC
p_block_BIC

return

