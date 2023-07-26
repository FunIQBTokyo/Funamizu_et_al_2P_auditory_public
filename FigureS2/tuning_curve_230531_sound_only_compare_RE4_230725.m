%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_230531_sound_only_compare_RE4_230725
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
%analysis_folder = 'stimulus';
analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

temp = ['cd ',default_folder];
eval(temp); %move directory

clear neuron_number

all_sig_sound_or = [];
all_sig_sound_only = [];

all_stim_sound = [];
all_rew_sound = [];
all_stim_sound_evi = [];
all_rew_sound_evi = [];
all_p_sound = [];
all_p_sound_evi = [];

all_median_all = [];
all_median_correct = [];
all_median_error = [];
all_BF_neuron = [];
all_p_RE = [];
     
all_median_block_L = [];
all_median_block_R = [];

all_p_kruskal_stim = [];

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, moto_sig_kruskal_both] = ...
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
        clear p_kruskal_stim BF_neuron p_RE_neuron
        for k = 1:6,
            temp_evi = find(stim_evi == k);
            temp_correct = intersect(temp_evi, stim_correct);
            temp_error = intersect(temp_evi, stim_error);
            temp_block_L = intersect(temp_evi, stim_block_L);
            temp_block_R = intersect(temp_evi, stim_block_R);
            
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
            %get the significant test value
            median_block_L(:,k) = median(temp_stim(temp_block_L,:),1)';
            median_block_R(:,k) = median(temp_stim(temp_block_R,:),1)';
        end
        %Detect BF
        for l = 1:size_neuron,
            p_kruskal_stim(l,1) = kruskalwallis(temp_stim(:,l),stim_evi,'off');

            BF_neuron(l,1) = find(median_all(l,:) == max(median_all(l,:)),1);
            BF_neuron(l,2) = find(median_correct(l,:) == max(median_correct(l,:)),1);
            BF_neuron(l,3) = find(median_error(l,:) == max(median_error(l,:)),1);
        end

        all_p_RE = [all_p_RE; p_RE_neuron];
        all_p_kruskal_stim = [all_p_kruskal_stim; p_kruskal_stim];
        all_median_all = [all_median_all; median_all];
        all_median_correct = [all_median_correct; median_correct];
        all_median_error = [all_median_error; median_error];
        all_median_block_L = [all_median_block_L; median_block_L];
        all_median_block_R = [all_median_block_R; median_block_R];
        all_BF_neuron = [all_BF_neuron; BF_neuron];
    end    

    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2];
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

%Pick up the sig_sound_neuron with overlap
%Use the or sig neurons
all_sig_sound_or = find(all_sig_sound_or(:,1) == 1);
%all_sig_sound_and = find(all_sig_sound_and(:,1) == 1);
stim_sig_sound_only = find(all_sig_sound_only(:,1) == 1);
rew_sig_sound_only  = find(all_sig_sound_only(:,2) == 1);
sig_sound_only = union(stim_sig_sound_only,rew_sig_sound_only);

if length(analysis_folder) == 8, %Stimulus
    sig_sound_only = stim_sig_sound_only; %df/f
    moto_sig_task = moto_sig_kruskal_stim;
elseif length(analysis_folder) == 6, %reward
    sig_sound_only = rew_sig_sound_only; %df/f
    moto_sig_task = moto_sig_kruskal_rew;
else
    hoge
end
sig_kruskal_stim = find(all_p_kruskal_stim < 0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[length(all_median_correct), length(all_sig_sound_only)]
if length(all_median_correct) ~= length(all_sig_sound_only),
    hoge
end

make_tuning_curve_reward_error_low_high_neurons(all_median_correct,all_median_error,all_BF_neuron(:,2),sig_kruskal_sound)

temp_neuron1 = find(all_BF_neuron(:,2) < 3.5); %low neurons
temp_neuron2 = find(all_BF_neuron(:,2) > 3.5); %high neurons
temp_median_correct1_3 = all_median_correct(temp_neuron1,3);
temp_median_error1_3 = all_median_error(temp_neuron1,3);
temp_median_correct1_4 = all_median_correct(temp_neuron1,4);
temp_median_error1_4 = all_median_error(temp_neuron1,4);
temp_median_correct2_3 = all_median_correct(temp_neuron2,3);
temp_median_error2_3 = all_median_error(temp_neuron2,3);
temp_median_correct2_4 = all_median_correct(temp_neuron2,4);
temp_median_error2_4 = all_median_error(temp_neuron2,4);
figure
subplot(2,2,1)
boxplot([temp_median_correct1_3; temp_median_error1_3], [zeros(length(temp_neuron1),1);ones(length(temp_neuron1),1)])
subplot(2,2,2)
boxplot([temp_median_correct1_4; temp_median_error1_4], [zeros(length(temp_neuron1),1);ones(length(temp_neuron1),1)])
subplot(2,2,3)
boxplot([temp_median_correct2_3; temp_median_error2_3], [zeros(length(temp_neuron2),1);ones(length(temp_neuron2),1)])
subplot(2,2,4)
boxplot([temp_median_correct2_4; temp_median_error2_4], [zeros(length(temp_neuron2),1);ones(length(temp_neuron2),1)])

make_tuning_curve_prefer_explain(all_median_all,all_BF_neuron(:,2),sig_kruskal_sound);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function make_tuning_curve_reward_error_low_high_neurons(all_median_correct,all_median_error,all_BF_neuron,sig_sound_only)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];
all_median_correct = all_median_correct(sig_sound_only,:);
all_median_error = all_median_error(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);

correct_all = [];
error_all = [];

temp_neuron1 = find(all_BF_neuron < 3.5);
temp_neuron2 = find(all_BF_neuron > 3.5);

temp_median_correct1 = all_median_correct(temp_neuron1,:);
temp_median_error1 = all_median_error(temp_neuron1,:);
temp_median_correct2 = all_median_correct(temp_neuron2,:);
temp_median_error2 = all_median_error(temp_neuron2,:);

temp_prefer_correct = [fliplr(temp_median_correct1); temp_median_correct2];
temp_prefer_error = [fliplr(temp_median_error1); temp_median_error2];

figure
subplot(1,3,1)
plot_median_se_moto_x_axis(temp_median_correct1,freq_x,[255 102 21]./255,2)
hold on
plot_median_se_moto_x_axis(temp_median_error1,freq_x,[105 105 105]./255,2)
set(gca,'xlim',[-0.1 1.1])
subplot(1,3,2)
plot_median_se_moto_x_axis(temp_median_correct2,freq_x,[255 102 21]./255,2)
hold on
plot_median_se_moto_x_axis(temp_median_error2,freq_x,[105 105 105]./255,2)
set(gca,'xlim',[-0.1 1.1])
subplot(1,3,3)
plot_median_se_moto_x_axis(temp_prefer_correct,freq_x,[255 102 21]./255,2)
hold on
plot_median_se_moto_x_axis(temp_prefer_error,freq_x,[105 105 105]./255,2)
set(gca,'xlim',[-0.1 1.1])

size(temp_prefer_correct)
size(temp_prefer_error)

for i = 1:6,
    p(i) = signrank(temp_prefer_correct(:,i),temp_prefer_error(:,i));
end
p

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function make_tuning_curve_prefer_explain(all_median_all,all_BF_neuron,sig_sound_only)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use_color = jet(max(Sound_category));
number_color = 9;
use_color = jet(number_color);
% % use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
% %              use_color(number_color-2,:); use_color(number_color-1,:); use_color(number_color,:)];
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];

freq_x = [0 0.25 0.45 0.55 0.75 1];
all_median_all = all_median_all(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);

figure
for j = 1:3,
    temp_neuron1 = find(all_BF_neuron == j);

    temp_median = all_median_all(temp_neuron1,:);
    temp_median_flip = fliplr(temp_median);
    
    subplot(2,2,1)
    plot_median_se_moto_x_axis(temp_median,freq_x,use_color(j,:),2)
    hold on
    set(gca,'xlim',[-0.1 1.1])
    subplot(2,2,2)
    plot_median_se_moto_x_axis(temp_median_flip,freq_x,use_color(j,:),2)
    set(gca,'xlim',[-0.1 1.1])
    
    length_neuron_use(j) = length(temp_neuron1);
    prefer_trace(j).matrix = temp_median_flip;
end
for j = 4:6,
    temp_neuron1 = find(all_BF_neuron == j);

    temp_median = all_median_all(temp_neuron1,:);
    temp_median_flip = fliplr(temp_median);
    
    subplot(2,2,3)
    plot_median_se_moto_x_axis(temp_median,freq_x,use_color(j,:),2)
    hold on
    set(gca,'xlim',[-0.1 1.1])
    subplot(2,2,4)
    plot_median_se_moto_x_axis(temp_median,freq_x,use_color(j,:),2)
    set(gca,'xlim',[-0.1 1.1])
    
    length_neuron_use(j) = length(temp_neuron1);
    prefer_trace(j).matrix = temp_median;
end
length_neuron_use
sum(length_neuron_use)

prefer_color = [0 0 0; 0.3 0.3 0.3; 0.6 0.6 0.6];
figure
for j = 1:3,
    temp1 = prefer_trace(j).matrix;
    temp2 = prefer_trace(7-j).matrix;
    temp1 = [temp1; temp2];
    subplot(1,3,j)
    plot_median_se_moto_x_axis(temp1,freq_x,prefer_color(j,:),2)
    set(gca,'xlim',[-0.1 1.1])
end

%Make pie figure
number_color = 9;
use_color = jet(number_color);
% % use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
% %              use_color(number_color-2,:); use_color(number_color-1,:); use_color(number_color,:)];
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];

figure
test_pie = pie(length_neuron_use);
for i = 1:length(length_neuron_use),
    set(test_pie(2*i-1), 'FaceColor', use_color(i,:));
end

return


