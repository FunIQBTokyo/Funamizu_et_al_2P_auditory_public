%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuning_curve_190429_sound_only_compare_RE5_ver20230607
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

all_median_all = [];
all_median_correct = [];
all_median_error = [];
all_BF_neuron = [];
all_p_RE = [];
     
all_median_block_L = [];
all_median_block_R = [];

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
        stim_task = data.stim_task; %[Sound, reward, choice, Evidence, Block];
    elseif length(analysis_folder) == 6, %reward
        stim_sound = data.rew_sound;
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
    
     all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[length(all_median_correct), length(all_sig_sound_only), length(all_mouse_number)]
if length(all_median_correct) ~= length(all_sig_sound_only)
    hoge
end
if length(all_median_correct) ~= length(all_mouse_number)
    hoge
end
%Start plotting the activity of neurons
make_tuning_curve_reward_error_low_high_neurons(all_median_correct,all_median_error,all_BF_neuron(:,2),sig_kruskal_sound,all_mouse_number)

% [p(1,:),length_neuron(1),easy_sabun] = make_tuning_curve_reward_BF_neurons(all_median_correct,all_median_error,all_BF_neuron(:,2),all_p_RE,sig_kruskal_sound,[1,6]);
% [p(2,:),length_neuron(2),mid_sabun] = make_tuning_curve_reward_BF_neurons(all_median_correct,all_median_error,all_BF_neuron(:,2),all_p_RE,sig_kruskal_sound,[2,5]);
% [p(3,:),length_neuron(3),dif_sabun] = make_tuning_curve_reward_BF_neurons(all_median_correct,all_median_error,all_BF_neuron(:,2),all_p_RE,sig_kruskal_sound,[3,4]);
% 
% p
% 
% p(1,3)
% p(1,4)
% p(2,3)
% p(2,4)
% p(3,3)
% p(3,4)
% length_neuron
% 
% easy_sabun = double(easy_sabun);
% mid_sabun = double(mid_sabun);
% dif_sabun = double(dif_sabun);
% 
% easy_number = zeros(length_neuron(1),1);
% mid_number = ones(length_neuron(2),1);
% dif_number = ones(length_neuron(3),1) * 2;
% 
% figure
% subplot(1,2,1)
% h1 = boxplot([easy_sabun(:,3);mid_sabun(:,3);dif_sabun(:,3)],[easy_number;mid_number;dif_number]);
% set(h1(7,:),'Visible','off')
% set(gca,'ylim',[-0.3 0.4])
% 
% subplot(1,2,2)
% h2 = boxplot([easy_sabun(:,4);mid_sabun(:,4);dif_sabun(:,4)],[easy_number;mid_number;dif_number]);
% set(h2(7,:),'Visible','off')
% set(gca,'ylim',[-0.3 0.4])
% 
% p_sabun(1,1) = ranksum(easy_sabun(:,3),mid_sabun(:,3));
% p_sabun(1,2) = ranksum(easy_sabun(:,3),dif_sabun(:,3));
% p_sabun(1,3) = ranksum(mid_sabun(:,3),dif_sabun(:,3));
% p_sabun(2,1) = ranksum(easy_sabun(:,4),mid_sabun(:,4));
% p_sabun(2,2) = ranksum(easy_sabun(:,4),dif_sabun(:,4));
% p_sabun(2,3) = ranksum(mid_sabun(:,4),dif_sabun(:,4));
% 
% p_sabun(1,1)
% p_sabun(1,2)
% p_sabun(1,3)
% p_sabun(2,1)
% p_sabun(2,2)
% p_sabun(2,3)
% p_sabun




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p, length_neuron, sabun_prefer] = make_tuning_curve_reward_BF_neurons(all_median_correct,all_median_error,all_BF_neuron,p_RE,sig_sound_only,LH_neuron)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];
all_median_correct = all_median_correct(sig_sound_only,:);
all_median_error = all_median_error(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);
p_RE = p_RE(sig_sound_only,:);

correct_all = [];
error_all = [];

temp_neuron1 = find(all_BF_neuron == LH_neuron(1));
temp_neuron2 = find(all_BF_neuron == LH_neuron(2));

temp_median_correct1 = all_median_correct(temp_neuron1,:);
temp_median_error1 = all_median_error(temp_neuron1,:);
temp_median_correct2 = all_median_correct(temp_neuron2,:);
temp_median_error2 = all_median_error(temp_neuron2,:);
p_RE1 = p_RE(temp_neuron1,:);
p_RE2 = p_RE(temp_neuron2,:);

temp_prefer_correct = [fliplr(temp_median_correct1); temp_median_correct2];
temp_prefer_error = [fliplr(temp_median_error1); temp_median_error2];
p_RE_prefer = [fliplr(p_RE1); p_RE2];

figure
subplot(1,3,1)
plot_median_se_moto_x_axis(temp_prefer_correct,freq_x,[255 102 21]./255,2)
hold on
plot_median_se_moto_x_axis(temp_prefer_error,freq_x,[105 105 105]./255,2)
set(gca,'xlim',[-0.1 1.1])
set(gca,'ylim',[0 0.21])

length_neuron = length(temp_neuron1) + length(temp_neuron2);

subplot(1,3,2)
plot_each_neuron_RE(temp_prefer_correct, temp_prefer_error, p_RE_prefer, LH_neuron, length_neuron, 3);
subplot(1,3,3)
plot_each_neuron_RE(temp_prefer_correct, temp_prefer_error, p_RE_prefer, LH_neuron, length_neuron, 4);

size(temp_prefer_correct)
size(temp_prefer_error)

for i = 1:6,
    p(i) = signrank(double(temp_prefer_correct(:,i)),double(temp_prefer_error(:,i)));
end
sabun_prefer = temp_prefer_correct - temp_prefer_error;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_each_neuron_RE(temp_prefer_correct, temp_prefer_error, p_RE_prefer, LH_neuron, length_neuron, tone_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp_sig = find(p_RE_prefer(:,tone_number) < 0.05);
temp_plus  = find(temp_prefer_correct(:,tone_number) - temp_prefer_error(:,tone_number) > 0);
temp_minus = find(temp_prefer_correct(:,tone_number) - temp_prefer_error(:,tone_number) < 0);
temp_plus = intersect(temp_plus, temp_sig);
temp_minus = intersect(temp_minus, temp_sig);
temp_non_sig = setdiff([1:length_neuron],temp_sig);

if LH_neuron(1) == 1,
    plot(temp_prefer_correct(temp_non_sig,tone_number), temp_prefer_error(temp_non_sig,tone_number), 'ko')
    hold on
    plot(temp_prefer_correct(temp_plus,tone_number), temp_prefer_error(temp_plus,tone_number), 'ro')
    hold on
    plot(temp_prefer_correct(temp_minus,tone_number), temp_prefer_error(temp_minus,tone_number), 'bo')
elseif LH_neuron(1) == 2,
    plot(temp_prefer_correct(temp_non_sig,tone_number), temp_prefer_error(temp_non_sig,tone_number), 'kd')
    hold on
    plot(temp_prefer_correct(temp_plus,tone_number), temp_prefer_error(temp_plus,tone_number), 'rd')
    hold on
    plot(temp_prefer_correct(temp_minus,tone_number), temp_prefer_error(temp_minus,tone_number), 'bd')
else
    plot(temp_prefer_correct(temp_non_sig,tone_number), temp_prefer_error(temp_non_sig,tone_number), 'kx')
    hold on
    plot(temp_prefer_correct(temp_plus,tone_number), temp_prefer_error(temp_plus,tone_number), 'rx')
    hold on
    plot(temp_prefer_correct(temp_minus,tone_number), temp_prefer_error(temp_minus,tone_number), 'bx')
end    
hold on
plot([-1,7],[-1,7],'k')
set(gca,'xlim',[-1,7],'ylim',[-1 7])
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function make_tuning_curve_reward_error_low_high_neurons(all_median_correct,all_median_error,all_BF_neuron,sig_sound_only,all_mouse_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];
all_median_correct = all_median_correct(sig_sound_only,:);
all_median_error = all_median_error(sig_sound_only,:);
all_BF_neuron = all_BF_neuron(sig_sound_only);
all_mouse_number = all_mouse_number(sig_sound_only,:);

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

temp_mouse_number = [all_mouse_number(temp_neuron1,:); all_mouse_number(temp_neuron2,:)];

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

temp_prefer_correct = double(temp_prefer_correct);
temp_prefer_error = double(temp_prefer_error);
size(temp_prefer_correct)
size(temp_prefer_error)

for i = 1:6
    data = temp_prefer_correct(:,i)-temp_prefer_error(:,i);
    p(i) = signrank(data);
    [lme,AIC_model,BIC_model,p_AIC_BIC(i,:)] = fitlme_analysis_20210520_1([data,temp_mouse_number]);
end
p
p_AIC_BIC(:,2)

length(temp_mouse_number)

return
