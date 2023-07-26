%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuning_curve_180928_ROC_RE_each_FOV2_block3_correct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filename1 = dir('Delta_180605_1_*.mat');
load(filename1.name)
%load(filename1)
%all_sound all_sound_L all_sound_R all_sound_category all_sound_max_time 
%all_sig_sound all_sig_sound_S all_sig_sound_L all_sig_sound_R 
%all_block_L all_block_R 
%all_block_LL all_block_LR all_block_RL all_block_RR 
%all_block_category_L all_block_category_R all_block_max_time
%all_roi_overlap
%[filename2, pathname2,findex]=uigetfile('*.mat','Overlap_file');
filename2 = dir('roi_overlap*.mat');
load(filename2.name)
%roi_overlap

filename4 = dir('block_mat_170727*.mat');
load(filename4.name)

filename5 = dir('overlap181021*.mat')
load(filename5.name)

filename6 = dir('Block_tif_180520_6.mat')
load(filename6.name)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Behave analysis
%Choice_trial = find(Outcome == 1 | Outcome == 2 | Outcome == 4);
Choice_trial = find(Outcome == 1 | Outcome == 2);

%Sound LR
Sound_trial = Correct_side(Choice_trial); 
Sound_L = find(Correct_side(Choice_trial) == 0);
Sound_R = find(Correct_side(Choice_trial) == 1);

%Reward or Error
RE_trial = Correct_side(Choice_trial) == Chosen_side(Choice_trial);
Reward = find(RE_trial == 1);
Error = find(RE_trial == 0);

%Choice side
LR_trial = Chosen_side(Choice_trial);
left  = find(LR_trial == 0);
right = find(LR_trial == 1);

L_Reward = intersect(Sound_L, Reward);
L_Error  = intersect(Sound_L, Error);
R_Reward = intersect(Sound_R, Reward);
R_Error  = intersect(Sound_R, Error);

%Evidence_strength
Sound_category = zeros(length(Choice_trial),1);
L_category = [3 2 1];
R_category = [4 5 6];
clear Evidence
Trial_L = [];
Trial_R = [];
RE_L = [];
RE_R = [];
Evidence_sound = unique(EvidenceStrength);
for i = 1:length(Evidence_sound),
    temp = find(EvidenceStrength(Choice_trial) == Evidence_sound(i));
    Evidence(i).matrix = temp;
    
    %Sound
    Trial_L = intersect(temp, Sound_L);
    Trial_R = intersect(temp, Sound_R);
    Sound_category(Trial_L) = L_category(i);
    Sound_category(Trial_R) = R_category(i);
    
%     %Sound, RewardError
%     RE_L(i,1).matrix = intersect(temp, L_Reward);
%     RE_L(i,2).matrix = intersect(temp, L_Error);
%     RE_R(i,1).matrix = intersect(temp, R_Reward);
%     RE_R(i,2).matrix = intersect(temp, R_Error);
end
%Check the number of trials for each category
for i = 1:max(Sound_category),
    temp = find(Sound_category == i);
    number_sound_trial(i) = length(temp);
end
number_sound_trial

Block2 = find(TrialBlock(Choice_trial) == 2);
Block3 = find(TrialBlock(Choice_trial) == 3);
if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        Block_R = Block2;
        Block_L = Block3;
    else % Left -> Right
        Block_L = Block2;
        Block_R = Block3;
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        Block_R = Block2;
        Block_L = Block3;
    else % Left -> Right
        Block_L = Block2;
        Block_R = Block3;
    end
end    

%%%%get the trace for use
temp_pre = 45;
temp_post = 180;
Sound_trace = sig_trace_analysis(delta_trace, frame_sound, Choice_trial, temp_pre, temp_post);

time_window = [45:72];
for i = 1:length(roi_overlap),
    temp_neuron = roi_overlap(i);
    trace = Sound_trace(temp_neuron).matrix;
    [size_y,size_x] = size(trace);

    trace = mean(trace(:, time_window),2);
    p_kruskalwallis(i) = kruskalwallis(trace,Sound_category,'off');
    
    clear median_correct 
    for j = 1:6,
        temp_evi = find(Sound_category == j);
        temp_evi = intersect(temp_evi,Reward);
        median_correct(j) = median(trace(temp_evi));
    end
    BF_correct(i) = find(median_correct == max(median_correct),1);
    max_correct(i) = max(median_correct);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig_neuron = find(p_kruskalwallis < 0.01);
use_sound_neuron = sig_neuron;

%From overlap_181021
sig_overlap = find(stim_box(:,2) == 1);
use_sound_neuron = intersect(sig_overlap, use_sound_neuron);
    
max_neuron = max_correct(use_sound_neuron);
[~,sort_neuron] = sort(max_neuron,'descend');
use_sound_neuron = use_sound_neuron(sort_neuron);

use_neuron = roi_overlap(use_sound_neuron);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use_color = jet(max(Sound_category));
number_color = 9;
use_color = jet(number_color);
% % use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
% %              use_color(number_color-2,:); use_color(number_color-1,:); use_color(number_color,:)];
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];
use_std = 2;
smooth_factor = 2;
percentile = [1,99];
time_window = [45:72];
trace_x = [0,0.25,0.45,0.55,0.75,1];
trace_x2 = [0 0.33 0.66 1];
more_plot = 0;

%use_neuron = use_neuron([7]);
use_neuron = use_neuron([2]);
%  more_plot = 1;

%Plot the sound_trace
length_plot = min(length(use_neuron), 5);
sub_sample = 100;
hist_x = [-inf,-0.5:0.25:5,inf];
plot_hist_x = hist_x([2:(length(hist_x)-1)]) + 0.125;
plot_hist_x = [plot_hist_x(1)-0.125, plot_hist_x];

for i = 1:length_plot,
    temp_neuron = use_neuron(i);

    %Reward Error
    trace = Sound_trace(temp_neuron).matrix;
    [size_y,size_x] = size(trace);
    
    trace_sound = mean(trace(:,time_window),2);
    Block_LL = intersect(Block_L, Sound_L);
    Block_LR = intersect(Block_L, Sound_R);
    Block_RL = intersect(Block_R, Sound_L);
    Block_RR = intersect(Block_R, Sound_R);
    min_trial = min([length(Block_LL),length(Block_LR),length(Block_RL),length(Block_RR)]);
    all_LL = zeros(min_trial,sub_sample);
    all_LR = zeros(min_trial,sub_sample);
    all_RL = zeros(min_trial,sub_sample);
    all_RR = zeros(min_trial,sub_sample);
    for j = 1:sub_sample,
        [hist_LL(j,:),all_LL(:,j)] = get_trace_distribution(trace_sound,Block_LL,min_trial,hist_x);
        [hist_LR(j,:),all_LR(:,j)] = get_trace_distribution(trace_sound,Block_LR,min_trial,hist_x);
        [hist_RL(j,:),all_RL(:,j)] = get_trace_distribution(trace_sound,Block_RL,min_trial,hist_x);
        [hist_RR(j,:),all_RR(:,j)] = get_trace_distribution(trace_sound,Block_RR,min_trial,hist_x);
    
        thre_L(j,1) = search_thre_decoder2([trace_sound(Block_LL);trace_sound(Block_LR)], [zeros(length(Block_LL),1);ones(length(Block_LR),1)]);
        thre_R(j,1) = search_thre_decoder2([trace_sound(Block_RL);trace_sound(Block_RR)], [zeros(length(Block_RL),1);ones(length(Block_RR),1)]);
    end
    all_LL = reshape(all_LL,min_trial*sub_sample,1);
    all_LR = reshape(all_LR,min_trial*sub_sample,1);
    all_RL = reshape(all_RL,min_trial*sub_sample,1);
    all_RR = reshape(all_RR,min_trial*sub_sample,1);
    sum_thre_L = search_thre_decoder2([all_LL;all_LR], [zeros(sub_sample*min_trial,1);ones(sub_sample*min_trial,1)]);
    sum_thre_R = search_thre_decoder2([all_RL;all_RR], [zeros(sub_sample*min_trial,1);ones(sub_sample*min_trial,1)]);
    
    figure
    subplot(2,2,1)
    %bar(plot_hist_x,mean(hist_LL),'b','FaceAlpha',0.3)
    %plot(plot_hist_x,mean(hist_LL),'b')
    area(plot_hist_x,mean(hist_LL),'FaceColor',[0 0 1],'FaceAlpha',0.3,'EdgeAlpha',0.3)
    hold on
    %bar(plot_hist_x,mean(hist_LR),'r','FaceAlpha',0.3)
    %plot(plot_hist_x,mean(hist_LR),'r')
    area(plot_hist_x,mean(hist_LR),'FaceColor',[1 0 0],'FaceAlpha',0.3,'EdgeAlpha',0.3)
    hold on
    %plot([median(thre_L),median(thre_L)],[0 0.25],'k')
    plot([sum_thre_L,sum_thre_L],[0 0.25],'k')
    set(gca,'xlim',[-1 5.25])
    subplot(2,2,3)
    %bar(plot_hist_x,mean(hist_RL),'b','FaceAlpha',0.3)
    %plot(plot_hist_x,mean(hist_RL),'b')
    area(plot_hist_x,mean(hist_RL),'FaceColor',[0 0 1],'FaceAlpha',0.3,'EdgeAlpha',0.3)
    hold on
    %bar(plot_hist_x,mean(hist_RR),'r','FaceAlpha',0.3)
    %plot(plot_hist_x,mean(hist_RR),'r')
    area(plot_hist_x,mean(hist_RR),'FaceColor',[1 0 0],'FaceAlpha',0.3,'EdgeAlpha',0.3)
    hold on
    %plot([median(thre_R),median(thre_R)],[0 0.25],'k')
    plot([sum_thre_R,sum_thre_R],[0 0.25],'k')
    set(gca,'xlim',[-1 5.25])
    
    subplot(2,2,2)
    plot(plot_hist_x,mean(hist_LL),'b')
    hold on
    plot(plot_hist_x,mean(hist_LR),'r')
    hold on
    plot([sum_thre_L,sum_thre_L],[0 0.25],'k')
    hold on
    plot(plot_hist_x,mean(hist_RL),'b--')
    hold on
    plot(plot_hist_x,mean(hist_RR),'r--')
    hold on
    plot([sum_thre_R,sum_thre_R],[0 0.25],'k--')
    set(gca,'xlim',[-1 5.25])
    
    %Make smoothing the trace
    for j = 1:size_x,
        temp_time = [j-smooth_factor:j+smooth_factor];
        temp_time = intersect(temp_time,[1:size_x]);
        temp_trace = sum(trace(:,temp_time),2) ./ length(temp_time);
        smooth_trace(:,j) = temp_trace;
    end
    trace = smooth_trace; %use smooth trace

    trace_L_Reward = trace(L_Reward,:);
    trace_L_Error =  trace(L_Error,:);
    trace_R_Reward = trace(R_Reward,:);
    trace_R_Error =  trace(R_Error,:);
    %Sound category
    clear mean_category reward_category error_category L_category R_category
    for j = 1:max(Sound_category),
        temp_trial = find(Sound_category == j);
        mean_category(j).matrix = trace(temp_trial,:);
        
        %Sound category with reward or error
        temp_trial_reward = intersect(temp_trial,Reward);
        temp_trial_error  = intersect(temp_trial,Error);
        reward_category(j).matrix = trace(temp_trial_reward,:);
        error_category(j).matrix = trace(temp_trial_error,:);
        
        temp_trial_L = intersect(temp_trial,Block_L);
        temp_trial_R  = intersect(temp_trial,Block_R);
        L_category(j).matrix = trace(temp_trial_L,:);
        R_category(j).matrix = trace(temp_trial_R,:);
    end
    
    %Sound block
    %Block to sound
    %[activity_L, activity_R, LL_activity, LR_activity, RL_activity, RR_activity, L_category, R_category] = ...
    %      get_sound_trace_block(trace,Block_L,Block_R,Sound_L,Sound_R,Sound_category);

    [activity_L_rew, activity_R_rew, LL_activity_rew, LR_activity_rew, RL_activity_rew, RR_activity_rew, L_category_rew, R_category_rew] = ...
          get_sound_trace_block_reward(trace,Block_L,Block_R,Sound_L,Sound_R,Sound_category,Reward);

    [activity_L_err, activity_R_err, LL_activity_err, LR_activity_err, RL_activity_err, RR_activity_err, L_category_err, R_category_err] = ...
          get_sound_trace_block_reward(trace,Block_L,Block_R,Sound_L,Sound_R,Sound_category,Error);
    
    mean_reward_category = get_mean_activity_category_matrix(reward_category,time_window);
    mean_error_category  = get_mean_activity_category_matrix(error_category,time_window);
    mean_L_block_category = get_mean_activity_category_matrix(L_category,time_window);
    mean_R_block_category  = get_mean_activity_category_matrix(R_category,time_window);
    correct_L_block_category = get_mean_activity_category_matrix(L_category_rew,time_window);
    correct_R_block_category  = get_mean_activity_category_matrix(R_category_rew,time_window);
    err_L_block_category = get_mean_activity_category_matrix(L_category_err,time_window);
    err_R_block_category  = get_mean_activity_category_matrix(R_category_err,time_window);
    
    clear median_reward block_color
    for j = 1:6,
        median_reward(j) = median(mean_reward_category(j).matrix);
        %median_reward = median(mean_reward_category);
    end
    BF_neuron(i) = find(median_reward == max(median_reward),1);
    if BF_neuron(i) < 3.5,
        block_color(1,:) = [0 141 203]./255;
        block_color(2,:) = [0 0 0];
    else
        block_color(1,:) = [0 0 0];
        block_color(2,:) = [0 141 203]./255;
    end
    
    figure
    subplot(2,2,1) %reward activity
    for j = 1:max(Sound_category),
        plot_mean_se_moto(mean_category(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    subplot(2,2,2) %Reward error both included
    plot_mean_se_moto_x_matrix(mean_reward_category,trace_x,[255 102 21]./255,use_std)
    %plot_median_se_moto_x_axis_matrix(mean_reward_category,trace_x,[255 102 21]./255,use_std)
    hold on
    plot_mean_se_moto_x_matrix(mean_error_category,trace_x,[105 105 105]./255,use_std)
    %plot_median_se_moto_x_axis_matrix(mean_error_category,trace_x,[105 105 105]./255,use_std)
    set(gca,'xlim',[-0.1 1.1])
    subplot(2,2,3) %all trials
    plot_mean_se_moto_x_matrix(mean_R_block_category,trace_x,block_color(2,:),use_std)
    %plot_median_se_moto_x_axis_matrix(mean_R_block_category,trace_x,[1 0 0],use_std)
    hold on
    plot_mean_se_moto_x_matrix(mean_L_block_category,trace_x,block_color(1,:),use_std)
    %plot_median_se_moto_x_axis_matrix(mean_L_block_category,trace_x,[0 0 1],use_std)
    set(gca,'xlim',[-0.1 1.1])
    subplot(2,2,4) %correct trials
    plot_mean_se_moto_x_matrix(correct_R_block_category,trace_x,block_color(2,:),use_std)
    %plot_median_se_moto_x_axis_matrix(err_R_block_category,trace_x,[1 0 0],use_std)
    hold on
    plot_mean_se_moto_x_matrix(correct_L_block_category,trace_x,block_color(1,:),use_std)
    %plot_median_se_moto_x_axis_matrix(err_L_block_category,trace_x,[0 0 1],use_std)
    set(gca,'xlim',[-0.1 1.1])
    
    if more_plot == 1;
    figure
    subplot(2,2,1) %BLock L
    for j = 1:max(Sound_category),
        plot_mean_se_moto(L_category(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    subplot(2,2,2) %Block R
    for j = 1:max(Sound_category),
        plot_mean_se_moto(R_category(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    subplot(2,2,3) %BLock L
    for j = 1:max(Sound_category),
        plot_mean_se_moto(L_category_rew(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    subplot(2,2,4) %Block R
    for j = 1:max(Sound_category),
        plot_mean_se_moto(R_category_rew(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    end
    
    figure
    subplot(1,2,1) %reward activity
    for j = 1:max(Sound_category),
        %plot_mean_se_moto(L_category(j).matrix,use_color(j,:),use_std)
        plot_mean_se_moto(L_category_rew(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    set(gca,'ylim',[-1, 6.2])
    subplot(1,2,2) %reward activity
    for j = 1:max(Sound_category),
        %plot_mean_se_moto(R_category(j).matrix,use_color(j,:),use_std)
        plot_mean_se_moto(R_category_rew(j).matrix,use_color(j,:),use_std)
        hold on
    end
    set(gca,'xlim',[1 225],'xtick',[0:45:225])
    set(gca,'ylim',[-1, 6.2])
end

BF_neuron'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [hist_LL,trace_LL] = get_trace_distribution(trace_sound,Block_LL,min_trial,hist_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        temp_LL = randperm(length(Block_LL));
        temp_LL = temp_LL([1:min_trial]);
        temp_LL = Block_LL(temp_LL);
        trace_LL = trace_sound(temp_LL);
        hist_LL = histcounts(trace_LL, hist_x);
        hist_LL = hist_LL ./ min_trial;
        return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function thre = search_thre_decoder2(Activ, Sound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sound0 = find(Sound == 0);
Activ0 = Activ(Sound0);
Activ0 = sort(Activ0);
Activ_min = min(Activ0) - eps;
Activ_max = max(Activ0) + eps;
Activ0 = [Activ_min;Activ0;Activ_max];

for i = 1:length(Activ0);
    [correct_number(i), ~] = make_thre_decoder(Activ0(i), Activ, Sound);
end
correct_number = find(correct_number == max(correct_number),1);
thre = Activ0(correct_number);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_number, correct_matrix] = make_thre_decoder(b, Activ, Sound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp0 = find(Activ == b);
temp1 = find(Activ > b);
temp2 = find(Activ < b);

vali_answer = zeros(length(Sound),1);
vali_answer(temp1) = 1;

vali_answer2 = ~vali_answer;

correct(:,1) = vali_answer == Sound;
correct(:,2) = vali_answer2 == Sound;

correct(temp0,1) = 0.5;
correct(temp0,2) = 0.5;

sum_correct = sum(correct);
max_correct = find(sum_correct == max(sum_correct),1);

correct_matrix = correct(:,max_correct);
correct_number = max(sum_correct);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_mean_se_moto_x_matrix(trace, trace_x, trace_color,std_se)
%std_se: 0 -> plot only mean
%      : 1 -> plot with std
%      : 2 -> plot with se
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%trace
%y_axis: trials
%x_axis: time

block = length(trace);

for i = 1:block
    temp_trace = trace(i).matrix;
    %Extract nan trial
    nan_check = find(isnan(temp_trace) == 0); %detect_non_nan
    if length(temp_trace) ~= length(nan_check),
        disp('detect nan trial')
        [length(temp_trace), length(nan_check)]
    end
    temp_trace = temp_trace(nan_check);

    mean_trace(i) = mean(temp_trace);
    std_trace(i)  = std(mean_trace);
    se_trace(i) = std_trace(i) ./ (sqrt(length(temp_trace)));
end

std_plus  = mean_trace + std_trace;
std_minus = mean_trace - std_trace;
se_plus  = mean_trace + se_trace;
se_minus = mean_trace - se_trace;

temp_x = [trace_x, fliplr(trace_x)];

if std_se == 0, %mean only
plot(trace_x,mean_trace,'color',trace_color,'LineWidth',1)
box off
elseif std_se == 1, %std
%figure
%subplot(1,2,1)
fill(temp_x,[std_plus, fliplr(std_minus)],trace_color,'edgecolor','none')
alpha(0.1)
hold on
plot(trace_x,mean_trace,'color',trace_color,'LineWidth',1)
box off

elseif std_se == 2, %se
%subplot(1,2,2)
fill(temp_x,[se_plus, fliplr(se_minus)],trace_color,'edgecolor','none')
alpha(0.1)
hold on
plot(trace_x,mean_trace,'color',trace_color,'LineWidth',1)
box off

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mean_category = get_mean_activity_category_matrix(all_sound_category,time_window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:6,
    temp_category = all_sound_category(i).matrix;
    temp_category = temp_category(:,time_window);
    mean_category(i).matrix = mean(temp_category,2);
    %mean_category(i).matrix = median(temp_category,2);
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [activity_L, activity_R, LL_activity, LR_activity, RL_activity, RR_activity, L_category, R_category] = ...
          get_sound_trace_block_reward(Sound_trace,Block_L,Block_R,Sound_L,Sound_R,Sound_category,Reward)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot neural activity
%sig_sound

%Limited with only reward trials
Block_L = intersect(Block_L,Reward);
Block_R = intersect(Block_R,Reward);

Block_LL = intersect(Block_L,Sound_L);
Block_LR = intersect(Block_L,Sound_R);
Block_RL = intersect(Block_R,Sound_L);
Block_RR = intersect(Block_R,Sound_R);

%The activity is already normalize, don't need to scale
temp_activity = Sound_trace;
    
%During Block activity
activity_L = temp_activity(Block_L,:);
activity_R = temp_activity(Block_R,:);
    
%Block with sound activity
LL_activity = temp_activity(Block_LL,:);
LR_activity = temp_activity(Block_LR,:);
RL_activity = temp_activity(Block_RL,:);
RR_activity = temp_activity(Block_RR,:);
    
%Block with sound category
for j = 1:max(Sound_category),
    temp_trial = find(Sound_category == j);
    temp_trial_L = intersect(temp_trial,Block_L);
    temp_trial_L = temp_activity(temp_trial_L,:);
        
    temp_trial_R = intersect(temp_trial,Block_R);
    temp_trial_R = temp_activity(temp_trial_R,:);
        
    L_category(j).matrix = temp_trial_L;
    R_category(j).matrix = temp_trial_R;
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Sound_trace = sig_trace_analysis(delta_trace, frame_sound, Choice_trial, temp_pre, temp_post)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[trace_number,frame_length] = size(delta_trace);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalized the delta_trace_activity
mean_trace = mean(delta_trace');
mean_trace = mean_trace';
std_trace  = std(delta_trace');
std_trace = std_trace';
% for i = 1:trace_number,
%     if mean_trace(i) == 0 && std_trace(i) == 0,
%         i
%         disp('detect zero signal')
%         temp = find(delta_trace(i,:) == 0);
%         if length(temp) ~= length(delta_trace(i,:)),
%             hoge
%         end
%     else
%         delta_trace(i,:) = (delta_trace(i,:) - mean_trace(i)) ./ std_trace(i);
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for neuron = 1:trace_number,
    [neuron, trace_number]
    
    temp_sound  = zeros(length(Choice_trial),temp_pre + temp_post);
    
    for i = 1:length(Choice_trial);
        %Sound onset
        temp_frame = frame_sound(Choice_trial(i));
        if temp_frame-temp_pre+1 > 0,
            temp_sound(i,:) = delta_trace(neuron, temp_frame-temp_pre+1:temp_frame+temp_post);
        else
            temp_start_frame = -(temp_frame-temp_pre+1);
            temp_sound(i,temp_start_frame+2:temp_pre + temp_post) = delta_trace(neuron, 1:temp_frame+temp_post);
        end
    end
    Sound_trace(neuron).matrix = temp_sound;
end

return
