%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190920_standard_only_prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Decode_20190920','Multiselect','on');
% filename1
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920\integ20200404_with_est_block';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

pathname3 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731');
cd(pathname3)
filename3 = dir('*.mat');
% all_correct_random = [];
% all_correct_CV = [];

max_neuron = 140;
all_all_sum_CV = [];
all_all_sum_CV2 = [];
all_all_reward = [];
all_correct_tone = [];
all_all_r = [];
all_behave_correct = [];
all_behave_correct2 = [];
all_behave_reward = [];
all_opt_L_mix = [];
all_opt_R_mix = [];
all_opt_LR_mix = [];
behave_opt_L_mix = [];
behave_opt_R_mix = [];
behave_opt_LR_mix = [];

mouse_number = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    
    clear correct_same correct_other correct_mix
    clear y_L y_R x05 
    clear correct_random correct_block
    clear correct_same correct_other correct_mix
    clear correct_b_same correct_b_mix correct_b_other
    clear all_sum_CV all_shuffle_CV all_sum_train all_shuffle_train
    clear all_r all_sum_CV2 all_reward correct_tone
    clear opt_L_mix opt_R_mix opt_LR_mix
    length_session = length(stim);
    
    %Based on length_neuron, make the mouse session lines
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];

    %Sound choice integrate
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            [correct_sum,correct_sum2,reward_rate,opt_mix,~] = get_correct_rate_session_20190920(stim(j).matrix);
%             [correct_S,correct_S2,reward_rate_S,opt_S,r_ori_S,r_S] = get_correct_rate_session_20190920(stim_S(j).matrix);
%             [correct_L,correct_L2,reward_rate_L,opt_L,r_ori_L,r_L] = get_correct_rate_session_20190920(stim_L(j).matrix);
        elseif length(analysis_folder) == 6, %reward
            [correct_sum,correct_sum2,reward_rate,opt_mix,~] = get_correct_rate_session_20190920(rew(j).matrix);
%             [correct_S,correct_S2,reward_rate_S,opt_S,r_ori_S,r_S] = get_correct_rate_session_20190920(rew_S(j).matrix);
%             [correct_L,correct_L2,reward_rate_L,opt_L,r_ori_L,r_L] = get_correct_rate_session_20190920(rew_L(j).matrix);
        else
            hoge
        end
        opt_L_mix(j,:) = opt_mix(1,:);
        opt_R_mix(j,:) = opt_mix(2,:);
        opt_LR_mix(j,:) = opt_mix(3,:);
        
        if length(correct_sum) > 1,
            correct_tone(j,:) = correct_sum(2:4);
        else
            correct_tone(j,:) = nan(1,3);
        end
        correct_sum = correct_sum(1);
        
%        all_sum_CV(j,:) =  [correct_sum, correct_S, correct_L];
%        all_r(j,:) = [r_ori_S, r_ori_L, r_S, r_L];
        all_sum_CV(j,:) =  [correct_sum];
%        all_r(j,:) = [r_ori_S, r_ori_L];
        all_sum_CV2(j,:) = correct_sum2;
        all_reward(j,:) =  reward_rate;
    end
    all_all_sum_CV = [all_all_sum_CV; all_sum_CV];
    all_all_sum_CV2 = [all_all_sum_CV2; all_sum_CV2];
    all_correct_tone = [all_correct_tone; correct_tone];
    all_all_reward = [all_all_reward; all_reward];
%    all_all_r = [all_all_r; all_r];
    
    all_opt_L_mix = [all_opt_L_mix; opt_L_mix];
    all_opt_R_mix = [all_opt_R_mix; opt_R_mix];
    all_opt_LR_mix = [all_opt_LR_mix; opt_LR_mix];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Behavior data
    clear opt_L_mix opt_R_mix opt_LR_mix
    temp_filename = filename3(i).name;
    %temp_filename = cell2mat(temp_filename);
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    if length(analysis_folder) == 8, %Stimulus
        behave_correct  = stim_correct_sum;
        behave_correct2 = stim_correct_sum2;
        reward_sum2 = stim_reward_sum;
        for j = 1:length_session,
            temp = stim_opt_curve(j).matrix;
            opt_L_mix(j,:) = temp(1,:);
            opt_R_mix(j,:) = temp(2,:);
            opt_LR_mix(j,:) = temp(3,:);
        end
    elseif length(analysis_folder) == 6, %reward
        behave_correct  = rew_correct_sum;
        behave_correct2 = rew_correct_sum2;
        reward_sum2 = rew_reward_sum;
        for j = 1:length_session,
            temp = rew_opt_curve(j).matrix;
            opt_L_mix(j,:) = temp(1,:);
            opt_R_mix(j,:) = temp(2,:);
            opt_LR_mix(j,:) = temp(3,:);
        end
    else
        hoge
    end
    all_behave_correct = [all_behave_correct; behave_correct];
    all_behave_correct2 = [all_behave_correct2; behave_correct2];
    all_behave_reward = [all_behave_reward; reward_sum2];
    
    behave_opt_L_mix = [behave_opt_L_mix; opt_L_mix];
    behave_opt_R_mix = [behave_opt_R_mix; opt_R_mix];
    behave_opt_LR_mix = [behave_opt_LR_mix; opt_LR_mix];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%About noise correlation
mean_CV = mean(all_all_sum_CV,2);
temp_nan = find(isnan(mean_CV) == 0);

length(temp_nan)

%neuron
all_sum_opt = (all_all_sum_CV2(:,1) + all_all_sum_CV2(:,4)) ./ 2;
all_sum_non_opt = (all_all_sum_CV2(:,2) + all_all_sum_CV2(:,3)) ./ 2;
%bahavior
all_behave_opt = (all_behave_correct2(:,1) + all_behave_correct2(:,4)) ./ 2;
all_behave_non_opt = (all_behave_correct2(:,2) + all_behave_correct2(:,3)) ./ 2;

all_all_sum_CV = double(all_all_sum_CV);
all_all_reward = double(all_all_reward);
all_behave_reward = double(all_behave_reward);
all_behave_correct = double(all_behave_correct);
% %All trials
% plot_neuron_behave_correct(all_all_sum_CV(:,1), all_all_reward(:,1), all_behave_reward(:,3), all_behave_correct(:,3));
% %Easy tones
% %plot_neuron_behave_correct(all_correct_tone(:,1), all_all_reward(:,2), all_behave_reward(:,4), all_behave_correct(:,4));
% %Mid tones
% %plot_neuron_behave_correct(all_correct_tone(:,2), all_all_reward(:,3), all_behave_reward(:,5), all_behave_correct(:,5));
% %Dif tones
% %plot_neuron_behave_correct(all_correct_tone(:,3), all_all_reward(:,4), all_behave_reward(:,6), all_behave_correct(:,6));

%Get sabun for behavior and neurometric
sabun_behavior = mean(behave_opt_R_mix,2) - mean(behave_opt_L_mix,2);
sabun_neuron = mean(all_opt_R_mix,2) - mean(all_opt_L_mix,2);
%signrank(sabun_behavior)
%signrank(sabun_neuron)
signrank(sabun_behavior,sabun_neuron)
%[sabun_behavior,sabun_neuron]

% lme = fitlme_analysis_20210520_0(sabun_behavior,mouse_number);
% lme(2).lme
% lme = fitlme_analysis_20210520_0(sabun_neuron,mouse_number);
% lme(2).lme
lme = fitlme_analysis_20210520_0(sabun_behavior-sabun_neuron,mouse_number);
lme(2).lme

figure
plot(sabun_behavior,sabun_neuron,'k.')
hold on
plot([-0.2 1],[-0.2 1],'k')
set(gca,'xlim',[-0.2 1],'ylim',[-0.2 1])



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_neuron_behave_correct(correct_mix, all_all_reward, all_behave_reward, all_behave_correct)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%neuron
% correct_mix = all_all_sum_CV(:,1);
% all_all_reward = all_all_reward(:,1);
%bahavior
% all_behave_reward = all_behave_reward(:,3);
% all_behave_correct = all_behave_correct(:,3); %all trials
non_nan = find(isnan(correct_mix) == 0);
length(non_nan)

figure
subplot(2,2,1)
plot(correct_mix(non_nan), all_all_reward(non_nan), 'k.')
subplot(2,2,2)
plot(all_behave_correct(non_nan), all_behave_reward(non_nan), 'k.')

subplot(2,2,3)
plot(all_behave_correct(non_nan), correct_mix(non_nan), 'k.')
hold on
plot([0.4 1],[0.4 1],'k')
set(gca,'xlim',[0.4 1],'ylim',[0.4 1])
subplot(2,2,4)
plot(all_behave_reward(non_nan), all_all_reward(non_nan), 'k.')
hold on
plot([0.8 2.2],[0.8 2.2],'k')
set(gca,'xlim',[0.8 2.2],'ylim',[0.8 2.2])

%p-value
[median(all_behave_correct(non_nan)), median(correct_mix(non_nan))]
[median(all_behave_reward(non_nan)), median(all_all_reward(non_nan))]
signrank(all_behave_correct(non_nan), correct_mix(non_nan))
signrank(all_behave_reward(non_nan), all_all_reward(non_nan))

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_sum,correct_sum2,reward_rate,opt_mix,r_original,r_shuffle] = get_correct_rate_session_20190920(stim_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stim_session
%             if isfield(stim_session,'distri_sound'),
                correct_sum  = stim_session.correct_max;
                correct_sum2  = stim_session.correct_max2;
                reward_rate = stim_session.reward_max;
                opt_mix(1,:) = stim_session.opt_L_mix;
                opt_mix(2,:) = stim_session.opt_R_mix;
                opt_mix(3,:) = stim_session.opt_LR_mix;
%             else
%                 correct_sum = nan;
%                 correct_sum2 = nan;
%                 reward_rate = nan;
%                 opt_mix = nan(3,101);
%             end
            
            if isfield(stim_session,'r_original'),
                r_original  = stim_session.r_original;
                r_shuffle  = stim_session.r_shuffle;
                r_original = mean(r_original);
                r_shuffle = mean(r_shuffle);
            else
                r_original = nan;
                r_shuffle = nan;
            end            
return
