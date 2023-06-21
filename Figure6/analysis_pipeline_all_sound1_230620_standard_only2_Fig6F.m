%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_230620_standard_only2_Fig6F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920';

pathname2 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920\integ8_optimal_bias';

pathname3 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731');
cd(pathname3)
filename3 = dir('*.mat');

%Get the decoding performance 1
[all_all_sum_CV,all_all_sum_CV2,all_correct_tone,all_all_reward, ...
    all_opt_L_mix,all_opt_R_mix,all_opt_LR_mix,N_subject] = get_correct_reward_decoder_20230601(pathname1,analysis_folder);
cd(currentFolder);

%Get the decoding performance 1
[all2_all_sum_CV,al2_all_sum_CV2,all2_correct_tone,all2_all_reward, ...
    all2_opt_L_mix,all2_opt_R_mix,all2_opt_LR_mix,N_subject2] = get_correct_reward_decoder_20230601(pathname2,analysis_folder);
cd(currentFolder);

%check the assumption
temp = N_subject == N_subject2;
if min(temp) ~= 1
    hoge
else
    clear N_subject2
end

all_behave_correct = [];
all_behave_correct2 = [];
all_behave_reward = [];
behave_opt_L_mix = [];
behave_opt_R_mix = [];
behave_opt_LR_mix = [];
%Behavior result
for i = 1 : length(filename3)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Behavior data
    clear opt_L_mix opt_R_mix opt_LR_mix
    temp_filename = filename3(i).name;
    %temp_filename = cell2mat(temp_filename);
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    
    length_session = length(stim_opt_curve);
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

temp_x = [0:0.01:1];
figure
subplot(1,2,1)
boxplot([all_sum_opt(temp_nan), all_sum_non_opt(temp_nan)])
subplot(1,2,2)
boxplot([all_behave_opt(temp_nan), all_behave_non_opt(temp_nan)])
figure
subplot(1,2,1) %neurometric
plot_mean_se_moto_x_axis(all_opt_LR_mix(temp_nan,:),temp_x,[0 0 0],2)
hold on
plot_mean_se_moto_x_axis(all_opt_L_mix(temp_nan,:),temp_x,[0 0 1],2)
hold on
plot_mean_se_moto_x_axis(all_opt_R_mix(temp_nan,:),temp_x,[1 0 0],2)
set(gca,'xlim',[-0.1 1.1], 'ylim',[0 1])
subplot(1,2,2) %behavior
plot_mean_se_moto_x_axis(behave_opt_LR_mix(temp_nan,:),temp_x,[0 0 0],2)
hold on
plot_mean_se_moto_x_axis(behave_opt_L_mix(temp_nan,:),temp_x,[0 0 1],2)
hold on
plot_mean_se_moto_x_axis(behave_opt_R_mix(temp_nan,:),temp_x,[1 0 0],2)
set(gca,'xlim',[-0.1 1.1], 'ylim',[0 1])

%All trials
%plot_neuron_behave_correct(all_all_sum_CV(:,1), all_all_reward(:,1), all_behave_reward(:,3), all_behave_correct(:,3));
plot_neuron_behave_correct2(all_all_sum_CV(:,1),all2_all_sum_CV(:,1), all_behave_correct(:,3),...
                            all_all_reward(:,1),all2_all_reward(:,1), all_behave_reward(:,3),N_subject);
% %Easy tones
% %plot_neuron_behave_correct(all_correct_tone(:,1), all_all_reward(:,2), all_behave_reward(:,4), all_behave_correct(:,4));
% plot_neuron_behave_correct2(all_correct_tone(:,1),all2_correct_tone(:,1), all_behave_correct(:,4),...
%                             all_all_reward(:,2),all2_all_reward(:,2), all_behave_reward(:,4));
% %Mid tones
% %plot_neuron_behave_correct(all_correct_tone(:,2), all_all_reward(:,3), all_behave_reward(:,5), all_behave_correct(:,5));
% plot_neuron_behave_correct2(all_correct_tone(:,2),all2_correct_tone(:,2), all_behave_correct(:,5),...
%                             all_all_reward(:,3),all2_all_reward(:,3), all_behave_reward(:,5));
% %Dif tones
% %plot_neuron_behave_correct(all_correct_tone(:,3), all_all_reward(:,4), all_behave_reward(:,6), all_behave_correct(:,6));
% plot_neuron_behave_correct2(all_correct_tone(:,3),all2_correct_tone(:,3), all_behave_correct(:,6),...
%                             all_all_reward(:,4),all2_all_reward(:,4), all_behave_reward(:,6));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_all_sum_CV,all_all_sum_CV2,all_correct_tone,all_all_reward, ...
          all_opt_L_mix,all_opt_R_mix,all_opt_LR_mix,N_subject] = get_correct_reward_decoder_20230601(pathname1,analysis_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(pathname1)
filename1 = dir('*.mat');
%filename1

max_neuron = 140;
all_all_sum_CV = [];
all_all_sum_CV2 = [];
all_all_reward = [];
all_correct_tone = [];
all_all_r = [];
all_opt_L_mix = [];
all_opt_R_mix = [];
all_opt_LR_mix = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i).name; 
    fpath = fullfile(pathname1, temp_filename);
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
    mouse_session(i) = length_session;

    %Sound choice integrate
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            [correct_sum,correct_sum2,reward_rate,opt_mix,~] = get_correct_rate_session_20190920(stim(j).matrix);
        elseif length(analysis_folder) == 6, %reward
            [correct_sum,correct_sum2,reward_rate,opt_mix,~] = get_correct_rate_session_20190920(rew(j).matrix);
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
        
        all_sum_CV(j,:) =  [correct_sum];
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
end
%Do the mixed effect model analysis
mouse_session = cumsum(mouse_session);
for i = 1:length(mouse_session)
    if i == 1
        N_subject(1:mouse_session(i)) = i;
    else
        N_subject(mouse_session(i-1)+1:mouse_session(i)) = i+1;
    end
end
N_subject = N_subject';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_neuron_behave_correct2(correct_mix,correct_mix2, all_behave_correct,all_all_reward,all_all_reward2,all_behave_reward,N_subject)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%neuron
% correct_mix = all_all_sum_CV(:,1);
% all_all_reward = all_all_reward(:,1);
%bahavior
% all_behave_reward = all_behave_reward(:,3);
% all_behave_correct = all_behave_correct(:,3); %all trials
non_nan1 = find(isnan(correct_mix) == 0);
non_nan2 = find(isnan(correct_mix2) == 0);
non_nan = intersect(non_nan1,non_nan2);

length_x = [zeros(length(non_nan),1); ones(length(non_nan),1); ones(length(non_nan),1)*2];

figure
subplot(2,2,1)
plot(correct_mix(non_nan), all_all_reward(non_nan), 'k.')
subplot(2,2,2)
plot(all_behave_correct(non_nan), all_behave_reward(non_nan), 'k.')

subplot(2,2,3)
boxplot([all_behave_correct(non_nan); correct_mix(non_nan); correct_mix2(non_nan)], length_x)
set(gca,'ylim',[0.4 1])

subplot(2,2,4)
boxplot([all_behave_reward(non_nan); all_all_reward(non_nan); all_all_reward2(non_nan)], length_x)
set(gca,'ylim',[0.9 2.3])

figure
subplot(1,3,1)
plot(all_behave_correct(non_nan), correct_mix(non_nan), 'k.')
hold on
plot([0.4 1],[0.4 1],'k')
set(gca,'xlim',[0.4 1],'ylim',[0.4 1])
subplot(1,3,2)
plot(all_behave_correct(non_nan), correct_mix2(non_nan), 'k.')
hold on
plot([0.4 1],[0.4 1],'k')
set(gca,'xlim',[0.4 1],'ylim',[0.4 1])
subplot(1,3,3)
plot(correct_mix(non_nan), correct_mix2(non_nan), 'k.')
hold on
plot([0.4 1],[0.4 1],'k')
set(gca,'xlim',[0.4 1],'ylim',[0.4 1])

figure
subplot(1,3,1)
plot(all_behave_reward(non_nan), all_all_reward(non_nan), 'k.')
hold on
plot([1 2],[1 2],'k')
set(gca,'xlim',[1 2],'ylim',[1 2])
subplot(1,3,2)
plot(all_behave_reward(non_nan), all_all_reward2(non_nan), 'k.')
hold on
plot([1 2],[1 2],'k')
set(gca,'xlim',[1 2],'ylim',[1 2])
subplot(1,3,3)
plot(all_all_reward(non_nan), all_all_reward2(non_nan), 'k.')
hold on
plot([1 2],[1 2],'k')
set(gca,'xlim',[1 2],'ylim',[1 2])

correct_mix = double(correct_mix);
correct_mix2 = double(correct_mix2);
all_all_reward = double(all_all_reward);
all_all_reward2 = double(all_all_reward2);

length(non_nan)
N_subject = N_subject(non_nan);

p_correct = signrank(double(correct_mix2(non_nan)), double(correct_mix(non_nan)))
p_reward = signrank(double(all_all_reward2(non_nan)), double(all_all_reward(non_nan)))

lme = fitlme_analysis_20210520_0(correct_mix2(non_nan)-correct_mix(non_nan),N_subject);
lme(2).lme
lme = fitlme_analysis_20210520_0(all_all_reward2(non_nan)-all_all_reward(non_nan),N_subject);
lme(2).lme

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_sum,correct_sum2,reward_rate,opt_mix,r_original,r_shuffle] = get_correct_rate_session_20190920(stim_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stim_session
                correct_sum  = stim_session.correct_max;
                correct_sum2  = stim_session.correct_max2;
                reward_rate = stim_session.reward_max;
                opt_mix(1,:) = stim_session.opt_L_mix;
                opt_mix(2,:) = stim_session.opt_R_mix;
                opt_mix(3,:) = stim_session.opt_LR_mix;
            
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

