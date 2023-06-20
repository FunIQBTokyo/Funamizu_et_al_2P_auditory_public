%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Figure1D_20230530_get_parameter_full_model_reward

current_folder = pwd;
folder{1} = 'E:\Tone_discri1\a023\reward5';
folder{2} = 'E:\Tone_discri1\a025\reward5';
folder{3} = 'E:\Tone_discri1\a026\reward5';
folder{4} = 'E:\Tone_discri1\a030\reward5';
folder{5} = 'E:\Tone_discri1\fr00\reward5';
folder{6} = 'E:\Tone_discri1\fr02\reward5';

% [filename1, pathname1,findex]=uigetfile('*.mat','Block_mat','Multiselect','on');
% filename1

x_evi_plot = [0:0.01:1];

count = 0;
for i = 1:length(folder)
    temp_path = folder{i};
    cd(temp_path)
    filename1 = dir('block_mat*.mat');
    filename1
    
    mouse_session(i) = length(filename1);
    for filecount = 1 : length(filename1)
        clear data temp_filename temp_pass fpath
    
        temp_filename = filename1(filecount).name; 
        fpath = fullfile(temp_path, temp_filename);
        %load(fpath);
    
        count = count + 1;
        [opt_L(count,:), opt_R(count,:), x_L(count,:),x_R(count,:)] = ...
            behave_analysis_block1_170222_plot4_full_model(fpath);
        close all
    end
end
cd(current_folder)

count

mean_y_L = mean(opt_L,2);
mean_y_R = mean(opt_R,2);

sabun_RL = mean_y_R - mean_y_L;
hist_x = [-inf,-0.5:0.1:0.5,inf];
length(hist_x)
hist_sabun = histcounts(sabun_RL,hist_x);
figure
bar(hist_sabun)
set(gca,'xlim',[0 length(hist_x)])

sabun_RL
count
signrank(sabun_RL)

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

% size(sabun_RL)
% size(N_subject)

lme = fitlme_analysis_20210520_0(sabun_RL,N_subject);
%lme(1).lme
lme(2).lme

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [opt_L, opt_R, X_L, X_R] = behave_analysis_block1_170222_plot4_full_model(filename1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evi_x = [0:0.01:1];
freq_x = [0 0.25 0.45 0.55 0.75 1];

%[filename1, pathname1]=uigetfile('*.mat','Block_mat');
load(filename1)

Choice_trial = find(Outcome == 1 | Outcome == 2);

% %Outcome
% outcome_EW     = 0; %early withdrawal
% outcome_IC     = 1; %incorrect choice
% outcome_reward = 2; %reward was dispensed (either automatically in early training, or after correct choice)
% outcome_NC     = 3; %no choice was made and time elapsed
% outcome_UN     = 4; %undefined or Free water:

temp_evi = unique(EvidenceStrength);
temp_evi_low  = 0.5 - temp_evi/2;
temp_evi_high = 0.5 + temp_evi/2;
temp_evi_all = [temp_evi_low', temp_evi_high'];
use_evidence = temp_evi + 0.1; %0.3 0.6 1
use_evidence = [0, use_evidence'];
tone_evidence = sort(temp_evi_all);

%Put tone evidence in all trials;
trial_evidence = zeros(length(Outcome),1);
left  = find(Correct_side == 0);
right = find(Correct_side == 1);
for i = 1:length(temp_evi),
    temp = find(EvidenceStrength == temp_evi(i));
    temp_left  = intersect(temp,left);
    temp_right = intersect(temp,right);
    trial_evidence(temp_left)  = temp_evi_low(i);
    trial_evidence(temp_right) = temp_evi_high(i);
end

use_trial_all = [1 : length(Outcome)];
use_trial1 = find(TrialBlock == 1);
use_trial2 = find(TrialBlock == 2);
use_trial3 = find(TrialBlock == 3);
use_trial4 = find(TrialBlock == 4);
use_trial1 = intersect(use_trial1,use_trial_all);
use_trial2 = intersect(use_trial2,use_trial_all);
use_trial3 = intersect(use_trial3,use_trial_all);
use_trial4 = intersect(use_trial4,use_trial_all);

if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        block_R = use_trial2;
        block_L = use_trial3;
        
    else % Left -> Right
        block_L = use_trial2;
        block_R = use_trial3;
        
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        block_R = use_trial2;
        block_L = use_trial3;
        
    else % Left -> Right
        block_L = use_trial2;
        block_R = use_trial3;
        
    end
end   

opt = optimset('Display','off');
% para = [1 1 0 0
%         1 0.2 0 0
%         0.5 0.2 0 0
%         3 1 0 0
%         0.1 0.2 0 0
%         2 0.5 0 0
%         5 0 0.1 0.1
%         1 0.1 0 0
%         1 0.3 0.1 0.1
%         0.5 0.5 0.1 0.1];
para = [-100 100 0 0
        -1000 1000 0 0
        -inf inf 0 0
        -10 10 0 0
        -10000 10000 0 0
        -1 1 0 0
        -100 100 0.1 0.1
        -1000 1000 0.1 0.1
        -10000 10000 0.1 0.1
        -inf inf 0.1 0.1];
    
%Standard fit
[choice_stim_all, right_trial_all, number_trial_all] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial_all);
[choice_stim2, right_trial2, number_trial2] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, block_L);
[choice_stim3, right_trial3, number_trial3] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, block_R);

right_prob_L = right_trial2 ./ number_trial2;
right_prob_R = right_trial3 ./ number_trial3;
lapse_L = [right_prob_L(1), 1-right_prob_L(6)]; %limit for lapse
lapse_R = [right_prob_R(1), 1-right_prob_R(6)]; %limit for lapse

%need to update the block_L and block_R
block_L = intersect(block_L, Choice_trial);
block_R = intersect(block_R, Choice_trial);

for i = 1:10,
    [X_L(i,:),FCAL_L(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,para(i,:),opt,Chosen_side(block_L), trial_evidence(block_L), evi_x, lapse_L);
    [X_R(i,:),FCAL_R(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,para(i,:),opt,Chosen_side(block_R), trial_evidence(block_R), evi_x, lapse_R);
end

min_L = find(FCAL_L == min(FCAL_L),1);
min_R = find(FCAL_R == min(FCAL_R),1);
X_L = X_L(min_L,:);
X_R = X_R(min_R,:);

[log_likelli,opt_L] = Opt_psychometric_max(X_L, Chosen_side(block_L), trial_evidence(block_L), evi_x, lapse_L);
[log_likelli,opt_R] = Opt_psychometric_max(X_R, Chosen_side(block_R), trial_evidence(block_R), evi_x, lapse_R);

b_all = glmfit(tone_evidence',[right_trial_all' number_trial_all'],'binomial','link','logit');
b2 = glmfit(tone_evidence',[right_trial2' number_trial2'],'binomial','link','logit');
b3 = glmfit(tone_evidence',[right_trial3' number_trial3'],'binomial','link','logit');
p_fit_all = glmval(b_all,evi_x,'logit');
p_fit2 = glmval(b2,evi_x,'logit');
p_fit3 = glmval(b3,evi_x,'logit');

figure
subplot(1,2,1)
plot(evi_x, opt_L, 'b')
hold on
plot(evi_x, opt_R, 'r')
subplot(1,2,2)
plot(evi_x, p_fit2, 'b')
hold on
plot(evi_x, p_fit3, 'r')
hold on
plot(freq_x, right_trial2./number_trial2, 'b.')
hold on
plot(freq_x, right_trial3./number_trial3, 'r.')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function log_likelli = Opt_psychometric(para, y, tone_evi, evi_x, lapse_limit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[log_likelli,~] = Opt_psychometric_max(para, y, tone_evi, evi_x, lapse_limit);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [log_likelli,neurometric] = Opt_psychometric_max(para, y, tone_evi, evi_x, lapse_limit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%para(1): beta0
%para(2): beta1
%para(3): lambda1
%para(4): lambda2

%4 parameters
%Yn = lambda1 + (1-lambda1-lambda2)/(1+exp(-bx))

if para(1) > 0, %para(1) shoud be minus
    para(1) = 0;
end
if para(2) < 0, %para(2) shoud be positive
    para(2) = 0;
end
if para(3) < 0,
    para(3) = 0;
elseif para(3) > lapse_limit(1)
    para(3) = lapse_limit(1);
end
if para(4) < 0,
    para(4) = 0;
elseif para(4) > lapse_limit(2)
    para(4) = lapse_limit(2);
end

temp_data = para(1) + para(2) * tone_evi;
temp_data = -temp_data;
temp_data = 1 + exp(temp_data);
temp_data = (1-para(3)-para(4))./temp_data;
temp_data = para(3) + temp_data;

y0 = find(y == 0); %make likelihood
temp_data(y0) = 1-temp_data(y0);

log_likelli = log(temp_data);
log_likelli = sum(log_likelli);
log_likelli = -log_likelli;

%get the tuning function with evi_x
temp_data = para(1) + para(2) * evi_x;
temp_data = -temp_data;
temp_data = 1 + exp(temp_data);
temp_data = (1-para(3)-para(4))./temp_data;
neurometric = para(3) + temp_data;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Select use trials and make figures
function [choice_stim, right_number_trial, number_trial] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, trials)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Select choice
%Center lick error should not count for the block trial!!
Choice_trial = find(Outcome == 1 | Outcome == 2);
Choice_trial = intersect(Choice_trial, trials); %trials selection

Correct_trial = Correct_side(Choice_trial);
Chosen_trial = Chosen_side(Choice_trial);
Evidence_trial = EvidenceStrength(Choice_trial);

length(Choice_trial)
clear right_prob
clear choice_stim
choice_stim = [];
for i = 1:2,
    temp = find(Correct_trial == i-1);
    temp_select = Chosen_trial(temp);
    temp_evidence = Evidence_trial(temp);
    
    for j = 1:length(use_evidence)-1,
        temp_trial = find(temp_evidence > use_evidence(j) & temp_evidence < use_evidence(j+1));
        temp_trial = temp_select(temp_trial);
        temp_correct = find(temp_trial == 1);
        right_prob(i,j) = length(temp_correct) / length(temp_trial);
        right_trial(i,j) = length(temp_correct);
        number_trial(i,j) = length(temp_trial);
        
        %Keep the choice stim record
        temp_stim = (length(use_evidence)-1) * (i-1) + j;
        temp_choice_stim = [temp_trial, ones(length(temp_trial),1) * temp_stim];
        choice_stim = [choice_stim; temp_choice_stim];
    end
end

plot_right_prob = [right_prob(1,3),right_prob(1,2),right_prob(1,1),right_prob(2,1),right_prob(2,2),right_prob(2,3)]
%Left reward to Right reward
right_number_trial = [right_trial(1,3),right_trial(1,2),right_trial(1,1),right_trial(2,1),right_trial(2,2),right_trial(2,3)]
number_trial = [number_trial(1,3),number_trial(1,2),number_trial(1,1),number_trial(2,1),number_trial(2,2),number_trial(2,3)]
% figure
% plot(plot_left_prob)
% set(gca,'xlim',[0 7],'ylim',[0 1])

return
