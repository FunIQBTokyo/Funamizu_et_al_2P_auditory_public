%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function FigureS1_behave_analysis_block1_170222_plot2

[filename1, pathname1]=uigetfile('*.mat','Block_mat');
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

%Plot trial series
plot_stim_choice(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial_all)
plot_stim_choice_prob(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial_all)

% use_evidence
% temp_evi
% hoge

[choice_stim_all, right_trial_all, number_trial_all] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial_all);
[choice_stim1, right_trial1, number_trial1] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial1);
[choice_stim2, right_trial2, number_trial2] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial2);
[choice_stim3, right_trial3, number_trial3] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial3);
[choice_stim4, right_trial4, number_trial4] = ...
    psycho_plot2(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial4);

b_all = glmfit(tone_evidence',[right_trial_all' number_trial_all'],'binomial','link','logit');
b2 = glmfit(tone_evidence',[right_trial2' number_trial2'],'binomial','link','logit');
b3 = glmfit(tone_evidence',[right_trial3' number_trial3'],'binomial','link','logit');
yfit_all = glmval(b_all,tone_evidence','logit','size',number_trial_all');
yfit2 = glmval(b2,tone_evidence','logit','size',number_trial2');
yfit3 = glmval(b3,tone_evidence','logit','size',number_trial3');
p_fit_all = yfit_all./number_trial_all';
p_fit2 = yfit2./number_trial2';
p_fit3 = yfit3./number_trial3';

x_evi_plot = [0:0.01:1];
p_fit_all2 = glmval(b_all,x_evi_plot,'logit');
p_fit2_2 = glmval(b2,x_evi_plot,'logit');
p_fit3_2 = glmval(b3,x_evi_plot,'logit');

temp_p_fit_all = b_all(1) + b_all(2) .* tone_evidence;
temp_p_fit2 = b2(1) + b2(2) .* tone_evidence;
temp_p_fit3 = b3(1) + b3(2) .* tone_evidence;
temp_p_fit_all = exp(temp_p_fit_all) ./ (1+exp(temp_p_fit_all));
temp_p_fit2 = exp(temp_p_fit2) ./ (1+exp(temp_p_fit2));
temp_p_fit3 = exp(temp_p_fit3) ./ (1+exp(temp_p_fit3));

%Estimate the 0.5 point for tone_evidence
y_05(1) = -b_all(1)/b_all(2);
y_05(2) = -b2(1)/b2(2);
y_05(3) = -b3(1)/b3(2);
y_05

%95%confidence interval‚àŒvŽZ‚·‚éB
for i = 1:length(tone_evidence),
    [p_right_all(i),conf_all(i,:)] = binofit(right_trial_all(i), number_trial_all(i), 0.05);
    [p_right2(i),conf2(i,:)] = binofit(right_trial2(i), number_trial2(i), 0.05);
    [p_right3(i),conf3(i,:)] = binofit(right_trial3(i), number_trial3(i), 0.05);
end

%Use only block2 and block3 to fit the regression model
%Choice_trial
choice_trial2 = intersect(use_trial2,Choice_trial);
choice_trial3 = intersect(use_trial3,Choice_trial);
choice_trial23 = [choice_trial2; choice_trial3];
use_evidence23 = trial_evidence(choice_trial23);
use_choice23   = Chosen_side(choice_trial23) + 1;
bias2 = ones(length(choice_trial2),1) * -1;
bias3 = ones(length(choice_trial3),1);
bias23 = [bias2;bias3];
bias_evidence23 = use_evidence23 .* bias23;
temp_x = [use_evidence23, bias23]; %Only bias terms

[B,dev,stats] = mnrfit(temp_x,use_choice23);
p1 = stats.p(1);
p_evi = stats.p(2)
p_bias = stats.p(3)
temp_x_data = B(1) + B(2)*use_evidence23 + B(3)*bias23;
temp_x_data = exp(temp_x_data) ./ (1+exp(temp_x_data));
figure
boxplot(temp_x_data,use_choice23)

%Plot
%All
figure
plot_psycho_curve2(p_right_all, conf_all, p_fit_all2, tone_evidence,x_evi_plot,[0,0,0])

figure
if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        plot_psycho_curve2(p_right2, conf2, p_fit2_2, tone_evidence, x_evi_plot,[1,0,0])
        hold on
        plot_psycho_curve2(p_right3, conf3, p_fit3_2, tone_evidence,x_evi_plot,[0,0,1])
    else % Left -> Right
        plot_psycho_curve2(p_right2, conf2, p_fit2_2, tone_evidence, x_evi_plot,[0,0,1])
        hold on
        plot_psycho_curve2(p_right3, conf3, p_fit3_2, tone_evidence,x_evi_plot,[1,0,0])
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        plot_psycho_curve2(p_right2, conf2, p_fit2_2, tone_evidence, x_evi_plot,[1,0,0])
        hold on
        plot_psycho_curve2(p_right3, conf3, p_fit3_2, tone_evidence,x_evi_plot,[0,0,1])
    else % Left -> Right
        plot_psycho_curve2(p_right2, conf2, p_fit2_2, tone_evidence, x_evi_plot,[0,0,1])
        hold on
        plot_psycho_curve2(p_right3, conf3, p_fit3_2, tone_evidence,x_evi_plot,[1,0,0])
    end
end    

%Kentei
block2 = ones(length(choice_stim2),1) * 2;
block3 = ones(length(choice_stim3),1) * 3;

kentei23 = [choice_stim2; choice_stim3];
block23 = [block2; block3];

%Kentei should not be ANOVA
p = anovan(kentei23(:,1),{kentei23(:,2), block23(:,1)});

p(2)
%[sum(number_trial_all), length(Outcome)]
sum(number_trial_all)

BlockTrial
BlockProb_Left = 1-BlockProb %Proportion of left trials
BlockReward

%Stim task
if BlockProb_Left(2) > BlockProb_Left(3),
    [y_05(2), y_05(3)]
elseif BlockProb_Left(2) < BlockProb_Left(3),
    [y_05(3), y_05(2)]
end
%Reward task
if BlockReward(2,1) > BlockReward(3,1),
    [y_05(2), y_05(3)]
elseif BlockReward(2,1) < BlockReward(3,1),
    [y_05(3), y_05(2)]
end

p_right_all
length(find(Outcome == 0)) %Check the center lick error

unique(EvidenceStrength)'
unique(Intensity)'

sum(right_trial_all)/sum(number_trial_all)

%Correct rate for left and right
correct_lr(1)  = (sum(number_trial_all(1:3)) - sum(right_trial_all(1:3))) / sum(number_trial_all(1:3));
correct_lr(2) = sum(right_trial_all(4:6)) / sum(number_trial_all(4:6));

correct_lr

right_trial_all./number_trial_all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_stim_choice(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, trials)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Choice_trial = find(Outcome == 1 | Outcome == 2);
Choice_trial = intersect(Choice_trial, trials); %trials selection
Correct_trial = Correct_side(Choice_trial);
Chosen_trial = Chosen_side(Choice_trial);
Evidence_trial = EvidenceStrength(Choice_trial);
%stim_strength
for i = 1:length(use_evidence)-1,
    temp = find(Evidence_trial > use_evidence(i) & Evidence_trial < use_evidence(i+1));
    Evidence_stim(temp) = i;
end

length(Choice_trial)

trial_color2 = [0,0,1;1,0,0];
figure
for i = 1:2,
    temp_trial = find(Correct_trial == i-1); %0 Left 1 Right
    temp_stim = Evidence_stim(temp_trial);
    temp_choice = Chosen_trial(temp_trial);
    
    for j = 1:length(temp_trial),
        temp_x = [temp_trial(j), temp_trial(j)];
        temp_y = [1,0]-temp_choice(j);
        temp_y = temp_y * temp_stim(j); %Bar hight shows the difficulty
        line(temp_x, temp_y,'color',trial_color2(i,:),'LineWidth',1);
        hold on
    end
end
set(gca,'xlim',[1,length(Choice_trial)],'ylim',[-4,4])
set(gca,'xtick',[0:40:length(Choice_trial)])

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_stim_choice_prob(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, trials)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Choice_trial = find(Outcome == 1 | Outcome == 2);
Choice_trial = intersect(Choice_trial, trials); %trials selection
Correct_trial = Correct_side(Choice_trial);
Chosen_trial = Chosen_side(Choice_trial);

%Get the closest 9 trials to calculate the probability of left or right
%choice probability
use_trial = [1:length(Choice_trial)];
left_reward = find(Correct_trial == 0);
right_reward = find(Correct_trial == 1);
ave_trial = 40;
std_norm = 40;
%Chosen_trial
%Gaussian is required

for i = 1:length(Choice_trial),
    temp_trial = abs(i - use_trial);
    [~,temp_trial] =  sort(temp_trial);
    temp_correct = Correct_trial(temp_trial);
    temp_left  = find(temp_correct == 0);
    temp_right = find(temp_correct == 1);
    
    distance_left = temp_left(1:ave_trial);
    distance_right = temp_right(1:ave_trial);
    temp_left_trial = temp_trial(temp_left(1:ave_trial));
    temp_right_trial = temp_trial(temp_right(1:ave_trial));
    
    temp_left  = Chosen_trial(temp_left_trial);
    temp_right = Chosen_trial(temp_right_trial);
    
    filter_left = normpdf([0:ave_trial-1],0,std_norm);
    filter_left = filter_left ./ sum(filter_left);
    filter_right = normpdf([0:ave_trial-1],0,std_norm);
    filter_right = filter_right ./ sum(filter_right);
    
    size(filter_left)
    size(temp_left)
    if length(temp_left) ~= 0,
        left_prob(i)  = 1 - sum(temp_left' .* filter_left);
    end
    if length(temp_right) ~= 0,
        right_prob(i) = sum(temp_right' .* filter_right);
    end
end

figure
plot(left_prob,'b')
hold on
plot(right_prob,'r')
set(gca,'xlim',[1,length(Choice_trial)],'ylim',[0.5,1])
set(gca,'xtick',[0:40:length(Choice_trial)])

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_psycho_curve2(p_right_all, conf_all, p_fit_all, tone_evidence, x_evi_plot, plot_color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(tone_evidence', p_right_all','.','color',plot_color) %Mean
for i = 1:length(tone_evidence),
    hold on
    plot([tone_evidence(i),tone_evidence(i)], [conf_all(i,1),conf_all(i,2)],'-','color',plot_color)
end
hold on
plot(x_evi_plot,p_fit_all,'-','LineWidth',1,'color',plot_color)

set(gca,'xlim',[-0.1 1.1])
set(gca,'ylim',[0 1])
%set(gca,'xtick',[0:0.05:1])
set(gca,'xtick',[0:0.1:1])
set(gca,'ytick',[0:0.1:1])
xlabel('[high tones - low tones]/s')
ylabel('Fraction rightward')

return

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

return
