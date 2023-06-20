%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Figure1C_20230324_get_parameter_one_session

[filename1, pathname1,findex]=uigetfile('*.mat','Block_mat');
load(filename1)

para_number = 4;
x_evi_plot = [0:0.01:1];

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
% plot_stim_choice(Outcome, Correct_side, Chosen_side, EvidenceStrength, use_evidence, use_trial_all)

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

%%%BinominalÇ≈fittingÇ∑ÇÈÅBLogistic regression
b_all = glmfit(tone_evidence',[right_trial_all' number_trial_all'],'binomial','link','logit');
b2 = glmfit(tone_evidence',[right_trial2' number_trial2'],'binomial','link','logit');
b3 = glmfit(tone_evidence',[right_trial3' number_trial3'],'binomial','link','logit');
yfit_all = glmval(b_all,tone_evidence','logit','size',number_trial_all');
yfit2 = glmval(b2,tone_evidence','logit','size',number_trial2');
yfit3 = glmval(b3,tone_evidence','logit','size',number_trial3');
p_fit_all = yfit_all./number_trial_all';
p_fit2 = yfit2./number_trial2';
p_fit3 = yfit3./number_trial3';

p_fit_all2 = glmval(b_all,x_evi_plot,'logit');
p_fit2_2 = glmval(b2,x_evi_plot,'logit');
p_fit3_2 = glmval(b3,x_evi_plot,'logit');

temp_p_fit_all = b_all(1) + b_all(2) .* tone_evidence;
temp_p_fit2 = b2(1) + b2(2) .* tone_evidence;
temp_p_fit3 = b3(1) + b3(2) .* tone_evidence;
temp_p_fit_all = exp(temp_p_fit_all) ./ (1+exp(temp_p_fit_all));
temp_p_fit2 = exp(temp_p_fit2) ./ (1+exp(temp_p_fit2));
temp_p_fit3 = exp(temp_p_fit3) ./ (1+exp(temp_p_fit3));

%95%confidence interval
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
if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        bias2 = ones(length(choice_trial2),1) * -1;
        bias3 = ones(length(choice_trial3),1);
        y05_ori(2) = -b2(1)/b2(2);
        y05_ori(1) = -b3(1)/b3(2);
    else % Left -> Right
        bias2 = ones(length(choice_trial2),1);
        bias3 = ones(length(choice_trial3),1) * -1;
        y05_ori(1) = -b2(1)/b2(2);
        y05_ori(2) = -b3(1)/b3(2);
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        bias2 = ones(length(choice_trial2),1) * -1;
        bias3 = ones(length(choice_trial3),1);
        y05_ori(2) = -b2(1)/b2(2);
        y05_ori(1) = -b3(1)/b3(2);
    else % Left -> Right
        bias2 = ones(length(choice_trial2),1);
        bias3 = ones(length(choice_trial3),1) * -1;
        y05_ori(1) = -b2(1)/b2(2);
        y05_ori(2) = -b3(1)/b3(2);
    end
end    
bias23 = [bias2;bias3];
bias_evidence23 = use_evidence23 .* bias23;

%Model parameter estimate
temp_thre  = ones(length(choice_trial23),1);
temp_lapse = ones(length(choice_trial23),1);
bias_lapse = bias23;
temp_x = [temp_thre, use_evidence23, temp_lapse, bias23, bias_evidence23, bias_lapse];

use_para = [1 0 0 0 0 0;
            1 1 0 0 0 0;
            1 1 1 0 0 0;
            1 1 0 1 0 0;
            1 1 0 1 1 0;
            1 1 0 0 1 0;
            1 1 1 1 0 0;
            1 1 1 1 1 0;
            1 1 1 1 0 1;
            1 1 1 1 1 1;
            1 1 1 0 1 0;
            1 1 1 0 1 1;
            1 1 1 0 0 1
            ]; 
        
use_para = use_para(para_number,:);

[size_y,size_x] = size(use_para);

init_para = zeros(1,6);
for i = 1:size_y,
    i
    temp_para = use_para(i,:);
    [para_max(i,:), ave_likeli(i), BIC_all(i), log_likelihood(i,:)] = ...
        BIC_logistic_regression2(temp_x,use_choice23,temp_para,init_para);
    log_likeilihood_all(i) = sum(log_likelihood(i,:));
end

ave_likeli
BIC_all
para_max

%Draw the model based psychometric curve
%Plot the modeled behavior
for i = 1:size_y,
    [temp_plot,y05] = Opt_psychometric_plot(para_max(i,:),x_evi_plot);
    figure
    plot(x_evi_plot,temp_plot(1,:),'b')
    hold on
    plot(x_evi_plot,temp_plot(2,:),'r')
    set(gca,'xlim',[-0.1 1.1])
    set(gca,'ylim',[0 1])
    %set(gca,'xtick',[0:0.05:1])
    set(gca,'xtick',[0:0.1:1])
    set(gca,'ytick',[0:0.1:1])
    xlabel('[high tones - low tones]/s')
    ylabel('Fraction rightward')
end

figure
plot_psycho_curve2(p_right_all, conf_all, p_fit_all2, tone_evidence,x_evi_plot,[0,0,0])

%figure
hold on
if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        plot_psycho_curve2(p_right2, conf2, temp_plot(2,:), tone_evidence+0.01, x_evi_plot,[1,0,0])
        hold on
        plot_psycho_curve2(p_right3, conf3, temp_plot(1,:), tone_evidence-0.01,x_evi_plot,[0,0,1])
    else % Left -> Right
        plot_psycho_curve2(p_right2, conf2, temp_plot(1,:), tone_evidence-0.01, x_evi_plot,[0,0,1])
        hold on
        plot_psycho_curve2(p_right3, conf3, temp_plot(2,:), tone_evidence+0.01,x_evi_plot,[1,0,0])
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        plot_psycho_curve2(p_right2, conf2, temp_plot(2,:), tone_evidence+0.01, x_evi_plot,[1,0,0])
        hold on
        plot_psycho_curve2(p_right3, conf3, temp_plot(1,:), tone_evidence-0.01,x_evi_plot,[0,0,1])
    else % Left -> Right
        plot_psycho_curve2(p_right2, conf2, temp_plot(1,:), tone_evidence-0.01, x_evi_plot,[0,0,1])
        hold on
        plot_psycho_curve2(p_right3, conf3, temp_plot(2,:), tone_evidence+0.01,x_evi_plot,[1,0,0])
    end
end    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [use_para, ave_likeli, BIC_all, log_likelihood] = BIC_logistic_regression2(temp_x,use_choice23,use_para,init_para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log(likelihood) = Tn*logYn + (1-Tn)*log(1-Yn)
%Yn = 1/(1+exp(-bx))
%Tn is the answer
%Above equations are from Bishopè„, pp. 205

number_para = find(use_para == 1);
init_para = init_para(number_para);
init_para0 = find(init_para == 0);

opt = optimset('Display','off');
%for i = 1 : 3,
for i = 1 : 100,
    rand_para = rand(1,length(number_para));
    [X_all(i,:),FCAL_all(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,rand_para,opt,...
        temp_x, use_choice23, use_para);
    
    if isreal(FCAL_all(i)) == 0,
        FCAL_all(i) = 0;
    end
end

temp_FCAL = find(FCAL_all == min(FCAL_all),1);
para_max = X_all(temp_FCAL,:);

%Get maximum prediction
[ave_likeli,likelihood] = Opt_psychometric_max(para_max, temp_x, use_choice23, use_para);

%Update the use_para with real parameter
temp_para = find(use_para ~= 0);
use_para(temp_para) = para_max;

if use_para(3) < 0,
    use_para(3) = 0;
elseif use_para(3) > 0.5
    use_para(3) = 0.5;
end
if use_para(6) < 0,
    use_para(6) = 0;
elseif use_para(6) > 0.5
    use_para(6) = 0.5;
end

%use_para = use_para
%ave_likeli = ave_likeli
BIC_all = -2 * sum(log(likelihood)) + length(para_max) * log(length(use_choice23));
log_likelihood = log(likelihood);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [temp_p, y05] = Opt_psychometric_plot(para, temp_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%temp_x = [temp_thre, use_evidence23, temp_lapse, bias23, bias_evidence23, bias_lapse];
%Yn = 1/(1+exp(-bx))

use_bias = 1;
for i = 1:length(temp_x)
    temp_data = [1,temp_x(i),1,use_bias,use_bias*temp_x(i),use_bias];

    exp_para = para([1,2,4,5]);
    temp_exp = temp_data([1,2,4,5]);
    temp_exp = sum(temp_exp .* exp_para);
    temp_exp = 1 ./ (1 + exp(-temp_exp));

    temp_lapse = para([3,6]) .* temp_data([3,6]);
    temp_lapse = sum(temp_lapse);
    temp_p(1,i) = temp_lapse + (1-2.*temp_lapse) .* temp_exp;
end
use_bias = -1;
for i = 1:length(temp_x)
    temp_data = [1,temp_x(i),1,use_bias,use_bias*temp_x(i),use_bias];

    exp_para = para([1,2,4,5]);
    temp_exp = temp_data([1,2,4,5]);
    temp_exp = sum(temp_exp .* exp_para);
    temp_exp = 1 ./ (1 + exp(-temp_exp));

    temp_lapse = para([3,6]) .* temp_data([3,6]);
    temp_lapse = sum(temp_lapse);
    temp_p(2,i) = temp_lapse + (1-2.*temp_lapse) .* temp_exp;
end

%Get the value for y_05
%temp_x = [temp_thre, use_evidence23, temp_lapse, bias23, bias_evidence23, bias_lapse];
use_bias = 1;
y05(1) = -(para(1) + para(4) * use_bias) ./ (para(2) + para(5) * use_bias);
use_bias = -1;
y05(2) = -(para(1) + para(4) * use_bias) ./ (para(2) + para(5) * use_bias);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ave_likeli,likelihood] = Opt_psychometric_max(para, temp_x, use_choice23, use_para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

para_non_zero = find(use_para ~= 0);
temp_para = zeros(1,length(use_para));
temp_para(para_non_zero) = para;

if temp_para(3) < 0,
    temp_para(3) = 0;
elseif temp_para(3) > 0.5
    temp_para(3) = 0.5;
end
if temp_para(6) < 0,
    temp_para(6) = 0;
elseif temp_para(6) > 0.5
    temp_para(6) = 0.5;
end

%temp_x = [temp_thre, use_evidence23, temp_lapse, bias23, bias_evidence23, bias_lapse];
%Yn = 1/(1+exp(-bx))

[N_trial,size_x] = size(temp_x);

likelihood = zeros(1,N_trial);

para_data = repmat(temp_para,N_trial,1);
exp_para = para_data(:,[1,2,4,5]);

temp_exp = temp_x(:,[1,2,4,5]);
temp_exp = sum(temp_exp .* exp_para, 2);
temp_exp = 1 ./ (1 + exp(-temp_exp));

%Decide the lapse parameter
temp_lapse = zeros(N_trial,1);
if use_para(6) ~= 0,
    temp1 = find(temp_x(:,6) == 1); %left
    temp2 = find(temp_x(:,6) == -1); %right
    temp_lapse(temp1) = para_data(temp1,3);
    temp_lapse(temp2) = para_data(temp2,6);
else
    temp_lapse = para_data(:,3);
end
% temp_lapse = para_data(:,[3,6]) .* temp_x(:,[3,6]);
% temp_lapse = sum(temp_lapse,2);
temp_p = temp_lapse + (1-2.*temp_lapse) .* temp_exp;

% %Correct likelihood between 0 and 1
% temp1 = find(temp_p <= 0);
% temp2 = find(temp_p >= 1);
% temp_p(temp2) = 1-eps;

temp1 = find(use_choice23 == 1);
temp2 = find(use_choice23 == 2);
likelihood(temp1) = 1-temp_p(temp1);
likelihood(temp2) = temp_p(temp2);

if length(temp1)+length(temp2) ~= N_trial,
    [length(temp1),length(temp2),N_trial]
    hoge
end

%likelihood keisan
log_likeli = sum(log(likelihood));
log_likeli = log_likeli / N_trial;
ave_likeli = exp(log_likeli);
ave_likeli = -ave_likeli;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ave_likeli_mean = Opt_psychometric(para, temp_x, use_choice23, use_para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ave_likeli_mean,~] = Opt_psychometric_max(para, temp_x, use_choice23, use_para);

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
set(gca,'xtick',[0:0.1:1])
set(gca,'ytick',[0:0.1:1])
xlabel('[high tones - low tones]/s')
ylabel('Fraction rightward')

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

return
