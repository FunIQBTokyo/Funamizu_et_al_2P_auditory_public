%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function behave_20230614_get_parameter_each_kaiseki

%Stimulus
path1 = 'E:\Tone_discri1\all_mouse_behavior\behave_20230612_get_parameter\stimulus';
reward = [2,2;2,2;2,2;2,2];
%Reward
% path1 = 'E:\Tone_discri1\all_mouse_behavior\behave_20230612_get_parameter\reward';
% reward = [2,2;3,1;1,3;2,2];

cd(path1)
filename1 = dir('*.mat');
filename1

order_LR = [];
length_trial = [];
log_likelli = [];
para_all = [];
para_L = [];
para_R = [];
para_4 = [];
opt_trace_all = [];
opt_trace_L = [];
opt_trace_R = [];
opt_trace_4 = [];
right_trial_all = [];
right_trial_L = [];
right_trial_R = [];
right_trial_4 = [];
number_trial_all = [];
number_trial_L = [];
number_trial_R = [];
number_trial_4 = [];

%     save(save_file{i},'order_LR','length_trial','log_likelli',...
%         'para_all','para_L','para_R','para_4',...
%         'opt_trace_all','opt_trace_L','opt_trace_R','opt_trace_4',...
%         'right_trial_all','right_trial_L','right_trial_R','right_trial_4',...
%         'number_trial_all','number_trial_L','number_trial_R','number_trial_4');    

mouse_number = [];
for i = 1:length(filename1)
    temp_filename = filename1(i).name; 
    fpath = fullfile(path1, temp_filename);
    data = load(fpath);

    order_LR = [order_LR; data.order_LR];
    length_trial = [length_trial; data.length_trial];
    log_likelli = [log_likelli; data.log_likelli];

    para_all = [para_all; data.para_all];
    para_L = [para_L; data.para_L];
    para_R = [para_R; data.para_R];
    para_4 = [para_4; data.para_4];
    opt_trace_all = [opt_trace_all; data.opt_trace_all];
    opt_trace_L = [opt_trace_L; data.opt_trace_L];
    opt_trace_R = [opt_trace_R; data.opt_trace_R];
    opt_trace_4 = [opt_trace_4; data.opt_trace_4];
    
    right_trial_all = [right_trial_all; data.right_trial_all];
    right_trial_L = [right_trial_L; data.right_trial_L];
    right_trial_R = [right_trial_R; data.right_trial_R];
    right_trial_4 = [right_trial_4; data.right_trial_4];
    number_trial_all = [number_trial_all; data.number_trial_all];
    number_trial_L = [number_trial_L; data.number_trial_L];
    number_trial_R = [number_trial_R; data.number_trial_R];
    number_trial_4 = [number_trial_4; data.number_trial_4];
    
    length_session = length(data.order_LR);
    
    %Based on length_neuron, make the mouse session lines
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];
end
%para: bias, sensitivity, lapse1, lapse2
session_number = [1:length(mouse_number)]';
mouse_number = [mouse_number,session_number];

x_hist = [-inf,-0.5:0.1:0.5,inf];
ave_L = mean(opt_trace_L,2);
ave_R = mean(opt_trace_R,2);
sabun_LR = ave_R - ave_L;
hist_LR = histcounts(sabun_LR,x_hist);

evi_x = [0:0.01:1];
figure
subplot(1,2,1)
plot(evi_x,mean(opt_trace_L),'b')
hold on
plot(evi_x,mean(opt_trace_R),'r')
subplot(1,2,2)
bar(hist_LR)

%Compare the correct rate and reward performance
[correct_rate(:,1), reward_per_trial(:,1)] = get_correct_rate_reward(right_trial_all, number_trial_all, reward(1,:));
[correct_rate(:,2), reward_per_trial(:,2)] = get_correct_rate_reward(right_trial_L, number_trial_L, reward(2,:));
[correct_rate(:,3), reward_per_trial(:,3)] = get_correct_rate_reward(right_trial_R, number_trial_R, reward(3,:));
[correct_rate(:,4), reward_per_trial(:,4)] = get_correct_rate_reward(right_trial_4, number_trial_4, reward(4,:));

%Make the block2 and block3
para2 = para_R;
para3 = para_L;
correct2 = correct_rate(:,3); %R
correct3 = correct_rate(:,2); %L
reward2 = reward_per_trial(:,3); %R
reward3 = reward_per_trial(:,2); %L
opt2 = opt_trace_R; %R
opt3 = opt_trace_L; %L

temp = find(order_LR == 1);
for i = 1:length(temp)
    para2(temp(i),:) = para_L(temp(i),:);
    para3(temp(i),:) = para_R(temp(i),:);
    
    correct2(temp(i),:) = correct_rate(temp(i),2); %L
    correct3(temp(i),:) = correct_rate(temp(i),3); %R
    reward2(temp(i),:) = reward_per_trial(temp(i),2); %L
    reward3(temp(i),:) = reward_per_trial(temp(i),3); %R
    
    opt2(temp(i),:) = opt_trace_L(temp(i),:); %R
    opt3(temp(i),:) = opt_trace_R(temp(i),:); %L
end

correct_rate(:,5) = correct2;
correct_rate(:,6) = correct3;
reward_per_trial(:,5) = reward2;
reward_per_trial(:,6) = reward3;

opt_vector(:,1) = mean(opt_trace_all,2);
opt_vector(:,2) = mean(opt_trace_L,2);
opt_vector(:,3) = mean(opt_trace_R,2);
opt_vector(:,4) = mean(opt_trace_4,2);
opt_vector(:,5) = mean(opt2,2);
opt_vector(:,6) = mean(opt3,2);

%BAsed on LR correct rate, (-1 and 1)
%Find the order of LR
correct_LR = [correct_rate(:,2);correct_rate(:,3)]; %L,R
reward_LR = [reward_per_trial(:,2);reward_per_trial(:,3)]; %L,R
opt_LR = [opt_vector(:,2);opt_vector(:,3)]; %L,R

x_LR = [ones(length(correct_rate),1) * -1; ones(length(correct_rate),1)];
temp = find(order_LR == -1); %L->R
x_23 = ones(length(correct_rate),1);
x_23(temp) = -1;
temp = -1 * x_23;
x_23 = [x_23; temp];
mouse_number = [mouse_number; mouse_number];

temp1 = ones(length(x_LR),1);
temp_x = [x_LR,x_23,mouse_number];
%Regress

disp('regress correct rate')
[b,dev,stats] = glmfit([x_LR,x_23],correct_LR,'normal','Link','identity');
b
stats.p(2)
stats.p(3)
p_AIC_BIC = fitlme_analysis_20230612_2_regress([correct_LR,temp_x]);
p_AIC_BIC(2)
p_AIC_BIC(3)

disp('regress reward amount')
[b,dev,stats] = glmfit([x_LR,x_23],reward_LR,'normal','Link','identity');
b
stats.p(2)
stats.p(3)
p_AIC_BIC = fitlme_analysis_20230612_2_regress([reward_LR,temp_x]);
p_AIC_BIC(2)
p_AIC_BIC(3)

disp('regress psychometric')
[b,dev,stats] = glmfit([x_LR,x_23],opt_LR,'normal','Link','identity');
b
stats.p(2)
stats.p(3)
p_AIC_BIC = fitlme_analysis_20230612_2_regress([opt_LR,temp_x]);
p_AIC_BIC(2)
p_AIC_BIC(3)

% [b,bint,r,rint,stats] = regress(correct_LR,[temp1,x_LR,x_23],0.01);
% bint = bint(:,1).*bint(:,2);
% [b, bint]
% 
% [b,bint,r,rint,stats] = regress(reward_LR,[temp1,x_LR,x_23],0.01);
% bint = bint(:,1).*bint(:,2);
% [b, bint]
% 
% [b,bint,r,rint,stats] = regress(opt_LR,[temp1,x_LR,x_23],0.01);
% bint = bint(:,1).*bint(:,2);
% [b, bint]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_rate, reward_per_trial] = get_correct_rate_reward(right_trial, number_trial, reward)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_session,~] = size(right_trial)

for session = 1:size_session
    use_right = right_trial(session,:);
    use_number = number_trial(session,:);
    
    clear correct_trial reward_sum
    correct_trial = use_right;
    for i = 1:3
        correct_trial(i) = use_number(i) - use_right(i);
    end
    correct_rate(session,1) = sum(correct_trial) ./ sum(use_number);
    %reward amount
    for i = 1:3
        reward_sum(i) = correct_trial(i) * reward(1);
    end
    for i = 4:6
        reward_sum(i) = correct_trial(i) * reward(2);
    end
    reward_per_trial(session,1) = sum(reward_sum) ./ sum(use_number);
end
return

