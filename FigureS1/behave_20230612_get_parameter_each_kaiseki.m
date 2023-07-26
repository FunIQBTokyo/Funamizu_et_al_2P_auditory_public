%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function behave_20230612_get_parameter_each_kaiseki

%Stimulus
% path1 = 'E:\Tone_discri1\all_mouse_behavior\behave_20230612_get_parameter\stimulus';
% reward = [2,2;2,2;2,2;2,2];
%Reward
path1 = 'E:\Tone_discri1\all_mouse_behavior\behave_20230612_get_parameter\reward';
reward = [2,2;3,1;1,3;2,2];

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

temp = find(order_LR == 1);%L->R
for i = 1:length(temp)
    para2(temp(i),:) = para_L(temp(i),:);
    para3(temp(i),:) = para_R(temp(i),:);
    
    correct2(temp(i)) = correct_rate(temp(i),2); %L
    correct3(temp(i)) = correct_rate(temp(i),3); %R
    reward2(temp(i)) = reward_per_trial(temp(i),2); %L
    reward3(temp(i)) = reward_per_trial(temp(i),3); %R
    
    opt2(temp(i),:) = opt_trace_L(temp(i),:); %R
    opt3(temp(i),:) = opt_trace_R(temp(i),:); %L
end

correct_rate(:,5) = correct2;
correct_rate(:,6) = correct3;
reward_per_trial(:,5) = reward2;
reward_per_trial(:,6) = reward3;

disp('check the order of LR')
temp1 = find(order_LR == 1);
temp_1 = find(order_LR == -1);
if length(temp1) + length(temp_1) ~= length(order_LR)
    hoge
end
mean(order_LR)
signrank(order_LR)

median(correct_rate)
median(reward_per_trial)
median(length_trial)

thre4 = 100; %Threshold the sessions which have more than 100 trials in block4
use_trial = find(length_trial(:,4) > thre4);
[length(use_trial), length(length_trial)]

opt_vector(:,1) = mean(opt_trace_all,2);
opt_vector(:,2) = mean(opt_trace_L,2);
opt_vector(:,3) = mean(opt_trace_R,2);
opt_vector(:,4) = mean(opt_trace_4,2);
opt_vector(:,5) = mean(opt2,2);
opt_vector(:,6) = mean(opt3,2);

%test
test1 = correct_rate(:,2)-correct_rate(:,3);
test2 = correct_rate(:,5)-correct_rate(:,6);
test = abs(test1) == abs(test2);
if min(test) ~= 1
    test
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%start analysis
disp('Correct rate')
analysis_correct_rate_session(correct_rate,mouse_number,use_trial,[-0.25 0.25]);
disp('Reward amount')
analysis_correct_rate_session(reward_per_trial,mouse_number,use_trial,[-0.5 0.5]);
disp('Psychometric')
analysis_correct_rate_session(opt_vector,mouse_number,use_trial,[-0.6 0.6]);

% 
% disp('Focus on the bias')
% bias_L = para_L(:,1);
% bias_R = para_R(:,1);
% [median(bias_L), median(bias_R)]
% figure
% subplot(1,2,1)
% compare2parameters(bias_L,bias_R,mouse_number);
% 
% %Focus on the sensitivity
% disp('sensitivity of LR')
% sense_L = para_L(:,2);
% sense_R = para_R(:,2);
% [median(sense_L), median(sense_R)]
% subplot(1,2,2)
% compare2parameters(sense_L,sense_R,mouse_number);
% 
% disp('Focus on the sensitivity234')
% figure
% compare3parameters(para2(:,2),para3(:,2),para_4(:,2),mouse_number,use_trial);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_correct_rate_session(correct_rate,mouse_number,use_trial,y_lim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Correct performance Between L and R and 4 blocks
disp('Comparison Between L and R blocks')
% figure
% subplot(1,2,1)
%compare2parameters(correct_rate(:,2),correct_rate(:,3),mouse_number); %L and R
disp('compare with LR and post')
compare3parameters(correct_rate(:,2),correct_rate(:,3),correct_rate(:,4),mouse_number,use_trial,y_lim); %L and R
% subplot(1,2,2)
% compare2parameters(reward_per_trial(:,2),reward_per_trial(:,3),mouse_number); %L and R
% compare3parameters(reward_per_trial(:,2),reward_per_trial(:,3),reward_per_trial(:,4),mouse_number,use_trial);

disp('Comparison Between 2 and 3 4 blocks')
%compare2parameters(correct_rate(:,5),correct_rate(:,6),mouse_number); %L and R
compare3parameters(correct_rate(:,5),correct_rate(:,6),correct_rate(:,4),mouse_number,use_trial,y_lim);
% compare3parameters(reward_per_trial(:,5),reward_per_trial(:,6),reward_per_trial(:,4),mouse_number,use_trial);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compare2parameters(para1,para2,mouse_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size(para1)
p(1) = signrank(para1-para2);
lme = fitlme_analysis_20210520_0(para1-para2,mouse_number);
p(2) = lme(2).lme.Coefficients.pValue;
p

figure
temp_x = 1 + 0.1*(rand(length(para1),1) - 0.5);
boxplot(para1-para2)
hold on
plot(temp_x,para1-para2,'k.')
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function compare3parameters(para1,para2,para3,mouse_number,use_trial,y_lim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size(para1)
%Between L and R, don't need to remove the sessions
box0 = para1-para2;
figure
subplot(1,3,1)
temp_x = 1 + 0.1*(rand(length(box0),1) - 0.5);
boxplot(box0)
hold on
plot(temp_x,box0,'k.')
set(gca,'ylim',y_lim)

p_moto(1) = signrank(para1-para2);
lme = fitlme_analysis_20210520_0(para1-para2,mouse_number);
p_moto(2) = lme(2).lme.Coefficients.pValue;
p_moto

%Remove sessions based on block4
para1 = para1(use_trial);
para2 = para2(use_trial);
para3 = para3(use_trial);
size(para1)
mouse_number = mouse_number(use_trial);

temp_x = 1 + 0.1*(rand(length(para1),1) - 0.5);

p(1,1) = signrank(para1-para2);
p(1,2) = signrank(para3-para2);
p(1,3) = signrank(para1-para3);
lme = fitlme_analysis_20210520_0(para1-para2,mouse_number);
p(2,1) = lme(2).lme.Coefficients.pValue;
lme = fitlme_analysis_20210520_0(para3-para2,mouse_number);
p(2,2) = lme(2).lme.Coefficients.pValue;
lme = fitlme_analysis_20210520_0(para1-para3,mouse_number);
p(2,3) = lme(2).lme.Coefficients.pValue;
p
p(2,2)
p(2,3)

box0 = para1-para2;
box1 = para1-para3;
box2 = para2-para3;
x1 = ones(length(box1),1);
x2 = ones(length(box2),1)*2;

subplot(1,3,2)
temp_x = 1 + 0.1*(rand(length(box0),1) - 0.5);
boxplot(box0)
hold on
plot(temp_x,box0,'k.')
set(gca,'ylim',y_lim)

subplot(1,3,3)
boxplot([box1;box2],[x1;x2]);
temp_x1 = 1 + 0.1*(rand(length(box1),1) - 0.5);
temp_x2 = 2 + 0.1*(rand(length(box2),1) - 0.5);
hold on
plot(temp_x1,box1,'k.')
hold on
plot(temp_x2,box2,'k.')
set(gca,'ylim',y_lim)

return

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

