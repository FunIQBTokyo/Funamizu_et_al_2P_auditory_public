%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_230621_session_neurometric
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
block_rew_stim = [2,2; 2,2];             
block_rew_rew  = [3,1; 1,3];             

pathname1 = 'e:\Tone_discri1\fr00\fr00_ToneClouds_aki_block5_Oct31_2017_Session1';
%pathname1 = currentFolder;
cd(pathname1)

filename1 = dir('SLR_190920Delta_180605_1*.mat');
file_behave = dir('Single2_decode190925_*.mat');
             
if length(filename1) == 1
    filename1 = filename1.name;
    filename1 = [pathname1,'\',filename1];
    load(filename1);
end
if length(file_behave) == 1
    file_behave = file_behave.name;
    
    file_behave = [pathname1,'\',file_behave];
    data_behave = load(file_behave);
end

if length(Decode_stim.correct1) ~= 0
    [stim.matrix,CV_repeat,~] = get_decoding_score_integrate_190920(Decode_stim, data_behave.Decode_stim,block_rew_stim);
else
    stim.matrix = nan;
    CV_repeat = nan;
end
if length(Decode_rew.correct1) ~= 0
    [rew.matrix,CV_repeat,~] = get_decoding_score_integrate_190920(Decode_rew, data_behave.Decode_rew, block_rew_rew);
else
    rew.matrix = nan;
    CV_repeat = nan;
end

plot_neurometric_function(stim);
plot_neurometric_function(rew);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_neurometric_function(stim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freq_x = [0 0.25 0.45 0.55 0.75 1];
evi_x = [0:0.01:1];

right_prob_L = stim.matrix.right_prob_L;
right_prob_R = stim.matrix.right_prob_R;
Sound_Evi_L = stim.matrix.Sound_Evi_L;
Sound_Evi_R = stim.matrix.Sound_Evi_R;
conf_L = stim.matrix.conf_L;
conf_R = stim.matrix.conf_R;

opt_L_mix = stim.matrix.opt_L_mix;
opt_R_mix = stim.matrix.opt_R_mix;

figure
subplot(1,2,1)
plot_each_decode_likelihood_plot_only(right_prob_L,Sound_Evi_L,[0 0 1])
hold on
plot_each_decode_likelihood_plot_only(right_prob_R,Sound_Evi_R,[1 0 0])
hold on
h = boxplot([right_prob_L; right_prob_R],[Sound_Evi_L; Sound_Evi_R]);
set(h(7,:),'Visible','off')
set(gca,'ylim',[-0.05 1.05])

subplot(1,2,2)
plot(evi_x,opt_L_mix,'b')
for k = 1:6,
    hold on
    plot([freq_x(k),freq_x(k)],[conf_L(k,1),conf_L(k,2)],'b')
end
hold on
plot(evi_x,opt_R_mix,'r')
for k = 1:6,
    hold on
    plot([freq_x(k),freq_x(k)],[conf_R(k,1),conf_R(k,2)],'r')
end
set(gca,'xlim',[-0.1 1.1])
set(gca,'ylim',[-0.05 1.05])

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [decode_likeli,CV_repeat, s_each_neuron,check_likeli] = get_decoding_score_integrate_190920(decode, behave_data, block_rew)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Decode_integrate.Sound = Sound;
%     Decode_integrate.choice = choice;
%     Decode_integrate.reward = reward;
%     Decode_integrate.Sound_Evi = Sound_Evi;
%     Decode_integrate.Block = Block;
%     Decode_integrate.use_sig_neuron = use_sig_neuron;

Sound = behave_data.Sound;
choice = behave_data.choice;
reward = behave_data.reward;
Sound_Evi = behave_data.Sound_Evi;
Block = behave_data.Block;
% posterior = decode.binary_posterior;
decode_likeli.length_neuron = length(decode.use_sig_neuron);
decode_likeli.correct1 = decode.correct1; 
decode_likeli.correct2 = decode.correct2; 
decode_likeli.reward_rate1 = decode.reward_rate1; 
CV_repeat = decode.CV_repeat;

% r = decode.r; 
% r_s = decode.r_s; 
if isfield(decode,'r'),
    decode_likeli.r_original = decode.r;
    decode_likeli.r_shuffle = decode.r_s;
end
s_each_neuron = nan;
if isfield(decode,'s_each_neuron'),
    s_each_neuron = decode.s_each_neuron;
end

if isfield(decode,'likelihood_sound'),
    check_likeli = 1;
else
    check_likeli = 0;
end
correct_CV = decode.likelihood_sound; 
b_sound = decode.b_sound; 
[size_trial, size_CV] = size(correct_CV);

%Get the number of non-zero b
for i = 1:size_CV,
    temp_b = b_sound(i).matrix;
    clear temp_non_zero_b
    for j = 1:10,
        temp_temp_b = temp_b(j,:);
        temp_non_zero_b(j) = length(find(temp_temp_b ~= 0));
    end
    non_zero_b(i) = nanmedian(temp_non_zero_b);
end
non_zero_b = nanmedian(non_zero_b);
decode_likeli.non_zero_b = non_zero_b; 

%Get the optimal neurometic curve
block1 = find(Block == 1); %left
block2 = find(Block == 2); %right

correct_distri = make_population_neuron_decode_20230620_mix(correct_CV,Sound,reward,Sound_Evi,choice,Block,block_rew);

decode_likeli.correct_max = correct_distri.correct_sound;
decode_likeli.correct_max2 = correct_distri.correct_sound2;
decode_likeli.reward_max = correct_distri.reward_rate;

decode_likeli.opt_L_mix = correct_distri.opt_L_mix;
decode_likeli.opt_R_mix = correct_distri.opt_R_mix;
decode_likeli.opt_LR_mix = correct_distri.opt_LR_mix;

decode_likeli.likeli_mix = correct_distri.likeli_mix;
decode_likeli.right_prob = correct_distri.right_prob;

decode_likeli.right_prob_L = correct_distri.right_prob_L;
decode_likeli.right_prob_R = correct_distri.right_prob_R;
decode_likeli.Sound_Evi_L = correct_distri.Sound_Evi_L;
decode_likeli.Sound_Evi_R = correct_distri.Sound_Evi_R;

decode_likeli.conf_L = correct_distri.conf_L;
decode_likeli.conf_R = correct_distri.conf_R;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function correct_distri = make_population_neuron_decode_20230620_mix(likeli_mix,Sound,reward,Sound_Evi,choice,Block,block_rew)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%First make only for decoding result
%length_neuron = length(likeli_sound);

[size_trial,size_CV] = size(likeli_mix);
block_L = find(Block == 1);
block_R = find(Block == 2);

for i = 1:6,
    sound_evi(i).matrix = find(Sound_Evi == i);
end
sound_easy = union(sound_evi(1).matrix, sound_evi(6).matrix);
sound_mid = union(sound_evi(2).matrix, sound_evi(5).matrix);
sound_dif = union(sound_evi(3).matrix, sound_evi(4).matrix);

parfor i = 1:size_CV,
    mix_train = likeli_mix(:,i);
    
    %This time only focus on the mix_train
    correct_mix = get_single_binary_answer(mix_train); 
    
    correct_mix05 = find(correct_mix == 0.5);
    %Correct rate for sound
    temp_sound_mix = single(correct_mix == Sound);
    temp_sound_mix(correct_mix05) = 0.5;
    %sound_mix(:,i) = temp_sound_mix; %0.5 was assiged to correct in all trials!!
    
    temp_correct1 = sum(temp_sound_mix) ./ length(temp_sound_mix); %mix
    temp_easy = sum(temp_sound_mix(sound_easy)) ./ length(sound_easy); %mix
    temp_mid = sum(temp_sound_mix(sound_mid)) ./ length(sound_mid); %mix
    temp_dif = sum(temp_sound_mix(sound_dif)) ./ length(sound_dif); %mix
    
    [mix_L0, mix_L1, mix_R0, mix_R1] = ...
        get_4_correct_rate(temp_sound_mix, Sound, block_L, block_R);
    temp_correct2 = (mix_L0 + mix_L1 + mix_R0 + mix_R1) / 4;
    
    correct_rate(i,:) = [temp_correct1, temp_easy, temp_mid, temp_dif, temp_correct2];
    correct_rate2(i,:) = [mix_L0, mix_L1, mix_R0, mix_R1];
    reward_rate(i,:) = get_4_reward_rate(temp_sound_mix, Sound, block_L, block_R, Sound_Evi, block_rew);
    %reward_rate(i) = (mix_L0.*block_rew(1,1) + mix_L1.*block_rew(1,2) + mix_R0.*block_rew(2,1) + mix_R1.*block_rew(2,2)) ./ 4;
    
    %likeli_left likeli_right likeli_mix
    %Get the neurometric function individually in each random
    %Make the correct vector with same other mix
    Sound_evi_vector = [Sound_Evi(block_L); Sound_Evi(block_R)];
    correct_mix_vector = [correct_mix(block_L); correct_mix(block_R)];
    
    [opt_L_mix(i,:),  opt_R_mix(i,:),  opt_LR_mix(i,:),~,temp_conf_L,temp_conf_R] = get_neurometric4_opt_mix(correct_mix_vector,  Sound_evi_vector,block_L,block_R);
    conf_L_low(:,i) = temp_conf_L(:,1);
    conf_L_high(:,i) = temp_conf_L(:,2);
    conf_R_low(:,i) = temp_conf_R(:,1);
    conf_R_high(:,i) = temp_conf_R(:,2);

end

opt_L_mix = nanmean(opt_L_mix);
opt_R_mix = nanmean(opt_R_mix);
opt_LR_mix = nanmean(opt_LR_mix);

conf_L(:,1) = nanmean(conf_L_low,2);
conf_L(:,2) = nanmean(conf_L_high,2);
conf_R(:,1) = nanmean(conf_R_low,2);
conf_R(:,2) = nanmean(conf_R_high,2);

likeli_mix = nanmean(likeli_mix,2);

%change likelihood to right choice probability
right_prob = likeli_mix;

right_prob_L = right_prob(block_L);
right_prob_R = right_prob(block_R);
Sound_Evi_L = Sound_Evi(block_L);
Sound_Evi_R = Sound_Evi(block_R);

correct_sound = nanmean(correct_rate);
correct_sound2 = nanmean(correct_rate2);
reward_rate = nanmean(reward_rate);

correct_distri.correct_sound  = correct_sound;
correct_distri.correct_sound2 = correct_sound2;
correct_distri.reward_rate = reward_rate;

correct_distri.opt_L_mix = opt_L_mix;
correct_distri.opt_R_mix = opt_R_mix;
correct_distri.opt_LR_mix = opt_LR_mix;

correct_distri.likeli_mix = likeli_mix;
correct_distri.right_prob = right_prob;

correct_distri.right_prob_L = right_prob_L;
correct_distri.right_prob_R = right_prob_R;
correct_distri.Sound_Evi_L = Sound_Evi_L;
correct_distri.Sound_Evi_R = Sound_Evi_R;

correct_distri.conf_L = conf_L;
correct_distri.conf_R = conf_R;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y,x05,correct_evi,mean_bino,conf_bino] = get_neurometic3_conf(correct_rate, Sound_Evi, freq_x, evi_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear length_trial
for i = 1:length(freq_x),
    temp = find(Sound_Evi == i);
    temp_correct = correct_rate(temp);
    
    length_trial(i) = length(temp);
    length_correct(i) = sum(temp_correct);
%     if i < 3.5,
%         length_correct(i) = length_trial(i) - length_correct(i);
%     end
    [mean_bino(i),conf_bino(i,:)] = binofit(length_correct(i),length_trial(i));
end

b = glmfit(freq_x',[length_correct' length_trial'],'binomial','link','logit');
x05 = -b(1) ./ b(2); %threthold
y = glmval(b,evi_x,'logit');

correct_evi = length_correct ./ length_trial;
return

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_each_decode_likelihood_plot_only(likeli_mix_L,Sound_Evi_L,use_color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:6,
    temp_evi = find(Sound_Evi_L == i);
    temp_likeli = likeli_mix_L(temp_evi);
    temp_x = 0.2 * (rand(length(temp_evi),1) - 0.5);
    temp_x = temp_x + i;
    plot(temp_x,temp_likeli,'.','color',use_color)
    hold on
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function correct_rate = get_single_binary_answer(use_sound)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[length_neuron,length_trial] = size(use_sound);

binary_use_sound = round(use_sound,3);
temp_sound = zeros(length_neuron,length_trial);
%temp = find(use_sound > 0.5);
temp = find(binary_use_sound > 0.5);
temp_sound(temp) = 1;
temp = find(binary_use_sound == 0.5);
temp_sound(temp) = 0.5;
%Correct rate
correct_rate = temp_sound;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [decode_left0, decode_left1, decode_right0, decode_right1] = ...
    get_4_correct_rate(temp_decode, Sound_target, block_left, block_right)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sound_left  = Sound_target(block_left);
Sound_right = Sound_target(block_right);

temp_decode_left = temp_decode(block_left);
temp_decode_right = temp_decode(block_right);

Sound_left0 = find(Sound_left == 0);
Sound_left1 = find(Sound_left == 1);
Sound_right0 = find(Sound_right == 0);
Sound_right1 = find(Sound_right == 1);

decode_left0 = temp_decode_left(Sound_left0);
decode_left1 = temp_decode_left(Sound_left1);
decode_right0 = temp_decode_right(Sound_right0);
decode_right1 = temp_decode_right(Sound_right1);

decode_left0 = sum(decode_left0) / length(Sound_left0);
decode_left1 = sum(decode_left1) / length(Sound_left1);
decode_right0 = sum(decode_right0) / length(Sound_right0);
decode_right1 = sum(decode_right1) / length(Sound_right1);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function decode_all = get_4_reward_rate(temp_decode, Sound_target, block_left, block_right, Sound_Evi, block_rew)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:6,
    sound_evi(i).matrix = find(Sound_Evi == i);
end
sound_easy = union(sound_evi(1).matrix, sound_evi(6).matrix);
sound_mid = union(sound_evi(2).matrix, sound_evi(5).matrix);
sound_dif = union(sound_evi(3).matrix, sound_evi(4).matrix);

decode_all(1)  = get_reward_rate_trials(temp_decode, Sound_target, block_left, block_right, block_rew);

temp_block_left = intersect(block_left, sound_easy);
temp_block_right = intersect(block_right, sound_easy);
decode_all(2) = get_reward_rate_trials(temp_decode, Sound_target, temp_block_left, temp_block_right, block_rew);

temp_block_left = intersect(block_left, sound_mid);
temp_block_right = intersect(block_right, sound_mid);
decode_all(3) = get_reward_rate_trials(temp_decode, Sound_target, temp_block_left, temp_block_right, block_rew);

temp_block_left = intersect(block_left, sound_dif);
temp_block_right = intersect(block_right, sound_dif);
decode_all(4) = get_reward_rate_trials(temp_decode, Sound_target, temp_block_left, temp_block_right, block_rew);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function decode_all = get_reward_rate_trials(temp_decode, Sound_target, block_left, block_right, block_rew)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

length_trial = length(block_left) + length(block_right);

Sound_left  = Sound_target(block_left);
Sound_right = Sound_target(block_right);

temp_decode_left = temp_decode(block_left);
temp_decode_right = temp_decode(block_right);

Sound_left0 = find(Sound_left == 0);
Sound_left1 = find(Sound_left == 1);
Sound_right0 = find(Sound_right == 0);
Sound_right1 = find(Sound_right == 1);

decode_left0 = temp_decode_left(Sound_left0);
decode_left1 = temp_decode_left(Sound_left1);
decode_right0 = temp_decode_right(Sound_right0);
decode_right1 = temp_decode_right(Sound_right1);

decode_left0 = sum(decode_left0) .* block_rew(1,1);
decode_left1 = sum(decode_left1) .* block_rew(1,2);
decode_right0 = sum(decode_right0) .* block_rew(2,1);
decode_right1 = sum(decode_right1) .* block_rew(2,2);

decode_all = (decode_left0 + decode_left1 + decode_right0 + decode_right1) ./ length_trial;

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [opt_L,opt_R,opt_LR,x05,conf_bino_L,conf_bino_R] = get_neurometric4_opt_mix(correct_mix,Sound_Evi,block_L,block_R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

evi_x = [0:0.01:1];
freq_x = [0 0.25 0.45 0.55 0.75 1];
block_LR = [block_L; block_R];

[binary_likeli,  Sound_freq] = process_likeli_binary(correct_mix, Sound_Evi, freq_x,0.5);

[y_L, x05(1), correct_block_L,mean_bino_L,conf_bino_L] = get_neurometic3_conf(binary_likeli(block_L), Sound_Evi(block_L), freq_x, evi_x);
[y_R, x05(2), correct_block_R,mean_bino_R,conf_bino_R] = get_neurometic3_conf(binary_likeli(block_R), Sound_Evi(block_R), freq_x, evi_x);
[y_LR, x05(2), correct_block_LR] = get_neurometic3(binary_likeli(block_LR), Sound_Evi(block_LR), freq_x, evi_x);

opt = optimset('Display','off');
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
    
% correct_block_L
% correct_block_R
lapse_L = [correct_block_L(1), 1-correct_block_L(6)]; %limit for lapse
lapse_R = [correct_block_R(1), 1-correct_block_R(6)]; %limit for lapse
lapse_LR = [correct_block_LR(1), 1-correct_block_LR(6)]; %limit for lapse

for i = 1:10,
    [X_L(i,:),FCAL_L(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,para(i,:),opt,binary_likeli(block_L), Sound_freq(block_L), evi_x, lapse_L);
    [X_R(i,:),FCAL_R(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,para(i,:),opt,binary_likeli(block_R), Sound_freq(block_R), evi_x, lapse_R);
    [X_LR(i,:),FCAL_LR(i),EXITFLAG,OUTPUT] = fminsearch(@Opt_psychometric,para(i,:),opt,binary_likeli(block_LR), Sound_freq(block_LR), evi_x, lapse_LR);
end

min_L = find(FCAL_L == min(FCAL_L),1);
min_R = find(FCAL_R == min(FCAL_R),1);
min_LR = find(FCAL_LR == min(FCAL_LR),1);
X_L = X_L(min_L,:);
X_R = X_R(min_R,:);
X_LR = X_LR(min_LR,:);

[log_likelli,opt_L] = Opt_psychometric_max(X_L, binary_likeli(block_L), Sound_freq(block_L), evi_x, lapse_L);
[log_likelli,opt_R] = Opt_psychometric_max(X_R, binary_likeli(block_R), Sound_freq(block_R), evi_x, lapse_R);
[log_likelli,opt_LR] = Opt_psychometric_max(X_LR, binary_likeli(block_LR), Sound_freq(block_LR), evi_x, lapse_LR);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [binary_likeli,Sound_freq] = process_likeli_binary(likeli_mix, Sound_Evi, freq_x,thre)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Make binarize at thre
likeli_mix = round(likeli_mix,3);
temp05 = find(likeli_mix == thre);
temp1 = find(likeli_mix > thre);
%temp0 = find(likeli_mix < thre);
binary_likeli = zeros(length(likeli_mix),1);
binary_likeli(temp1) = 1;
binary_likeli(temp05) = 0.5;

binary_likeli2 = binary_likeli;

clear Sound_freq
binary_value = 0;
for i = 1:6,
    temp = find(Sound_Evi == i);
    Sound_freq(temp) = freq_x(i);

    %check 0.5
    temp_binary = binary_likeli2(temp);
    temp_binary1 = find(temp_binary == 0.5);
    temp_binary2(i) = length(find(temp_binary == 0.5));
    for j = 1:length(temp_binary1),
        temp_trial = temp(temp_binary1(j));
        binary_likeli2(temp_trial) = binary_value;
        binary_value = not(binary_value);
    end
end
%temp_binary2
temp = find(binary_likeli == 0.5);
%if length(temp) > 10,
if length(temp) ~= 0,
    %[binary_likeli(temp), binary_likeli2(temp)]
    [sum(binary_likeli(temp)), sum(binary_likeli2(temp))]
end
binary_likeli = binary_likeli2;
clear binary_likeli2

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y,x05,correct_evi] = get_neurometic3(correct_rate, Sound_Evi, freq_x, evi_x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear length_trial
for i = 1:length(freq_x),
    temp = find(Sound_Evi == i);
    temp_correct = correct_rate(temp);
    
    length_trial(i) = length(temp);
    length_correct(i) = sum(temp_correct);
%     if i < 3.5,
%         length_correct(i) = length_trial(i) - length_correct(i);
%     end
end

b = glmfit(freq_x',[length_correct' length_trial'],'binomial','link','logit');
x05 = -b(1) ./ b(2); %threthold
y = glmval(b,evi_x,'logit');

correct_evi = length_correct ./ length_trial;
return

