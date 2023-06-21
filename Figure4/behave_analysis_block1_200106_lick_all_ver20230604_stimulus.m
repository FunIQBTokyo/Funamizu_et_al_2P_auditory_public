%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

%function B = behave_analysis_block1_200106_lick_all_ver20230604
function behave_analysis_block1_200106_lick_all_ver20230604_stimulus

current_folder = pwd;
folder{1} = 'E:\Tone_discri1\a023\block5';
folder{2} = 'E:\Tone_discri1\a025\block5';
folder{3} = 'E:\Tone_discri1\a026\block5';
folder{4} = 'E:\Tone_discri1\a030\block5';
folder{5} = 'E:\Tone_discri1\fr00\block5';
folder{6} = 'E:\Tone_discri1\fr02\block5';

% [filename1, pathname1,findex]=uigetfile('*.mat','Block_mat','Multiselect','on');
% filename1

%time_bin = 10;
time_bin = 5;
count = 0;
for i = 1:length(folder)
    temp_path = folder{i};
    cd(temp_path)
    filename1 = dir('block_mat*.mat');
    filename1
    
    mouse_session(i) = length(filename1);
    for filecount = 1 : length(filename1)
    clear data temp_filename temp_pass fpath
    clear Reward_LCR Outcome Correct_side Chosen_side
    clear EvidenceStrength TrialBlock
    clear reward_trial Stim_prob
    
    temp_filename = filename1(filecount).name; 
    fpath = fullfile(temp_path, temp_filename);
    %data = load(fpath);
%    data = load(temp_filename);

    count = count + 1;
    [B(count,:), tone_left(count,:), tone_right(count,:), ...
        Evi_tone(count,:), length_trial(count,:), B_prior(count,:)] = ...
    behave_analysis_block1_200106_lick_each(fpath, time_bin);

    %B(filecount,:) = -B(filecount,:);
    end
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


mean(Evi_tone)
mean(length_trial)

[size_session,size_B] = size(B);
%mean_B = mean(B)
mean_B = median(B)
figure
subplot(2,1,1)
for i = 1:length(filename1),
    plot(B(i,[2:size_B]),'color',[0.5 0.5 0.5])
    hold on
end
plot(mean_B(2:size_B))
%set(gca,'xlim',[0.5 6.5])
set(gca,'xlim',[0.5 12.5])

subplot(2,1,2)
plot_median_se_moto(B(:,[2:size_B]),[0 0 0],1)
set(gca,'xlim',[0.5 12.5])


p(1) = kruskalwallis(B(:,[2:size_B]),[],'off');

figure
plot(mean(tone_left),'b')
hold on
plot(mean(tone_right),'r')
set(gca,'xlim',[0.5 12.5])

figure
boxplot(B(:,[2:size_B]))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot about B_prior
mean_B_prior = median(B_prior)
figure
subplot(1,2,1)
% for i = 1:length(filename1),
%     plot(B_prior(i,[2:size_B]),'color',[0.5 0.5 0.5])
%     hold on
% end
% plot(B_prior(2:size_B))
% %set(gca,'xlim',[0.5 6.5])
plot_median_se_moto(B_prior(:,[2:size_B]),[0 0 0],1)
set(gca,'xlim',[0.5 12.5])

subplot(1,2,2)
boxplot(B_prior(:,size_B+1))
hold on
temp_x = 1+(rand(size_session,1)-0.5) * 0.1; 
plot(temp_x, B_prior(:,size_B+1), 'k.')

p(2) = kruskalwallis(B_prior(:,[2:size_B]),[],'off');
figure
boxplot(B_prior(:,[2:size_B]))

p

for i = 1:size_B-1,
    p_each(i) = signrank(B(:,i+1));
    p_each_prior(i) = signrank(B_prior(:,i+1));
    
    lme = fitlme_analysis_20210520_0(B_prior(:,i+1),N_subject);
    lme(2).lme;
    p_lme(i) = lme(2).lme.Coefficients.pValue;
end
%p_each
p_each_prior
p_lme

%Focus on the p_each_prior


temp_x = [25:50:575];
temp_x0 = ones(length(temp_x),1);
%Get slope for each B_prior(:,[2:size_B])
for i = 1 : length(filename1)
    temp_B = B_prior(i,[2:size_B]);
    temp_regress_B = regress(temp_B', [temp_x', temp_x0]);
    regress_B(i,:) = temp_regress_B';
    y_0_600(i,1) = regress_B(i,2);
    y_0_600(i,2) = 600 * regress_B(i,1) + regress_B(i,2);
    for j = 1:length(temp_x),
        y_bin(i,j) = temp_x(j) * regress_B(i,1) + regress_B(i,2);
    end
end
median_regress_B = median(regress_B);
median_regress_B(1)
median_regress_B(2)
median(y_0_600)
median_y_bin = median(y_bin)
median_y_bin ./ sum(median_y_bin)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [B, mean_left, mean_right, Evi_tone_value, length_trial, B_prior] = ...
    behave_analysis_block1_200106_lick_each(filename1, time_bin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%[filename1, pathname1]=uigetfile('*.mat','Block_mat');
load(filename1)
%all_trial_time trial_Tup trial_sound trial_lick_L trial_lick_C trial_lick_R 
%Correct_side Chosen_side Outcome EvidenceStrength Trial_time Tone_cloud
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TrialBlock TrialCount BlockTrial BlockProb BlockReward Reward_LCR

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
% use_evidence = temp_evi + 0.1; %0.3 0.6 1
% use_evidence = [0, use_evidence'];
%Evidence = 1/2 + r/2; r = Evidence strength
%tone_evidence = [0 0.2 0.35 0.65 0.8 1];
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

%Correct Choice Evidence
Correct_trial = Correct_side(Choice_trial);
Chosen_trial = Chosen_side(Choice_trial);
Evidence_trial = trial_evidence(Choice_trial);
Choice_block = TrialBlock(Choice_trial);
number_trial = [1:length(Choice_trial)];
tone_cloud = Tone_cloud(Choice_trial,:);
EvidenceStrength = EvidenceStrength(Choice_trial);
TrialBlock = TrialBlock(Choice_trial);
use_trial2 = find(TrialBlock == 2);
use_trial3 = find(TrialBlock == 3);
use_trial23 = union(use_trial2, use_trial3);

TrialBlock01 = TrialBlock;
if BlockProb(2) ~= BlockProb(3) %Block change task
    if BlockProb(2) > BlockProb(3) % Right -> Left
        TrialBlock01(use_trial2) = 1;
        TrialBlock01(use_trial3) = -1;
    else % Left -> Right
        TrialBlock01(use_trial2) = -1;
        TrialBlock01(use_trial3) = 1;
    end
else %Reward change task
    if BlockReward(2,1) < BlockReward(2,2) % Right -> Left
        TrialBlock01(use_trial2) = 1;
        TrialBlock01(use_trial3) = -1;
    else % Left -> Right
        TrialBlock01(use_trial2) = -1;
        TrialBlock01(use_trial3) = 1;
    end
end    

%Pick up difficult stimuli
Evi_diff = find(EvidenceStrength < 0.9);
Evi_diff = intersect(Evi_diff, use_trial23);
[length(use_trial23), length(Evi_diff)]
%Evi_diff = [1:length(EvidenceStrength)];

%Use time bin to make regression value
[size_y,size_x] = size(tone_cloud);
temp_size = ceil(size_x/time_bin);
for i = 1:temp_size,
    temp_min = (i-1)*time_bin+1;
    temp_max = i*time_bin;
    temp_max = min(temp_max, size_x);
    bin_use(i).matrix = [temp_min : temp_max];
end
length_bin = length(bin_use);

%low  1-6
%high 13-18
for i = 1:length(Choice_trial),
    temp_tone = tone_cloud(i,:);
    for j = 1:length_bin,
        temp_temp_tone = temp_tone(bin_use(j).matrix);
        temp0 = find(temp_temp_tone <= 8);
        temp1 = find(temp_temp_tone >= 9);
        binary_tone(i,j) = (length(temp1)-length(temp0)) ./ length(bin_use(j).matrix);
        %binary_tone(i,j) = mean(temp_tone(bin_use(j).matrix));
    end
    
    %Get the data in all sound
    temp0 = find(temp_tone <= 8);
    temp1 = find(temp_tone >= 9);
    binary_all_tone(i,1) = (length(temp1)-length(temp0)) ./ length(temp_tone);
    sum_tone(i) = sum(temp_tone);
end
% size(tone_cloud)
% size(binary_tone)

%Check the tone cloud: Each probability did not have exact number of 
%low and high
% test_evidence = Evidence_trial(Evi_diff);
% test_binary = binary_all_tone(Evi_diff);
test_evidence = Evidence_trial(use_trial23);
test_binary = binary_all_tone(use_trial23);
% test_evidence = Evidence_trial;
% test_binary = binary_all_tone;
for i = 1:length(tone_evidence),
    temp = find(test_evidence == tone_evidence(i));
    sum_tone(temp)';
    
    temp = test_binary(temp);
    Evi_tone_value(i) = mean(temp);
    length_trial(i) = length(temp);
end
length_trial = length_trial ./ sum(length_trial);

%start regression
use_tone   = binary_tone(Evi_diff,:);
use_choice = Chosen_trial(Evi_diff);
use_sound = Correct_trial(Evi_diff);
use_block = TrialBlock01(Evi_diff); %0 or 1

%Pick only difficult trials
[B,dev,stats] = glmfit(use_tone,use_choice,'binomial','link','logit');
[B_prior,dev,stats] = glmfit([use_tone,use_block],use_choice,'binomial','link','logit');

est_choice =  use_tone * B(2:length(B)) + B(1);
est_choice = 1 ./ (1+exp(-est_choice));

choice0 = find(use_choice == 0);
choice1 = find(use_choice == 1);
sound0 = find(use_sound == 0);
sound1 = find(use_sound == 1);

temp_distri = use_tone;
temp_distri = reshape(temp_distri,size(temp_distri,1)*size(temp_distri,2),1);

binary_sum = sum(use_tone,2);
[median(est_choice(choice0)) median(est_choice(choice1))]
%est_choice

%Focus on use_choice
% tone_left  = use_tone(choice0,:);
% tone_right = use_tone(choice1,:);
tone_left  = use_tone(sound0,:);
tone_right = use_tone(sound1,:);

%Because this is for choice, so it shows the bias!!
mean_left  = mean(tone_left);
mean_right = mean(tone_right);

% figure
% hist(temp_distri)
% 
% figure
% subplot(1,2,1)
% boxplot(est_choice(choice0))
% subplot(1,2,2)
% boxplot(est_choice(choice1))
% 
% figure
% subplot(1,2,1)
% boxplot(binary_sum(choice0))
% subplot(1,2,2)
% boxplot(binary_sum(choice1))

B = B';

%stats

return

