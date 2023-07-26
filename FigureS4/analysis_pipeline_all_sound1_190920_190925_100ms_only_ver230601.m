%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190920_190925_100ms_only_ver230601
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Correct_sound_190920','Multiselect','on');
% filename1
pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/100ms_only');
% pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/111ms_only_first');
% pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/200ms_first');
% pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/300ms_first');
% pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/400ms_first');
% pathname1 = strcat('e:\Tone_discri1\all_mouse\single_decode\separate4\population_190920/500ms_first');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/single_decode/separate4/single_190925/100ms_only/');
cd(pathname2)
filename2 = dir('*.mat');

pathname3 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731');
cd(pathname3)
filename3 = dir('*.mat');

all_correct_random = [];
all_correct_CV = [];

max_neuron = 140;

all_opt_LR = [];
all_non_zero_b = [];
all_correct_max = [];
all_correct_max2 = [];
all_reward_max = [];

S_opt_LR = [];
S_non_zero_b = [];
S_correct_max = [];
S_correct_max2 = [];
S_reward_max = [];

L_opt_LR = [];
L_non_zero_b = [];
L_correct_max = [];
L_correct_max2 = [];
L_reward_max = [];

all_max_neuron = [];
all_max_neurometric = [];

all_behave_correct = [];
all_behave_correct2 = [];
all_behave_reward = [];
behave_opt_L_mix = [];
behave_opt_R_mix = [];
behave_opt_LR_mix = [];

mouse_number = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i).name;
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
%     temp_filename = filename1(i) 
%     temp_filename = cell2mat(temp_filename);
%     temp_path = pathname1;
%     fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    if length(analysis_folder) == 8,
        use_stim = stim;
%         use_S = stim_S;
%         use_L = stim_L;
    else
        use_stim = rew;
%         use_S = rew_S;
%         use_L = rew_L;
    end
    
    [all_opt_LR, all_non_zero_b, all_correct_max, all_correct_max2, all_reward_max] = ...
        get_neurometic_shuffle_neurons(use_stim, all_opt_LR, all_non_zero_b, all_correct_max, all_correct_max2, all_reward_max);
%     [S_opt_LR, S_non_zero_b, S_correct_max, S_correct_max2, S_reward_max] = ...
%         get_neurometic_shuffle_neurons(use_S, S_opt_LR, S_non_zero_b, S_correct_max, S_correct_max2, S_reward_max);
%     [L_opt_LR, L_non_zero_b, L_correct_max, L_correct_max2, L_reward_max] = ...
%         get_neurometic_shuffle_neurons(use_L, L_opt_LR, L_non_zero_b, L_correct_max, L_correct_max2, L_reward_max);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Single neurons
    clear opt_L_mix opt_R_mix opt_LR_mix
    temp_filename = filename2(i).name;
    %temp_filename = cell2mat(temp_filename);
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data4 = load(fpath);
    length_session = length(data4.stim);
    
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];
    
    clear temp_sound_decode_neuron temp_sound_neuron
    clear length_sig_sound session_sabun median_session_thre
    clear max_correct_neuron max_neurometric
    for j = 1:length_session,
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         correct_rate = [correct_rate_mix, correct_rate_easy, correct_rate_mid, correct_rate_dif, temp_correct2];
%         correct_rate2 = [mix_L0, mix_L1, mix_R0, mix_R1];
%         reward_rate = all easy mid diff
        %About decoding
        if length(analysis_folder) == 8, %Stimulus
            temp_correct = data4.stim(j).matrix.distri_sound.correct_rate; %same diff mix
            correct_sound2 = data4.stim(j).matrix.distri_sound.correct_rate2;
            reward_sound = data4.stim(j).matrix.distri_sound.reward_sound;
            temp_neurometric = data4.stim(j).matrix.distri_sound.opt_LR_mix;
        elseif length(analysis_folder) == 6, %reward
            temp_correct = data4.rew(j).matrix.distri_sound.correct_rate; %same diff mix
            correct_sound2 = data4.rew(j).matrix.distri_sound.correct_rate2;
            reward_sound = data4.rew(j).matrix.distri_sound.reward_sound;
            temp_neurometric = data4.rew(j).matrix.distri_sound.opt_LR_mix;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Find the best neuron
        temp_correct_mix = temp_correct(:,1);
        max_correct = find(temp_correct_mix == max(temp_correct_mix),1);
        max_correct_neuron(j,1) = temp_correct_mix(max_correct);
        max_neurometric(j,:) = temp_neurometric(max_correct,:);
        
%         all_correct = [all_correct; temp_correct];
%         all_correct_evi = [all_correct_evi; correct_sound2];
%         all_reward_evi = [all_reward_evi; reward_sound];
%         all_neurometric = [all_neurometric; temp_neurometric];
%         %get median based session
    end
    all_max_neuron = [all_max_neuron; max_correct_neuron];
    all_max_neurometric = [all_max_neurometric; max_neurometric];
    
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

all_behave_correct = all_behave_correct(:,3);

%Best single vs Population neuron
figure
subplot(1,2,1)
plot(all_max_neuron(:,1), all_correct_max(:,1),'k.')
hold on
plot([0.5 1],[0.5 1],'k')
%Behavior vs Population neuron
subplot(1,2,2)
plot(all_behave_correct(:,1), all_correct_max(:,1),'k.')
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.4 1])

[all_correct_max(:,1), all_behave_correct(:,1), all_correct_max(:,1) - all_behave_correct(:,1)]

all_max_neuron = double(all_max_neuron);
all_correct_max = double(all_correct_max);
all_behave_correct = double(all_behave_correct);

disp('single neuron or population neuron')
signrank(all_max_neuron(:,1),     all_correct_max(:,1))
lme = fitlme_analysis_20210520_0(all_max_neuron(:,1)-all_correct_max(:,1),mouse_number);
%lme(1).lme
lme(2).lme

disp('behavior or population neuron')
signrank(all_behave_correct(:,1), all_correct_max(:,1))
lme = fitlme_analysis_20210520_0(all_behave_correct(:,1)-all_correct_max(:,1),mouse_number);
%lme(1).lme
lme(2).lme

[median(all_max_neuron(:,1)), median(all_correct_max(:,1)), median(all_behave_correct(:,1))]
% [all_max_neuron(:,1), all_correct_max(:,1)]
% test_single_population = all_correct_max(:,1) - all_max_neuron(:,1);
% test = find(test_single_population < 0)
% test_single_population(test)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_opt_LR, all_non_zero_b, all_correct_max, all_correct_max2, all_reward_max] = ...
    get_neurometic_shuffle_neurons(stim, all_opt_LR, all_non_zero_b, all_correct_max, all_correct_max2, all_reward_max)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    length_session = length(stim); 
    %Sound choice integrate
%     correct_rate(count,:) = [temp_correct1, temp_easy, temp_mid, temp_dif, temp_correct2];
%     correct_rate2(count,:) = [mix_L0, mix_L1, mix_R0, mix_R1];
%     reward_rate(count,:) = get_4_reward_rate(sound_mix(:,temp_mix), Sound, block_L, block_R, Sound_Evi, block_rew);
    r_original = [];
    r_shuffle = [];
    clear opt_LR_mix non_zero_b correct_max correct_max2 reward_max
    for j = 1:length_session,
        opt_LR_mix(j,:)   = stim(j).matrix.opt_LR_mix;
        non_zero_b(j,1)   = stim(j).matrix.non_zero_b;
        correct_max(j,:)  = stim(j).matrix.correct_max;
        correct_max2(j,:) = stim(j).matrix.correct_max2;
        reward_max(j,:)   = stim(j).matrix.reward_max;
    end
    all_opt_LR = [all_opt_LR; opt_LR_mix];
    all_non_zero_b = [all_non_zero_b; non_zero_b];
    
    all_correct_max = [all_correct_max; correct_max];
    all_correct_max2 = [all_correct_max2; correct_max2];
    all_reward_max = [all_reward_max; reward_max];

    return
    
