%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_191219_population_ver20230620
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
use_block_length = 140;
%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_191219_PriorSound_decode';

% [filename1, pathname1,findex]=uigetfile('*.mat','Correct_sound_180912','Multiselect','on');
% filename1
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

all_correct_block = [];
all_correct_sound = [];
all_opt_L_mix = [];
all_opt_R_mix = [];
all_opt_LR_mix = [];
all_ave_likeli = [];

freq_x = [0 0.25 0.45 0.55 0.75 1];
evi_x = [0:0.01:1];

mouse_number = [];
for i = 1 : length(filename1)
    temp_filename = filename1(i) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);

    clear sound_correct choice_correct posterior_correct
    clear correct_block correct_sound
    clear opt_L_mix opt_R_mix opt_LR_mix
    clear opt_L_same opt_R_same opt_LR_same
    clear opt_L_other opt_R_other opt_LR_other
    clear reward_same reward_other reward_mix
    clear ave_likeli
    length_session = length(stim);
    
    %Based on length_neuron, make the mouse session lines
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];

    %Sound choice integrate
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            stim_sound  = stim(j).matrix;
            %correct_sound_stim = [correct_sound_stim; stim_correct_sound(j).matrix];
        elseif length(analysis_folder) == 6, %reward
            stim_sound   = rew(j).matrix;
            %correct_sound_stim = [correct_sound_stim; rew_correct_sound(j).matrix];
        else
            hoge
        end
        
        %Same other mix
        correct_block(j,:) = stim_sound.correct_rate;
        correct_sound(j,:) = stim_sound.correct_rate_sound;
        temp_ave_likeli = stim_sound.mean_likeli_prior;
        length_block = stim_sound.length_block;
        temp_use_block = [length_block(1)-use_block_length+1 : length_block(1)+use_block_length];
        ave_likeli(j,:) = temp_ave_likeli(temp_use_block)';
        
        %all easy mid dif sound
        opt_L_mix(j,:) = stim_sound.opt_L_mix;
        opt_R_mix(j,:) = stim_sound.opt_R_mix;
        opt_LR_mix(j,:) = stim_sound.opt_LR_mix;
    end
    
    all_ave_likeli = [all_ave_likeli; ave_likeli];
    all_correct_block = [all_correct_block; correct_block];
    all_correct_sound = [all_correct_sound; correct_sound];
    
    all_opt_L_mix = [all_opt_L_mix; opt_L_mix];
    all_opt_R_mix = [all_opt_R_mix; opt_R_mix];
    all_opt_LR_mix = [all_opt_LR_mix; opt_LR_mix];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%same other mix
size(all_correct_block) %same other mix
%size(all_correct_choice) %same other mix

all_correct_block = double(all_correct_block);
all_correct_sound = double(all_correct_sound);
[mean(all_correct_block(:,1)), mean(all_correct_sound(:,1))]

%Block correct rate
signrank(all_correct_block(:,1)-0.5)
lme = fitlme_analysis_20210520_0(all_correct_block(:,1)-0.5,mouse_number);
lme(2).lme

figure
boxplot(all_correct_block(:,1))
hold on
temp_x = (rand(length(all_correct_block(:,1)),1) - 0.5)*0.2 + 1;
plot(temp_x, all_correct_block(:,1), 'k.')
set(gca,'ylim',[0.4 1])

%Compare Sound and Block correct rate
all_correct_block = all_correct_block(:,1);
all_correct_sound = all_correct_sound(:,1);
figure
plot(all_correct_block,all_correct_sound,'k.')
hold on
plot([0.4 1],[0.4 1],'k')

signrank(all_correct_block-all_correct_sound)
lme = fitlme_analysis_20210520_0(all_correct_block-all_correct_sound,mouse_number);
lme(2).lme

