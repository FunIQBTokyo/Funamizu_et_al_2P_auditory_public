%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
%Sparse Logistic regression to analyze sound frequency
----------------------------------------------------------------------------
%}
function Decoding_analysis_neurometric_190717_compare_230607_reward

analysis_folder = 'stimulus';
%analysis_folder = 'reward';

[all_correct_sound, all_correct_choice] = get_decode_likeli_180717(analysis_folder);

[~,time_count] = size(all_correct_sound);
figure
plot_mean_se_moto(all_correct_sound,[0 0 1],2)
hold on
plot_mean_se_moto(all_correct_choice,[1 0 0],2)
set(gca,'xlim', [1 60])
set(gca,'xtick',[1:3:60])
set(gca,'xticklabel',[1:9:180])
%set(gca,'ylim', [0.45 0.9])
set(gca,'ylim', [0.45 0.85])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_correct_sound, all_correct_choice] = get_decode_likeli_180717(analysis_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

% [filename1, pathname1,findex]=uigetfile('*.mat','Decode_file','Multiselect','on');
% filename1
pathname1 = 'e:\Tone_discri1\all_mouse\single_decode\separate4\population_190717_time_seq\ver20230127_reward_time';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

all_correct_sound_same = [];
all_correct_sound_diff = [];
all_correct_sound_mix = [];
all_correct_choice_same = [];
all_correct_choice_diff = [];
all_correct_choice_mix = [];

all_correct_sound = [];
all_correct_choice = [];
for filename = 1 : length(filename1)
    clear data temp_filename temp_pass fpath
    
    temp_filename = filename1(filename) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    
    length_session = length(stim);
    
    clear correct_sound_same
    clear correct_sound_diff
    clear correct_sound_mix
    clear correct_choice_same
    clear correct_choice_diff
    clear correct_choice_mix
    clear correct_sound 
    clear correct_choice 
    for j = 1:length_session,
        if length(analysis_folder) == 8,
            distri = stim(j).matrix;
        else
            distri = rew(j).matrix;
        end
        length_time = length(distri.distri_sound);
        for k = 1:length_time,
            distri_sound = distri.distri_sound(k).matrix;
            %correct_sound(j,k) = distri_sound.correct_sound;
            correct_sound(j,k) = distri_sound.correct_sound_moto;
            distri_choice = distri.distri_choice(k).matrix;
            %correct_choice(j,k) = distri_choice.correct_choice;
            correct_choice(j,k) = distri_choice.correct_choice_moto;
        end
    end
    all_correct_sound = [all_correct_sound; correct_sound];
    all_correct_choice = [all_correct_choice; correct_choice];
end

return


