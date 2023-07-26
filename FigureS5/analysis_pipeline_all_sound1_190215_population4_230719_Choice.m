%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190215_population4_230719_Choice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Correct_sound_180912','Multiselect','on');
% filename1
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_230719_choice';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

s_all_correct_sound = [];
s_all_correct_choice = [];
c_all_correct_sound = [];
c_all_correct_choice = [];
all_opt_L_mix = [];
all_opt_R_mix = [];
all_opt_LR_mix = [];
all_opt_L_same = [];
all_opt_R_same = [];
all_opt_LR_same = [];
all_opt_L_other = [];
all_opt_R_other = [];
all_opt_LR_other = [];
        
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
    clear s_correct_sound s_correct_choice
    clear c_correct_sound c_correct_choice
    clear opt_L_mix opt_R_mix opt_LR_mix
    clear opt_L_same opt_R_same opt_LR_same
    clear opt_L_other opt_R_other opt_LR_other
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
        
        s_correct_sound(j,:) = stim_sound.distri_sound.correct_sound_moto;
        s_correct_choice(j,:) = stim_sound.distri_sound.correct_choice_moto;
        c_correct_sound(j,:) = stim_sound.distri_choice.correct_sound_moto;
        c_correct_choice(j,:) = stim_sound.distri_choice.correct_choice_moto;
        opt_L_mix(j,:) = stim_sound.distri_sound.opt_L_mix;
        opt_R_mix(j,:) = stim_sound.distri_sound.opt_R_mix;
        opt_LR_mix(j,:) = stim_sound.distri_sound.opt_LR_mix;
        opt_L_same(j,:) = stim_sound.distri_sound.opt_L_same;
        opt_R_same(j,:) = stim_sound.distri_sound.opt_R_same;
        opt_LR_same(j,:) = stim_sound.distri_sound.opt_LR_same;
        opt_L_other(j,:) = stim_sound.distri_sound.opt_L_other;
        opt_R_other(j,:) = stim_sound.distri_sound.opt_R_other;
        opt_LR_other(j,:) = stim_sound.distri_sound.opt_LR_other;
    end
    
    s_all_correct_sound = [s_all_correct_sound; s_correct_sound];
    s_all_correct_choice = [s_all_correct_choice; s_correct_choice];

    c_all_correct_sound = [c_all_correct_sound; c_correct_sound];
    c_all_correct_choice = [c_all_correct_choice; c_correct_choice];

    all_opt_L_mix = [all_opt_L_mix; opt_L_mix];
    all_opt_R_mix = [all_opt_R_mix; opt_R_mix];
    all_opt_LR_mix = [all_opt_LR_mix; opt_LR_mix];
    all_opt_L_same = [all_opt_L_same; opt_L_same];
    all_opt_R_same = [all_opt_R_same; opt_R_same];
    all_opt_LR_same = [all_opt_LR_same; opt_LR_same];
    all_opt_L_other = [all_opt_L_other; opt_L_other];
    all_opt_R_other = [all_opt_R_other; opt_R_other];
    all_opt_LR_other = [all_opt_LR_other; opt_LR_other];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%same other mix
size(s_all_correct_sound) %same other mix
size(s_all_correct_choice) %same other mix
size(mouse_number)

s_all_correct_sound = double(s_all_correct_sound);
c_all_correct_choice = double(c_all_correct_choice);

disp('sound correct')
get_correct_rate_sound_choice(s_all_correct_sound,mouse_number);
disp('choice correct')
get_correct_rate_sound_choice(c_all_correct_choice,mouse_number);

%Compare the correct rate performance between the sound and choice
figure
subplot(1,3,1)
plot(c_all_correct_choice(:,3),s_all_correct_choice(:,3),'.','color',[0 0 0]) %mix
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,2)
plot(c_all_correct_choice(:,1),s_all_correct_choice(:,1),'.','color',[0 0 0]) %same
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,3)
plot(c_all_correct_choice(:,1),s_all_correct_choice(:,3),'.','color',[0 0 0]) %same choice vs mix sound 
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
p_sound_choice(1) = signrank(c_all_correct_choice(:,3),s_all_correct_choice(:,3));
p_sound_choice(2) = signrank(c_all_correct_choice(:,1),s_all_correct_choice(:,1));
p_sound_choice(3) = signrank(c_all_correct_choice(:,1),s_all_correct_choice(:,3));
p_sound_choice
clear p_sound_choice

%Compare the correct rate performance between the sound and choice
figure
subplot(1,3,1)
plot(c_all_correct_choice(:,3),s_all_correct_sound(:,3),'.','color',[0 0 0]) %mix
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,2)
plot(c_all_correct_choice(:,1),s_all_correct_sound(:,1),'.','color',[0 0 0]) %same
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,3)
plot(c_all_correct_choice(:,1),s_all_correct_sound(:,3),'.','color',[0 0 0]) %same choice vs mix sound 
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])

p_sound_choice(1) = signrank(c_all_correct_choice(:,3),s_all_correct_sound(:,3));
p_sound_choice(2) = signrank(c_all_correct_choice(:,1),s_all_correct_sound(:,1));
p_sound_choice(3) = signrank(c_all_correct_choice(:,1),s_all_correct_sound(:,3));
p_sound_choice

lme = fitlme_analysis_20210520_0(c_all_correct_choice(:,3)-s_all_correct_sound(:,3),mouse_number);
lme(2).lme
lme = fitlme_analysis_20210520_0(c_all_correct_choice(:,1)-s_all_correct_sound(:,1),mouse_number);
lme(2).lme
lme = fitlme_analysis_20210520_0(c_all_correct_choice(:,1)-s_all_correct_sound(:,3),mouse_number);
lme(2).lme

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function get_correct_rate_sound_choice(all_correct_sound,mouse_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(1,3,1)
plot(all_correct_sound(:,3),all_correct_sound(:,1),'.','color',[0 0 0]) %mix same
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,2)
plot(all_correct_sound(:,2),all_correct_sound(:,1),'.','color',[0 0 0]) %other same
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,3,3)
plot(all_correct_sound(:,2),all_correct_sound(:,3),'.','color',[0 0 0]) %other mix 
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])

disp('compare median correct rate')
median(all_correct_sound)
sabun(1) = median(all_correct_sound(:,1)-all_correct_sound(:,3)) %same and mix
sabun(2) = median(all_correct_sound(:,1)-all_correct_sound(:,2)) %same and other
sabun

disp('compare same and mix')
all_correct_sound = double(all_correct_sound);
p_sound(1) = signrank(all_correct_sound(:,1),all_correct_sound(:,3));
p_sound(2) = signrank(all_correct_sound(:,1),all_correct_sound(:,2));
p_sound

lme = fitlme_analysis_20210520_0(all_correct_sound(:,1)-all_correct_sound(:,3),mouse_number);
lme(2).lme

% disp('compare same and other')
% lme = fitlme_analysis_20210520_0(all_correct_sound(:,1)-all_correct_sound(:,2),mouse_number);
% lme(2).lme

return

