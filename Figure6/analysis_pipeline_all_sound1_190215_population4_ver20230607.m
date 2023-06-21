%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190215_population4_ver20230607
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Correct_sound_180912','Multiselect','on');
% filename1
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_200401';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

all_correct_sound = [];
all_correct_choice = [];
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
    clear correct_sound correct_choice
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
        
        %correct_sound(j,:) = stim_sound.distri_sound.correct_sound;
        correct_sound(j,:) = stim_sound.distri_sound.correct_sound_moto;
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
    
    all_correct_sound = [all_correct_sound; correct_sound];
    %all_correct_choice = [all_correct_choice; correct_choice];
    
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
size(all_correct_sound) %same other mix
%size(all_correct_choice) %same other mix

figure
subplot(1,2,1)
plot(all_correct_sound(:,3),all_correct_sound(:,1),'.','color',[0 0 0])
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])
subplot(1,2,2)
plot(all_correct_sound(:,2),all_correct_sound(:,1),'.','color',[0 0 0])
hold on
plot([0.5 1],[0.5 1],'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])

median(all_correct_sound)
median(all_correct_sound(:,1)-all_correct_sound(:,3))
median(all_correct_sound(:,1)-all_correct_sound(:,2))

all_correct_sound = double(all_correct_sound);
signrank(all_correct_sound(:,1),all_correct_sound(:,3))
lme = fitlme_analysis_20210520_0(all_correct_sound(:,1)-all_correct_sound(:,3),mouse_number);
lme(2).lme

signrank(all_correct_sound(:,1),all_correct_sound(:,2))
lme = fitlme_analysis_20210520_0(all_correct_sound(:,1)-all_correct_sound(:,2),mouse_number);
lme(2).lme
%hoge

figure
subplot(1,2,1)
temp = all_correct_sound(:,1)-all_correct_sound(:,3);
h = boxplot(temp); %same - mix
set(h(7,:),'Visible','off')
hold on
temp_x = 0.1 * (rand(length(temp),1) - 0.5);
temp_x = temp_x + 1;
plot(temp_x,temp,'.','color',[0.5 0.5 0.5])

subplot(1,2,2)
temp = all_correct_sound(:,1)-all_correct_sound(:,2);
h = boxplot(temp); %same - mix
set(h(7,:),'Visible','off')
hold on
temp_x = 0.1 * (rand(length(temp),1) - 0.5);
temp_x = temp_x + 1;
plot(temp_x,temp,'.','color',[0.5 0.5 0.5])
set(gca,'ylim',[-0.05 0.11])

evi_x = [0:0.01:1];
figure
subplot(1,3,1)
plot_mean_se_moto_x_axis(all_opt_L_same,evi_x,[0 0 1],2)
hold on
plot_mean_se_moto_x_axis(all_opt_R_same,evi_x,[1 0 0],2)
set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])
subplot(1,3,2)
plot_mean_se_moto_x_axis(all_opt_L_mix,evi_x,[0 0 1],2)
hold on
plot_mean_se_moto_x_axis(all_opt_R_mix,evi_x,[1 0 0],2)
set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])
subplot(1,3,3)
plot_mean_se_moto_x_axis(all_opt_L_other,evi_x,[0 0 1],2)
hold on
plot_mean_se_moto_x_axis(all_opt_R_other,evi_x,[1 0 0],2)
set(gca,'xlim',[-0.1 1.1],'ylim',[0 1])

mean_L_same = mean(all_opt_L_same,2);
mean_R_same = mean(all_opt_R_same,2);
mean_L_mix = mean(all_opt_L_mix,2);
mean_R_mix = mean(all_opt_R_mix,2);
mean_L_other = mean(all_opt_L_other,2);
mean_R_other = mean(all_opt_R_other,2);

sabun_LR_same = mean_R_same - mean_L_same;
sabun_LR_mix = mean_R_mix - mean_L_mix;
sabun_LR_other = mean_L_other - mean_R_other;

hist_x = [-inf,-0.5:0.1:0.5,inf];
hist_sabun_same = histcounts(sabun_LR_same,hist_x);
hist_sabun_mix  = histcounts(sabun_LR_mix, hist_x);
hist_sabun_other  = histcounts(sabun_LR_other, hist_x);
figure
subplot(1,3,1)
bar(hist_sabun_same)
set(gca,'xlim',[0 length(hist_x)])
subplot(1,3,2)
bar(hist_sabun_mix)
set(gca,'xlim',[0 length(hist_x)])
subplot(1,3,3)
bar(hist_sabun_other)
set(gca,'xlim',[0 length(hist_x)])

%signrank(sabun_LR_same)
signrank(sabun_LR_mix)
% % signrank(hist_sabun_other)
%signrank(sabun_LR_other)
lme = fitlme_analysis_20210520_0(sabun_LR_mix,mouse_number);
lme(2).lme


%[sabun_LR_same, sabun_LR_mix]


