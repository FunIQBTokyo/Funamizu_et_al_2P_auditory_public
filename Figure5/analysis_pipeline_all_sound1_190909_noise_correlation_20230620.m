%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190909_noise_correlation_20230607
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Decode_20190909','Multiselect','on');
% filename1
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190909';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

max_neuron = 140;
all_all_sum_CV = [];
all_all_shuffle_CV = [];
all_all_sum_train = [];
all_all_shuffle_train = [];
all_all_r = [];
all_opt_LR = [];
all_opt_S = [];
all_opt_L = [];
mouse_number = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    
    clear correct_same correct_other correct_mix
    clear y_L y_R x05 
    clear correct_random correct_block
    clear correct_same correct_other correct_mix
    clear correct_b_same correct_b_mix correct_b_other
    clear all_sum_CV all_shuffle_CV all_sum_train all_shuffle_train
    clear all_r
    clear opt_LR opt_S opt_L
    length_session = length(stim);
    %Based on length_neuron, make the mouse session lines
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];
    
    %Sound choice integrate
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            [correct_sum,opt_LR(j,:),~] = get_correct_rate_session(stim(j).matrix);
            [correct_S,opt_S(j,:),r_ori_S,r_S] = get_correct_rate_session(stim_S(j).matrix);
            [correct_L,opt_L(j,:),r_ori_L,r_L] = get_correct_rate_session(stim_L(j).matrix);

        elseif length(analysis_folder) == 6, %reward
            [correct_sum,opt_LR(j,:),~] = get_correct_rate_session(rew(j).matrix);
            [correct_S,opt_S(j,:),r_ori_S,r_S] = get_correct_rate_session(rew_S(j).matrix);
            [correct_L,opt_L(j,:),r_ori_L,r_L] = get_correct_rate_session(rew_L(j).matrix);
        else
            hoge
        end
        correct_sum = correct_sum(1);
        correct_S = correct_S(1);
        correct_L = correct_L(1);
        
        all_sum_CV(j,:) =  [correct_sum, correct_S, correct_L];
        all_r(j,:) = [r_ori_S, r_ori_L, r_S, r_L];
    end
    all_all_sum_CV = [all_all_sum_CV; all_sum_CV];
    all_all_r = [all_all_r; all_r];
    all_opt_LR = [all_opt_LR; opt_LR];
    all_opt_S = [all_opt_S; opt_S];
    all_opt_L = [all_opt_L; opt_L];
%     all_all_shuffle_CV = [all_all_shuffle_CV; all_shuffle_CV];
%     all_all_sum_train = [all_all_sum_train; all_sum_train];
%     all_all_shuffle_train = [all_all_shuffle_train; all_shuffle_train];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

all_all_sum_CV
mean_CV = mean(all_all_sum_CV,2);
temp_nan = find(isnan(mean_CV) == 0);

length(temp_nan)

all_sum_CV = all_all_sum_CV(temp_nan,:);
mouse_number = mouse_number(temp_nan);
median(all_sum_CV)

all_all_r = all_all_r(temp_nan,:);
median(all_all_r)

figure
plot(all_sum_CV', 'color', [0.5 0.5 0.5])
hold on
boxplot(all_sum_CV)
% for i = 1:3,
%     hold on
%     temp_x = 0.2 * (rand(length(temp_nan),1) - 0.5) + i;
%     plot(temp_x, all_sum_CV(:,i), 'k.')
% end
all_sum_CV = double(all_sum_CV);
signrank(all_sum_CV(:,1), all_sum_CV(:,2))
signrank(all_sum_CV(:,1), all_sum_CV(:,3))

lme = fitlme_analysis_20210520_0(all_sum_CV(:,1)-all_sum_CV(:,2),mouse_number);
lme(2).lme
lme = fitlme_analysis_20210520_0(all_sum_CV(:,1)-all_sum_CV(:,3),mouse_number);
lme(2).lme

figure
subplot(1,2,1)
plot(all_sum_CV(:,1), all_sum_CV(:,2),'r.')
hold on
plot([0.5 1],[0.5 1],'k')
subplot(1,2,2)
plot(all_sum_CV(:,1), all_sum_CV(:,3),'b.')
hold on
plot([0.5 1],[0.5 1],'k')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_sum,opt_LR,r_original,r_shuffle] = get_correct_rate_session(stim_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stim_session
            if isfield(stim_session,'distri_sound'),
                correct_sum  = stim_session.distri_sound.correct_sound;
                opt_LR  = stim_session.distri_sound.opt_LR_mix;
            else
                correct_sum = nan;
                opt_LR  = nan(1,101);
            end
            
            if isfield(stim_session,'r_original'),
                r_original  = stim_session.r_original;
                r_shuffle  = stim_session.r_shuffle;
            else
                r_original = nan;
                r_shuffle = nan;
            end            
return


