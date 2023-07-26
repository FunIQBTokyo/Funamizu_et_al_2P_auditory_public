%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190920_correct_ver20230606
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

analysis_folder = 'stimulus'; %stimulus or reward
%analysis_folder = 'reward'; %stimulus or reward

% [filename1, pathname1,findex]=uigetfile('*.mat','Correct_sound_190920','Multiselect','on');
% filename1
pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

% pathname3 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731');
% cd(pathname3)
% filename3 = dir('*.mat');

all_correct_random = [];
all_correct_CV = [];

all_correct_rate1 = [];
all_correct_easy = [];
all_correct_mid = [];
all_correct_dif = [];
all_r_original = [];
all_r_shuffle = [];
all_all_sum_CV = [];

for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
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
%     [all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle] ...
%         = update_correct_rate(use_stim,all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle,max_neuron);

    mouse_session(i) = length(use_stim);
    clear session_correct  all_sum_CV
    for j = 1:length(use_stim)
        temp_stim = use_stim(j).matrix;
        temp_correct1  = temp_stim.correct1; %temp_correct1, temp_easy, temp_mid, temp_dif, temp_correct2
        %size(temp_correct1)
        %Use all neurons
        session_correct(j,1) = temp_correct1(length(temp_correct1),1);
        %Get the max correct rate
        [correct_sum,correct_sum2,reward_rate,opt_mix,~] = get_correct_rate_session_20190920(use_stim(j).matrix);
         all_sum_CV(j,:) =  correct_sum(1);
    end
    all_correct_rate1 = [all_correct_rate1; session_correct]; 
    all_all_sum_CV = [all_all_sum_CV; all_sum_CV];
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

%The logistic regression with all neurons
test = double(all_correct_rate1) - double(all_all_sum_CV);
signrank(test)
lme = fitlme_analysis_20210520_0(test,N_subject);
lme(2).lme


figure
plot(all_correct_rate1, all_all_sum_CV, 'k.')
hold on
plot([0.5 1], [0.5 1], 'k')
set(gca,'xlim',[0.5 1],'ylim',[0.5 1])



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [correct_sum,correct_sum2,reward_rate,opt_mix,r_original,r_shuffle] = get_correct_rate_session_20190920(stim_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%stim_session
%             if isfield(stim_session,'distri_sound'),
                correct_sum  = stim_session.correct_max;
                correct_sum2  = stim_session.correct_max2;
                reward_rate = stim_session.reward_max;
                opt_mix(1,:) = stim_session.opt_L_mix;
                opt_mix(2,:) = stim_session.opt_R_mix;
                opt_mix(3,:) = stim_session.opt_LR_mix;
%             else
%                 correct_sum = nan;
%                 correct_sum2 = nan;
%                 reward_rate = nan;
%                 opt_mix = nan(3,101);
%             end
            
            if isfield(stim_session,'r_original'),
                r_original  = stim_session.r_original;
                r_shuffle  = stim_session.r_shuffle;
                r_original = mean(r_original);
                r_shuffle = mean(r_shuffle);
            else
                r_original = nan;
                r_shuffle = nan;
            end            
return
