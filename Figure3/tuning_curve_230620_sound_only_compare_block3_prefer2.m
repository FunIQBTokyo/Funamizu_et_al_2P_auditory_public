%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_230620_sound_only_compare_block3_prefer2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

analysis_folder = 'stimulus';
[stim_length,stim_p_sound,stim_p_sig] = get_prior_modulation(analysis_folder);

analysis_folder = 'reward';
[rew_length,rew_p_sound,rew_p_sig] = get_prior_modulation(analysis_folder)

stim_length
rew_length

both_p_sound = intersect(stim_p_sound, rew_p_sound);
union_p_sound = union(stim_p_sound, rew_p_sound);
stim_p_only = setdiff(stim_p_sound, both_p_sound);
rew_p_only  = setdiff(rew_p_sound, both_p_sound);
non_p_sound = stim_length(1,1) - length(union_p_sound);

pie_sound = [non_p_sound, length(both_p_sound), length(stim_p_only), length(rew_p_only)]

both_p_sig = intersect(stim_p_sig, rew_p_sig);
union_p_sig = union(stim_p_sig, rew_p_sig);
stim_p_only = setdiff(stim_p_sig, both_p_sig);
rew_p_only  = setdiff(rew_p_sig, both_p_sig);
non_p_sig = stim_length(1,2) - length(union_p_sig);

pie_sig = [non_p_sig, length(both_p_sig), length(stim_p_only), length(rew_p_only)]

figure
subplot(1,2,1)
pie(pie_sound)
subplot(1,2,2)
pie(pie_sig)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [length_neuron,p_sound_number,p_sig_number] = get_prior_modulation(analysis_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
currentFolder = pwd;
%analysis_folder = 'stimulus';
%analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

%[filename1, pathname1,findex]=uigetfile('*.mat','Sound_file','Multiselect','on');
pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

cd(default_folder); %move directory

clear neuron_number
    
all_sig_sound_or = [];
all_sig_sound_only = [];

all_median_all = [];
all_median_correct = [];
all_median_error = [];
all_BF_neuron = [];
all_p_RE = [];
all_p_block = [];
all_p_block_correct = [];

all_median_block_L = [];
all_median_block_R = [];
all_median_block_L_correct = [];
all_median_block_R_correct = [];

all_p_kruskal_stim = [];
all_p_prior = [];

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, moto_sig_kruskal_both] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all

for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    %'stim_sound','stim_norm','stim_task','stim_order_block','stim_mean','stim_std',
    %'rew_sound','rew_norm','rew_task','rew_order_block','rew_mean','rew_std'
    if length(analysis_folder) == 8, %Stimulus
        %'stim_sound','stim_baseline','stim_task','stim_order_block','rew_sound','rew_baseline','rew_task','rew_order_block'
        stim_sound = data.stim_sound; %df/f
        %stim_baseline = data.stim_baseline;
        stim_task = data.stim_task; %[Sound, reward, choice, Evidence, Block];
    elseif length(analysis_folder) == 6, %reward
        stim_sound = data.rew_sound;
        %rew_baseline = data.rew_baseline;
        stim_task = data.rew_task; %[Sound, reward, choice, Evidence, Block];
    else
        hoge
    end

    length_session = length(stim_sound);
    for j = 1:length_session,
        temp_stim = stim_sound(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);

        %Each tone cloud
        stim_evi   = temp_stim_task(:,4);
        stim_reward = temp_stim_task(:,2);
        stim_block = temp_stim_task(:,5);
        stim_category = temp_stim_task(:,1);
        stim_correct = find(stim_reward == 1);
        stim_error   = find(stim_reward == 0);
        stim_block_L = find(stim_block == 0);
        stim_block_R  = find(stim_block == 1);
        stim_category_L = find(stim_category == 0);
        stim_category_R = find(stim_category == 1);
        %BF is determined with activity during reward
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear median_block_L_correct median_block_R_correct
        clear p_kruskal_stim BF_neuron p_RE_neuron p_block_neuron p_block_correct
        clear p_prior
        %Make new definition about block modulation
        %Within each tone cloud category, if the activity is different
        %block modulation
        block_category1L = intersect(stim_block_L, stim_category_L);
        block_category2L = intersect(stim_block_R, stim_category_L);
        block_category1R = intersect(stim_block_L, stim_category_R);
        block_category2R = intersect(stim_block_R, stim_category_R);
        for l = 1:size_neuron,
            p_prior(l,1) = ranksum(temp_stim(block_category1L,l),temp_stim(block_category2L,l));
            p_prior(l,2) = ranksum(temp_stim(block_category1R,l),temp_stim(block_category2R,l));
        end
        for k = 1:6,
            temp_evi = find(stim_evi == k);
            temp_correct = intersect(temp_evi, stim_correct);
            temp_error = intersect(temp_evi, stim_error);
            temp_block_L = intersect(temp_evi, stim_block_L);
            temp_block_R = intersect(temp_evi, stim_block_R);
            temp_block_L_correct = intersect(temp_correct, temp_block_L);
            temp_block_R_correct = intersect(temp_correct, temp_block_R);
            
            median_all(:,k) = median(temp_stim(temp_evi,:),1)';
            median_correct(:,k) = median(temp_stim(temp_correct,:),1)';
            if length(temp_error) ~= 0,
                median_error(:,k) =  median(temp_stim(temp_error,:),1)';
                for l = 1:size_neuron,
                    p_RE_neuron(l,k) = ranksum(temp_stim(temp_correct,l),temp_stim(temp_error,l));
                end
            else
                median_error(:,k) =  nan(size_neuron,1);
                p_RE_neuron(:,k) = nan(size_neuron,1);
            end
            
            for l = 1:size_neuron,
                p_block_neuron(l,k) = ranksum(temp_stim(temp_block_L,l),temp_stim(temp_block_R,l));
                if length(temp_block_L_correct) ~= 0 & length(temp_block_R_correct) ~= 0
                    p_block_correct(l,k) = ranksum(temp_stim(temp_block_L_correct,l),temp_stim(temp_block_R_correct,l));
                else
                    p_block_correct(l,k) = nan;
                end
            end
            
            %get the significant test value
            median_block_L(:,k) = median(temp_stim(temp_block_L,:),1)';
            median_block_R(:,k) = median(temp_stim(temp_block_R,:),1)';
            median_block_L_correct(:,k) = median(temp_stim(temp_block_L_correct,:),1)';
            median_block_R_correct(:,k) = median(temp_stim(temp_block_R_correct,:),1)';
        end
        %Detect BF
        for l = 1:size_neuron,
            p_kruskal_stim(l,1) = kruskalwallis(temp_stim(:,l),stim_evi,'off');

            BF_neuron(l,1) = find(median_all(l,:) == max(median_all(l,:)),1);
            BF_neuron(l,2) = find(median_correct(l,:) == max(median_correct(l,:)),1);
            BF_neuron(l,3) = find(median_error(l,:) == max(median_error(l,:)),1);
        end

        all_p_RE = [all_p_RE; p_RE_neuron];
        all_p_prior = [all_p_prior; p_prior];
        all_p_block = [all_p_block; p_block_neuron];
        all_p_block_correct = [all_p_block_correct; p_block_correct];
        all_p_kruskal_stim = [all_p_kruskal_stim; p_kruskal_stim];
        all_median_all = [all_median_all; median_all];
        all_median_correct = [all_median_correct; median_correct];
        all_median_error = [all_median_error; median_error];
        all_median_block_L = [all_median_block_L; median_block_L];
        all_median_block_R = [all_median_block_R; median_block_R];
        all_median_block_L_correct = [all_median_block_L_correct; median_block_L_correct];
        all_median_block_R_correct = [all_median_block_R_correct; median_block_R_correct];
        all_BF_neuron = [all_BF_neuron; BF_neuron];
    end    

    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2, sig_sound_ore];
    %Pick only sig_sound
    length_session = length(data2.stim_box);
    for j = 1:length_session,
        temp_stim = data2.stim_box(j).matrix;
        temp_rew  = data2.rew_box(j).matrix;
        temp_sound = [temp_stim(:,2), temp_rew(:,2)];
        all_sig_sound_only = [all_sig_sound_only; temp_sound];
    end
    
    %sig_roi_overlap_matrix = [sig_roi_overlap; sig_roi_overlap_S; sig_roi_overlap_L; sig_roi_overlap_R];
     all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

thre = 0.01; 
all_p_prior = min(all_p_prior,[],2); %Take the smaller value

all_BF_neuron = all_BF_neuron(:,2);
for i = 1:length(all_BF_neuron)
    all_p_BF(i,:) = all_p_block(i,all_BF_neuron(i));
end
thre = 0.05; 
all_p_prior = all_p_BF;

%Pick up the sig_sound_neuron with overlap
%Use the or sig neurons
all_sig_sound_or = find(all_sig_sound_or(:,1) == 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Get how much neurons had prior change
p_sig_neuron   = all_p_prior(all_sig_sound_or);
p_sound_neuron = all_p_prior(sig_kruskal_sound);

p_sig_number   = find(p_sig_neuron < thre);
p_sound_number = find(p_sound_neuron < thre);

length_neuron(1,:) = [length(sig_kruskal_sound), length(all_sig_sound_or)];
length_neuron(2,:) = [length(p_sound_number), length(p_sig_number)];

length_neuron

