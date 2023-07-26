%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuning_curve_230531_sound_only_compare_block4_prefer_230725
% function [sig_kruskal, sig_kruskal_stim, sig_kruskal_rew, sig_kruskal_both, sig_sound_timing, check_neuron_number, all_mouse_number] = ...
%                 tuning_curve_230531_sound_only_compare_block4_prefer_230725
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
% analysis_folder = 'stimulus';
% %analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/process_190214');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

temp = ['cd ',default_folder];
eval(temp); %move directory

clear neuron_number

all_sig_sound_or = [];
all_sig_sound_only = [];

stim_p_block = [];
stim_p_kruskal = [];
stim_median_block_L = [];
stim_median_block_R = [];
stim_BF_neuron = [];
rew_p_block = [];
rew_p_kruskal = [];
rew_median_block_L = [];
rew_median_block_R = [];
rew_BF_neuron = [];
     
count_session = 0;
all_mouse_number = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    
    p_block        = data.stim_data.p_block_dif;
    p_kruskal_stim = data.stim_data.p_kruskal;
    median_block_L = data.stim_data.median_block_L;
    median_block_R = data.stim_data.median_block_R;
    BF_neuron =      data.stim_data.BF;
    
%     p_kruskal_stim
%     p_block
%     median_block_L
%     median_block_R
%     BF_neuron
    for j = 1:length(p_block),
        stim_p_block = [stim_p_block; p_block(j).matrix];
        stim_p_kruskal = [stim_p_kruskal; p_kruskal_stim(j).matrix];
        stim_median_block_L = [stim_median_block_L; median_block_L(j).matrix];
        stim_median_block_R = [stim_median_block_R; median_block_R(j).matrix];
        stim_BF_neuron = [stim_BF_neuron; BF_neuron(j).matrix];
    end
    
    p_block        = data.rew_data.p_block_dif;
    p_kruskal_stim = data.rew_data.p_kruskal;
    median_block_L = data.rew_data.median_block_L;
    median_block_R = data.rew_data.median_block_R;
    BF_neuron =      data.rew_data.BF;

    for j = 1:length(p_block),
        rew_p_block = [rew_p_block; p_block(j).matrix];
        rew_p_kruskal = [rew_p_kruskal; p_kruskal_stim(j).matrix];
        rew_median_block_L = [rew_median_block_L; median_block_L(j).matrix];
        rew_median_block_R = [rew_median_block_R; median_block_R(j).matrix];
        rew_BF_neuron = [rew_BF_neuron; BF_neuron(j).matrix];
    end
    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Pick only sig_sound which increased the activity at the sound timing
    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2, sig_sound_ore];
    length_session = length(data2.stim_box);
    for j = 1:length_session,
        temp_stim = data2.stim_box(j).matrix;
        temp_rew  = data2.rew_box(j).matrix;
        temp_sound = [temp_stim(:,2), temp_rew(:,2)];
        all_sig_sound_only = [all_sig_sound_only; temp_sound];
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %sig_roi_overlap_matrix = [sig_roi_overlap; sig_roi_overlap_S; sig_roi_overlap_L; sig_roi_overlap_R];
    all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];

    %Each mouse, get the session number
    clear mouse_number
    [session_number,count_session] = get_session_number(data2.stim_box, count_session);
    mouse_number(1:length(session_number)) = i;
    %Update the index
    mouse_number = [mouse_number', session_number']; %mouse number, session number
    all_mouse_number = [all_mouse_number; mouse_number];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%Pick up the sig_sound_neuron with overlap
%Use the or sig neurons
all_sig_sound_or = find(all_sig_sound_or(:,1) == 1);
%all_sig_sound_and = find(all_sig_sound_and(:,1) == 1);
stim_sig_sound_only = find(all_sig_sound_only(:,1) == 1);
rew_sig_sound_only  = find(all_sig_sound_only(:,2) == 1);
sig_sound_timing = intersect(stim_sig_sound_only, rew_sig_sound_only);

sig_kruskal_stim = find(stim_p_kruskal < 0.01);
sig_kruskal_rew = find(rew_p_kruskal < 0.01);

sig_kruskal_stim = intersect(stim_sig_sound_only,sig_kruskal_stim);
sig_kruskal_rew = intersect(rew_sig_sound_only,sig_kruskal_rew);

sig_kruskal = union(sig_kruskal_stim,sig_kruskal_rew);
sig_kruskal_both = intersect(sig_kruskal_stim,sig_kruskal_rew);

check_neuron_number = [length(stim_p_kruskal), length(rew_p_kruskal)];
[length(sig_kruskal),length(sig_kruskal_stim),length(sig_kruskal_rew),length(sig_kruskal_both),length(sig_sound_timing)]

%Distribution of BF
stim_BF_sig = stim_BF_neuron(sig_kruskal_both,2);
rew_BF_sig  = rew_BF_neuron(sig_kruskal_both,2);

for i = 1:6,
    temp1 = find(stim_BF_sig == i);
    for j = 1:6,
        temp2 = find(rew_BF_sig == j);
        temp = intersect(temp1,temp2);
        BF_map(i,j) = length(temp);
        BF_rew(j) = length(temp2);
        prob_BF_map(i,j) = length(temp) / length(temp2);
        
        if i == j,
            p_rew(i,1) = kai2_test_one_sample([length(temp),length(temp2)],1/6);
            p_rew(i,2) = kai2_test_one_sample([length(temp),length(temp1)],1/6);
            prob_rew(i,1) = length(temp)/length(temp2);
            prob_rew(i,2) = length(temp)/length(temp1);
        end
    end
end
figure
subplot(1,2,1)
imagesc(BF_map)
axis xy
subplot(1,2,2)
imagesc(prob_BF_map)
axis xy

p_rew
min(p_rew(:,1))
max(p_rew(:,1))
p_rew(1,1)
p_rew(2,1)
p_rew(3,1)
p_rew(4,1)
p_rew(5,1)
p_rew(6,1)
BF_rew

prob_rew

[length(stim_BF_sig),length(rew_BF_sig)]
[r,p] = corr(double(stim_BF_sig),double(rew_BF_sig),'type','Spearman')

use_mouse_number = all_mouse_number(sig_kruskal_both,:);
%Partial correlation
[r,p] = partialcorr(double(stim_BF_sig),double(rew_BF_sig),use_mouse_number(:,1),'type','Spearman')
[r,p] = partialcorr(double(stim_BF_sig),double(rew_BF_sig),use_mouse_number(:,2),'type','Spearman')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [session_number,count_session] = get_session_number(stim_box, count_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    length_session = length(stim_box);
    clear session_number neuron_number_session 
    for j = 1:length_session
        temp_stim = stim_box(j).matrix;
        neuron_number_session(j) = size(temp_stim,1);
    end
    neuron_number_session = cumsum(neuron_number_session);
    for j = 1:length_session
        count_session = count_session + 1;
        if j == 1
            session_number(1:neuron_number_session(j)) = count_session;
        else
            session_number(neuron_number_session(j-1)+1:neuron_number_session(j)) = count_session;
        end
    end
    
    return
    

