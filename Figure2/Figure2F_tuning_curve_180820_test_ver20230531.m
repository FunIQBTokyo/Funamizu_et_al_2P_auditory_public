%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Figure2F_tuning_curve_180820_test_ver20230531
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[norm_all_stim,all_stim,stim_time,stim_frame,stim_mouse_number] = get_average_trace_20230531('stimulus');
[norm_all_rew, all_rew, rew_time, rew_frame,rew_mouse_number] = get_average_trace_20230531('reward');

%Check the stim_mouse and rew_mouse
temp = stim_mouse_number == rew_mouse_number;
temp = min(min(temp))
if temp == 0
    hoge
else
    %mouse, session
    mouse_number = stim_mouse_number;
    clear stim_mouse_number rew_mouse_number
end

%make imagesc plot
stim_bin = [0:9:225];
rew_bin  = [0:9:225];
for i = 1:length(stim_bin)-1,
    temp1 = find(stim_frame >  stim_bin(i));
    temp2 = find(stim_frame <= stim_bin(i+1));
    temp_stim = intersect(temp1,temp2);
    for j = 1:length(rew_bin)-1,
        temp1 = find(rew_frame >  rew_bin(j));
        temp2 = find(rew_frame <= rew_bin(j+1));
        temp_rew = intersect(temp1,temp2);
        temp = intersect(temp_stim,temp_rew);
        time_neuron_matrix(j,i) = length(temp);
        prob_neuron_matrix(j,i) = length(temp) / length(temp_stim);
    end
end
figure
subplot(1,2,1)
imagesc(time_neuron_matrix)
axis xy
subplot(1,2,2)
imagesc(prob_neuron_matrix)
axis xy

for i = 1:length(stim_bin)-1,
    temp1 = find(stim_frame >  stim_bin(i));
    temp2 = find(stim_frame <= stim_bin(i+1));
    temp_stim = intersect(temp1,temp2);
    for j = 1:length(rew_bin)-1,
        temp1 = find(rew_frame >  rew_bin(j));
        temp2 = find(rew_frame <= rew_bin(j+1));
        temp_rew = intersect(temp1,temp2);
        temp = intersect(temp_stim,temp_rew);
        time_neuron_matrix(i,j) = length(temp);
        prob_neuron_matrix(i,j) = length(temp) / length(temp_rew);
    end
end
figure
subplot(1,2,1)
imagesc(time_neuron_matrix)
axis xy
subplot(1,2,2)
imagesc(prob_neuron_matrix)
axis xy

size(stim_frame)
size(rew_frame)
nan_stim = find(isnan(stim_frame) == 1)
nan_rew  = find(isnan(rew_frame) == 1)
nan_stim_rew = union(nan_stim, nan_rew);
[r,p] = corrcoef(stim_frame,rew_frame)
[r,p] = corr(double(stim_frame'),double(rew_frame'),'Type','Spearman')

[r,p] = partialcorr(double(stim_frame'),double(rew_frame'),mouse_number(:,1),'Type','Spearman')
[r,p] = partialcorr(double(stim_frame'),double(rew_frame'),mouse_number(:,2),'Type','Spearman')

figure
plot(stim_frame,rew_frame,'k.')
set(gca,'xlim',[1 225],'xtick',[0:45:225])
set(gca,'ylim',[1 225],'xtick',[0:45:225])

figure
subplot(1,2,1)
imagesc(all_stim(stim_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
subplot(1,2,2)
imagesc(all_rew(stim_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
figure
subplot(1,2,1)
imagesc(all_stim(rew_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
subplot(1,2,2)
imagesc(all_rew(rew_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])


figure
subplot(1,2,1)
imagesc(norm_all_stim(stim_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
subplot(1,2,2)
imagesc(norm_all_rew(stim_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
figure
subplot(1,2,1)
imagesc(norm_all_stim(rew_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])
subplot(1,2,2)
imagesc(norm_all_rew(rew_time,:))
set(gca,'xlim',[1 225],'xtick',[0:45:225])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [norm_all_sound,all_sound,sort_time,max_time,all_mouse_number] = get_average_trace_20230531(analysis_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_20180610/',analysis_folder);
cd(pathname1)
filename1 = dir('*.mat');
%all_sound all_sound_L all_sound_R all_sound_category all_sound_max_time 
%all_sig_sound all_sig_sound_S all_sig_sound_L all_sig_sound_R 
%all_block_L all_block_R 
%all_block_LL all_block_LR all_block_RL all_block_RR 
%all_block_category_L all_block_category_R all_block_max_time
%all_roi_overlap

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

temp = ['cd ',default_folder];
eval(temp); %move directory
if length(analysis_folder) == 8, %Stimulus
    load('tuning_stimulus_task.mat')
elseif length(analysis_folder) == 6, %reward
    load('tuning_reward_task.mat')
else
    hoge
end

clear neuron_number
all_sound = [];

all_mean_trace = [];
all_std_trace = [];
    
all_sig_sound_and = [];
all_sig_sound_or = [];
all_sig_sound_rew = [];
all_sig_sound_stim = [];
all_sig_sound_only = [];

all_mouse_number = [];

count_session = 0;
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);

    roi_overlap = data.all_roi_overlap;
    
    [all_neuron_number(i),~] = size(data.R_activity.all_sound);
    neuron_number(i) = length(roi_overlap);
    
    %Correct for all sessions
    all_sound = [all_sound; data.R_activity.all_sound(roi_overlap,:)];
    
    mean_trace = data.mean_trace;
    std_trace = data.std_trace;
    all_mean_trace = [all_mean_trace; mean_trace(roi_overlap)];
    all_std_trace = [all_std_trace; std_trace(roi_overlap)];

    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2];
    %Pick only sig_sound
    length_session = length(data2.stim_box);
    clear session_number neuron_number_session mouse_number
    for j = 1:length_session,
        temp_stim = data2.stim_box(j).matrix;
        temp_rew  = data2.rew_box(j).matrix;
        temp_sound = [temp_stim(:,2), temp_rew(:,2)];
        all_sig_sound_only = [all_sig_sound_only; temp_sound];
        neuron_number_session(j) = size(temp_stim,1);
    end
    %This is the size of overlapping neurons
    %length(all_sig_sound_only)
    %sum(neuron_number_session)
    neuron_number_session = cumsum(neuron_number_session);
    for j = 1:length_session
        count_session = count_session + 1;
        if j == 1
            session_number(1:neuron_number_session(j)) = count_session;
        else
            session_number(neuron_number_session(j-1)+1:neuron_number_session(j)) = count_session;
        end
    end
    mouse_number(1:length(session_number)) = i;
%     size(session_number)
%     size(mouse_number)

    all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
    
    %Update the index
    mouse_number = [mouse_number', session_number']; %mouse number, session number
    
    all_mouse_number = [all_mouse_number; mouse_number];
end

%Check the number of neurons
if size(all_mouse_number,1) ~= size(all_sig_sound_or,1)
    size(all_mouse_number)
    size(all_sig_sound_or)
    hoge
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

temp_kijyun = all_mean_trace ./ all_std_trace;
[~,time_count] = size(all_sound);
temp_kijyun = repmat(temp_kijyun,1,time_count);
all_sound   = all_sound + temp_kijyun;

%Pick up the sig_sound_neuron with overlap
%Use the or sig neurons
all_sig_sound_or = find(all_sig_sound_or(:,1) == 1);

%Make the 0 to 1 activity
[length_neuron,time_count] = size(all_sound);
length_sig = length(all_sig_sound_or);

for i = 1:length_neuron,
    temp = all_sound(i,:);
    max_time(i) = find(temp == max(temp),1);
end

max_activ = max(all_sound,[],2);
min_activ = min(all_sound,[],2);
max_activ = repmat(max_activ,1,time_count);
min_activ = repmat(min_activ,1,time_count);

norm_all_sound = (all_sound - min_activ) ./ (max_activ-min_activ);

%Limit to all_sig_sound_or
norm_all_sound = norm_all_sound(all_sig_sound_or,:);
all_sound = all_sound(all_sig_sound_or,:);
max_time = max_time(all_sig_sound_or);
all_mouse_number = all_mouse_number(all_sig_sound_or,:);

[~,sort_time] = sort(max_time);

[length_neuron, length_sig]

return
