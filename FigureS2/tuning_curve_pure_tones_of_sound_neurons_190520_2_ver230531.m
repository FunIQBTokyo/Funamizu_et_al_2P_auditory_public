%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_pure_tones_of_sound_neurons_190520_2_ver230531
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;
%analysis_folder = 'stimulus';
analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

pathname3 = strcat('e:/Tone_discri1/all_mouse/pure_tone_analysis_20180610/thre_005_20180803_20190520/',analysis_folder);
cd(pathname3)
filename3 = dir('*.mat');

cd(default_folder);

clear neuron_number
all_BF_neuron = [];

all_freq_trace2 = [];
all_cloud_trace = [];
all_sig_pure_tone = [];
all_max_freq2 = [];
    
%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, moto_sig_kruskal_both,~,~,all_mouse_number] = ...
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
        stim_correct = find(stim_reward == 1);
        stim_error   = find(stim_reward == 0);
        stim_block_L = find(stim_block == 0);
        stim_block_R  = find(stim_block == 1);
        
        %BF is determined with activity during reward
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear median_block_L_correct median_block_R_correct
        clear p_kruskal_stim BF_neuron p_RE_neuron p_block_neuron p_block_correct
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

        all_BF_neuron = [all_BF_neuron; BF_neuron];
    end    

    temp_filename = filename3(i).name 
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    data3 = load(fpath);
    
    sig_sound = data3.all_sig_sound;
    roi_overlap = data3.all_roi_overlap;
    freq_trace2 = data3.freq2_trace;
    cloud_trace = data3.cloud_trace;
    max_freq2 = data3.all_max_freq2;

    %Use only the roi_overlap neurons
    length_sig_neuron = zeros(length(freq_trace2),1);
    length_sig_neuron(sig_sound) = 1;
    length_sig_neuron = length_sig_neuron(roi_overlap);
    
    freq_trace2 = freq_trace2(roi_overlap,:);
    cloud_trace = cloud_trace(roi_overlap,:);
    max_freq2 = max_freq2(roi_overlap);
    
    all_freq_trace2 = [all_freq_trace2; freq_trace2];
    all_cloud_trace = [all_cloud_trace; cloud_trace];
    all_sig_pure_tone = [all_sig_pure_tone; length_sig_neuron];
    all_max_freq2 = [all_max_freq2; max_freq2];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if length(all_max_freq2)~=length(all_mouse_number)
    [length(all_max_freq2),length(all_mouse_number)]
    hoge
end
%Check about the BF neurons
BF_sig_neuron = all_BF_neuron(sig_kruskal_sound,2);
all_freq_trace2 = all_freq_trace2(sig_kruskal_sound,:);
all_cloud_trace = all_cloud_trace(sig_kruskal_sound,:);
all_sig_pure_tone = all_sig_pure_tone(sig_kruskal_sound);
all_max_freq2 = all_max_freq2(sig_kruskal_sound);
all_mouse_number = all_mouse_number(sig_kruskal_sound,:);

all_sig_number = find(all_sig_pure_tone == 1);
hist_x = [0.5:1:9.5];

%Get the pure tone tuning curve for each BF
BF_cloud_matrix = zeros(6,9);
sig_BF_cloud_matrix = zeros(6,9);
BF_cloud_matrix2 = zeros(6,9);
sig_BF_cloud_matrix2 = zeros(6,9);
for i = 1:6,
    temp = find(BF_sig_neuron == i);
    sig_temp = intersect(all_sig_number, temp);
    
    BF_length(1,i) = length(temp);
    BF_length(2,i) = length(sig_temp);
    
    %pure tone
    BF_tuning(i).matrix = all_freq_trace2(temp,:);
    sig_BF_tuning(i).matrix = all_freq_trace2(sig_temp,:);

    BF_cloud(i).matrix = all_cloud_trace(temp,:);
    sig_BF_cloud(i).matrix = all_cloud_trace(sig_temp,:);
    
    %pure tone BF
    temp_BF = all_max_freq2(temp);
    temp_sig_BF = all_max_freq2(sig_temp);
    hist_BF(i,:) = histcounts(temp_BF,hist_x);
    hist_sig_BF(i,:) = histcounts(temp_sig_BF,hist_x);
    
    BF_cloud_matrix(i,:) = hist_BF(i,:) ./ sum(hist_BF(i,:));
    sig_BF_cloud_matrix(i,:) = hist_sig_BF(i,:) ./ sum(hist_sig_BF(i,:));
    each_BF(i).matrix = temp_BF;
    each_sig_BF(i).matrix = temp_sig_BF;
    
    %Add the mouse number and session number for each BF
    mouse_BF(i).matrix = all_mouse_number(temp,:);
    sig_mouse_BF(i).matrix = all_mouse_number(sig_temp,:);
end

for i = 1:9,
    BF_cloud_matrix2(:,i) = hist_BF(:,i) ./ sum(hist_BF(:,i));
    sig_BF_cloud_matrix2(:,i) = hist_sig_BF(:,i) ./ sum(hist_sig_BF(:,i));
end

BF_length
sum(BF_length,2)

number_color = 9;
use_color = jet(number_color);
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];

freq_x = [0 0.25 0.45 0.55 0.75 1];
figure
for i = 1:6,
    subplot(2,3,i)
%    plot_mean_se_moto(BF_tuning(i).matrix,use_color(i,:),2)
    plot_median_se_moto(BF_tuning(i).matrix,use_color(i,:),2)
end
figure
for i = 1:6,
    subplot(2,3,i)
%    plot_mean_se_moto(sig_BF_tuning(i).matrix,use_color(i,:),2)
    plot_median_se_moto(sig_BF_tuning(i).matrix,use_color(i,:),2)
end
figure
for i = 1:6,
    subplot(2,3,i)
%    plot_mean_se_moto(BF_cloud(i).matrix,use_color(i,:),2)
    plot_median_se_moto_x_axis(BF_cloud(i).matrix,freq_x,use_color(i,:),2)
end
figure
for i = 1:6,
    subplot(2,3,i)
%    plot_mean_se_moto(sig_BF_cloud(i).matrix,use_color(i,:),2)
    plot_median_se_moto_x_axis(sig_BF_cloud(i).matrix,freq_x,use_color(i,:),2)
end

figure
for i = 1:6,
    subplot(2,3,i)
    plot(hist_BF(i,:),'b')
    hold on
    plot(hist_sig_BF(i,:),'r')
    set(gca,'xlim',[0 10])
end
BF_cloud_matrix = BF_cloud_matrix'; %[pure tone x tone cloud]
sig_BF_cloud_matrix = sig_BF_cloud_matrix' %[pure tone x tone cloud];
BF_cloud_matrix = flipud(BF_cloud_matrix);
sig_BF_cloud_matrix = flipud(sig_BF_cloud_matrix);
BF_cloud_matrix2 = flipud(BF_cloud_matrix2); %[tone cloud x pure tone];
sig_BF_cloud_matrix2 = flipud(sig_BF_cloud_matrix2); %[tone cloud x pure tone];

sum(BF_cloud_matrix)
sum(sig_BF_cloud_matrix)
sum(BF_cloud_matrix2)
sum(sig_BF_cloud_matrix2)

figure
subplot(1,2,1)
imagesc(BF_cloud_matrix)
colormap gray
subplot(1,2,2)
imagesc(sig_BF_cloud_matrix)
colormap gray
figure
subplot(1,2,1)
imagesc(BF_cloud_matrix2)
colormap gray
subplot(1,2,2)
imagesc(sig_BF_cloud_matrix2)
colormap gray

%boxplot
temp_fig = [];
temp_sig_fig = [];
mouse_fig = [];
mouse_sig_fig = [];
for i = 1:6
    temp_BF = each_BF(i).matrix;
    temp_sig_BF = each_sig_BF(i).matrix;
    
    temp = ones(length(temp_BF),1) * i;
    temp_sig = ones(length(temp_sig_BF),1) * i;
    
    temp = [temp_BF, temp];
    temp_fig = [temp_fig; temp];
    
    temp_sig = [temp_sig_BF, temp_sig];
    temp_sig_fig = [temp_sig_fig; temp_sig];
    
    
    %About all mouse number
    mouse_fig = [mouse_fig; mouse_BF(i).matrix];
    mouse_sig_fig = [mouse_sig_fig; sig_mouse_BF(i).matrix];
end
figure
subplot(1,2,1)
boxplot_dot(temp_fig(:,1), temp_fig(:,2))
subplot(1,2,2)
boxplot_dot(temp_sig_fig(:,1), temp_sig_fig(:,2))

figure
subplot(1,2,1)
number_neuron_circle(temp_fig(:,1), temp_fig(:,2))
subplot(1,2,2)
number_neuron_circle(temp_sig_fig(:,1), temp_sig_fig(:,2))

%Correlation fro temp_fig temp_sig_fig
size(temp_fig)
size(temp_sig_fig)

disp('all_sound_neurons')
[r,p] = corr(temp_fig(:,1),temp_fig(:,2),'type','Spearman')
[r,p] = partialcorr(temp_fig(:,1),temp_fig(:,2),mouse_fig(:,1),'type','Spearman')
[r,p] = partialcorr(temp_fig(:,1),temp_fig(:,2),mouse_fig(:,2),'type','Spearman')

disp('sound_neurons at task')
[r_sig,p_sig] = corr(temp_sig_fig(:,1),temp_sig_fig(:,2),'type','Spearman')
[r_sig,p_sig] = partialcorr(temp_sig_fig(:,1),temp_sig_fig(:,2),mouse_sig_fig(:,1),'type','Spearman')
[r_sig,p_sig] = partialcorr(temp_sig_fig(:,1),temp_sig_fig(:,2),mouse_sig_fig(:,2),'type','Spearman')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function number_neuron_circle(data, number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

haba = 0.1;
t = linspace(0,2*pi,100);
length_neuron = length(data);

max_number = max(number);
for i = 1:max_number,
    temp = find(number == i);
    temp_data = data(temp);
    %temp_x = i + (rand(length(temp),1) - 0.5) .* haba;
    %plot(temp_x, temp_data, 'k.')
    %hold on
    
    for j = 1:9,
        temp = find(temp_data == j);
        temp_number(j) = length(temp);
    end
    %temp_number = temp_number ./ sum(temp_number);
    %temp_number = temp_number .* 0.005; % 100 neurons -> diameter 1 
    temp_number = temp_number .* 0.01; % 100 neurons -> diameter 2 
    %temp_number = 5 * temp_number ./ length_neuron;
    
    %Draw circle in each point
    for j = 1:9,
        plot(i + temp_number(j) .* sin(t), j + temp_number(j) .* cos(t), 'k')
        hold on
    end
end
%boxplot(data, number)

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function boxplot_dot(data, number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

haba = 0.1;
t = linspace(0,2*pi,100);

max_number = max(number);
for i = 1:max_number,
    temp = find(number == i);
    temp_data = data(temp);
    %temp_x = i + (rand(length(temp),1) - 0.5) .* haba;
    %plot(temp_x, temp_data, 'k.')
    %hold on
    
    for j = 1:9,
        temp = find(temp_data == j);
        temp_number(j) = length(temp);
    end
    temp_number = temp_number ./ sum(temp_number);
    temp_number = temp_number .* 0.5;
    
    %Draw circle in each point
    for j = 1:9,
        plot(i + temp_number(j) .* sin(t), j + temp_number(j) .* cos(t), 'k')
        hold on
    end
end
boxplot(data, number)

return

