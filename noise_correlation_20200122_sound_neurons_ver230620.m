%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function noise_correlation_20200122_sound_neurons_ver230620
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

pathname3 = strcat('e:/Tone_discri1/all_mouse/noise_correlation/mean_xy');
cd(pathname3)
filename3 = dir('*.mat');

temp = ['cd ',default_folder];
eval(temp); %move directory

clear neuron_number

all_sound_neuron = [];
all_size_neuron = [];
%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, moto_sig_kruskal_both] = ...
                tuning_curve_190429_sound_only_compare_block4_prefer;
close all

%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;

count = 0;
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename3(i).name 
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    data3 = load(fpath);

    %'stim_sound','stim_norm','stim_task','stim_order_block','stim_mean','stim_std',
    %'rew_sound','rew_norm','rew_task','rew_order_block','rew_mean','rew_std'
    if length(analysis_folder) == 8, %Stimulus
        stim_xy = data3.stim;
    elseif length(analysis_folder) == 6, %reward
        stim_xy = data3.rew;
    else
        hoge
    end
    
    clear number_thre length_neuron
    clear corr_same p_corr_same b_same p_b_same
    clear corr_other p_corr_other b_other p_b_other
    length_session = length(stim_xy);
    for j = 1:length_session,
        count = count + 1;
        mean_xy = stim_xy(j).matrix.mean_xy;
        size_neuron = length(mean_xy);
        %Get the sound responsive neurons for testing
        all_size_neuron(count) = size_neuron;
        sum_neuron = sum(all_size_neuron);
        haba_neuron = [sum_neuron - size_neuron + 1 : sum_neuron];
        [use_neuron,neuron_number] = intersect(haba_neuron,sig_kruskal_sound);
        all_sound_neuron(count) = length(neuron_number);
        all_neuron_number(count).matrix = neuron_number;
        mouse_number_session(count,:) = [i,j];
    end    
end

all_sound_neuron(1:10)'
all_sound_neuron(11:31)'
all_sound_neuron(32:45)'
all_sound_neuron(46:57)'
all_sound_neuron(58:74)'
all_sound_neuron(75:83)'
[sum(all_sound_neuron), sum(all_size_neuron)]
currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%Pick up the number of neurons
% mouse_number = 5;
% session_number = 3;
mouse_number = 2;
session_number = 3;

temp_mouse   = find(mouse_number_session(:,1) == mouse_number);
temp_session = find(mouse_number_session(:,2) == session_number);
use_count = intersect(temp_mouse, temp_session);
if length(use_count) ~= 1,
    hoge
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get the data from one session
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Activity is only during block23!!

[mean_sound,Sound_Evi,choice,mean_xy,BF_neuron,X_plot_all,Y_plot_all,X_plot_moto,Y_plot_moto] = ...
    get_data_from_one_session(analysis_folder,filename1, pathname1, filename3, pathname3, mouse_number, session_number, all_neuron_number(use_count).matrix);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Start analysis
mean_sound = mean_sound';
[trace_number,~] = size(mean_sound);
[all_size_neuron(use_count), trace_number]

%normalize activity
for i = 1:trace_number,
    norm_sound(i,:) = (mean_sound(i,:) - mean(mean_sound(i,:))) ./ std(mean_sound(i,:));
end
mean_sound = norm_sound;
clear norm_sound

number_color = 9;
use_color = jet(number_color);
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Make distance for clustering
Y = pdist(mean_xy,'euclidean');
%squareform(Y)
size(squareform(Y))
Z = linkage(Y);
figure
subplot(2,1,1)
[T1,T2,outperform] = dendrogram(Z,0);

plot_BF_neuron = BF_neuron(outperform)';
subplot(2,1,2)
%imagesc(BF_neuron(outperform)')
for i = 1:length(plot_BF_neuron),
    fill([i i+1 i+1 i],[0 0 1 1],use_color(plot_BF_neuron(i),:),'Edgecolor','none')
    hold on
end
set(gca,'xlim',[1 length(plot_BF_neuron)+1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:6,
    evi_trial(i).matrix = find(Sound_Evi == i);
end
residual_new = zeros(size(mean_sound));

%parfor i = 1:trace_number,
clear mean_evi std_evi
for i = 1:trace_number,
%for i = 1:3,
    %[i, trace_number]
    temp_sound  = mean_sound(i,:);
    
    temp_sound_new = temp_sound;
    for j = 1:6,
        temp = temp_sound(evi_trial(j).matrix);
        mean_evi(i,j) = mean(temp);
        std_evi(i,j) = std(temp);
        temp_sound_new(evi_trial(j).matrix) = (temp_sound(evi_trial(j).matrix) - mean_evi(i,j)) ./ std_evi(i,j);
    end
    residual_new(i,:) = temp_sound_new;
end
%warning('on',id);

%[rho_residual, BF_residual, outperform, rho_residual_raw, distance_raw] = make_clustering(norm_residual', BF_neuron, mean_xy, 0);

[rho_residual, BF_residual, outperform, rho_residual_raw, distance_raw, BF_pair] = make_clustering(residual_new, BF_neuron, mean_xy, 0);
[rho_activity, ~, outperform_activity, rho_activity_raw,~] = make_clustering(mean_evi, BF_neuron, mean_xy, 0);

%Plot the relationship between signal correlation and noice correlation,
%and maybe distance.
[size_y,size_x] = size(rho_activity);
plot_rho_activity = [];
plot_rho_residual = [];
plot_distance = [];
plot_BF_pair = [];
for i = 1:size_y,
    temp_activity = rho_activity_raw(i,[i+1:size_x]);
    temp_residual = rho_residual_raw(i,[i+1:size_x]);
    temp_distance = distance_raw(i,[i+1:size_x]);
    temp_BF = BF_pair(i,[i+1:size_x]);
    
    plot_rho_activity = [plot_rho_activity, temp_activity];
    plot_rho_residual = [plot_rho_residual, temp_residual];
    plot_distance = [plot_distance, temp_distance];
    plot_BF_pair = [plot_BF_pair, temp_BF];
end
[~,temp_I] = sort(plot_distance);
use_color = jet(length(temp_I));

%Get moving average for the raster plots
x_rho = [-1:0.1:0.8];
x_step_rho = 0.2;
%x_dist = [0:50:500];
x_dist = [0:50:350];
x_step_dist = 100;

non_BF_pair = find(plot_BF_pair == 0);
same_BF_pair = find(plot_BF_pair == 1);
other_BF_pair = find(plot_BF_pair == -1);

[y_mean_rho, x_plot_rho] = get_mean_ave_plot(x_rho, x_step_rho,plot_rho_activity,plot_rho_residual);
[y_mean_dist, x_plot_dist] = get_mean_ave_plot(x_dist, x_step_dist,plot_distance,plot_rho_residual);

[y_same_dist, ~] = get_mean_ave_plot(x_dist, x_step_dist,plot_distance(same_BF_pair),plot_rho_residual(same_BF_pair));
[y_other_dist, ~] = get_mean_ave_plot(x_dist, x_step_dist,plot_distance(other_BF_pair),plot_rho_residual(other_BF_pair));

figure
subplot(1,2,1)
%plot(plot_rho_activity, plot_rho_residual,'.','color',[0.7 0.7 0.7]);
plot(plot_rho_activity(non_BF_pair), plot_rho_residual(non_BF_pair),'.','color',[0.6 0.6 0.6]);
hold on
%plot(plot_rho_activity(same_BF_pair), plot_rho_residual(same_BF_pair),'.','color',[232 178 183]./255);
plot(plot_rho_activity(same_BF_pair), plot_rho_residual(same_BF_pair),'.','color',[221 106 156]./255);
hold on
%plot(plot_rho_activity(other_BF_pair), plot_rho_residual(other_BF_pair),'.','color',[142 209 224]./255);
plot(plot_rho_activity(other_BF_pair), plot_rho_residual(other_BF_pair),'.','color',[75 197 221]./255);
hold on
plot(x_plot_rho, y_mean_rho,'k');
set(gca,'xlim',[-1.1 1.1],'ylim',[-0.4 0.8])
%set(gca,'ylim',[-0.4 0.8])

subplot(1,2,2)
%plot(plot_distance, plot_rho_residual,'.','color',[0.7 0.7 0.7]);
plot(plot_distance(non_BF_pair), plot_rho_residual(non_BF_pair),'.','color',[0.6 0.6 0.6]);
hold on
%plot(plot_distance(same_BF_pair), plot_rho_residual(same_BF_pair),'.','color',[232 178 183]./255);
plot(plot_distance(same_BF_pair), plot_rho_residual(same_BF_pair),'.','color',[221 106 156]./255);
hold on
%plot(plot_distance(other_BF_pair), plot_rho_residual(other_BF_pair),'.','color',[142 209 224]./255);
plot(plot_distance(other_BF_pair), plot_rho_residual(other_BF_pair),'.','color',[75 197 221]./255);
hold on
plot(x_plot_dist, y_mean_dist,'k');
hold on
plot(x_plot_dist, y_same_dist,'r');
hold on
plot(x_plot_dist, y_other_dist,'b');
set(gca,'xlim',[0 600],'ylim',[-0.4 0.8])

[length(plot_rho_activity),length(plot_rho_residual),length(plot_distance)]



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y_plot, x_bin] = get_mean_ave_plot(x_rho, x_step_rho,plot_rho_activity,plot_rho_residual)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_bin = x_rho + x_step_rho/2;
for i = 1:length(x_rho),
    temp1 = x_rho(i);
    temp2 = x_rho(i) + x_step_rho;
    if i == 1;
        temp = find(plot_rho_activity <= temp2);
    elseif i == length(x_rho),
        temp = find(plot_rho_activity > temp1);
    else 
        temp = find(plot_rho_activity > temp1 &  plot_rho_activity <= temp2);
    end
    temp_y = plot_rho_residual(temp);
    y_plot(i) = mean(temp_y);
    %y_plot(i) = median(temp_y);
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [rho_original, plot_BF_neuron, outperform, rho_raw, distance_raw, BF_pair] = make_clustering(mean_sound, BF_neuron, mean_xy, cluster_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[trace_number,~] = size(mean_sound);

%Y = pdist(mean_sound,'euclidean');
Y = pdist(mean_sound,'correlation');
%squareform(Y)
size(squareform(Y))
Z = linkage(Y);

figure
subplot(2,2,1)
[T1,T2,outperform] = dendrogram(Z,0);
[T1,T2,~] = dendrogram(Z,cluster_number);
%dendrogram(Z,25)
%T1
%T2
%outperform

%BF categorizes to same other not_use
cluster_sound = mean_sound(outperform,:);
for i = 1:trace_number,
    for j = 1:trace_number,
        if i ~= j,
            rho_original(i,j) = corr(cluster_sound(i,:)', cluster_sound(j,:)','Type','Pearson');
            rho_raw(i,j) = corr(mean_sound(i,:)', mean_sound(j,:)','Type','Pearson');
            distance_raw(i,j) = sqrt(sum((mean_xy(i,:) - mean_xy(j,:)).*(mean_xy(i,:) - mean_xy(j,:))));
            
            BF_pair(i,j) = 0;
            if BF_neuron(i) == 1,
                if BF_neuron(j) == 1,
                    BF_pair(i,j) = 1;
                elseif BF_neuron(j) == 6,
                    BF_pair(i,j) = -1;
                end
            elseif BF_neuron(i) == 6,
                if BF_neuron(j) == 6,
                    BF_pair(i,j) = 1;
                elseif BF_neuron(j) == 1,
                    BF_pair(i,j) = -1;
                end
            end
        else
            rho_original(i,j) = nan;
            rho_raw(i,j) = nan;
            BF_pair(i,j) = nan;
        end
    end
end
subplot(2,2,2)
imagesc(rho_original)
%axis xy

[min(rho_original(:)), max(rho_original(:))]

number_color = 9;
use_color = jet(number_color);
% % use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
% %              use_color(number_color-2,:); use_color(number_color-1,:); use_color(number_color,:)];
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];

plot_BF_neuron = BF_neuron(outperform)';
subplot(2,2,4)
%imagesc(BF_neuron(outperform)')
for i = 1:length(plot_BF_neuron),
    fill([i i+1 i+1 i],[0 0 1 1],use_color(plot_BF_neuron(i),:),'Edgecolor','none')
    hold on
end
set(gca,'xlim',[1 length(plot_BF_neuron)+1])

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [temp_stim,stim_evi,stim_choice,mean_xy,BF_neuron,X_plot_all,Y_plot_all,X_plot_moto,Y_plot_moto] = ...
    get_data_from_one_session(analysis_folder,filename1,pathname1,filename3,pathname3,temp_mouse, temp_session, neuron_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp_filename = filename1(temp_mouse).name 
temp_path = pathname1;
fpath = fullfile(temp_path, temp_filename);
data = load(fpath);
    
temp_filename = filename3(temp_mouse).name 
temp_path = pathname3;
fpath = fullfile(temp_path, temp_filename);
data3 = load(fpath);

%'stim_sound','stim_norm','stim_task','stim_order_block','stim_mean','stim_std',
%'rew_sound','rew_norm','rew_task','rew_order_block','rew_mean','rew_std'
if length(analysis_folder) == 8, %Stimulus
    stim_sound = data.stim_sound; %df/f
    stim_task = data.stim_task; %[Sound, reward, choice, Evidence, Block];
    stim_xy = data3.stim;
elseif length(analysis_folder) == 6, %reward
    stim_sound = data.rew_sound;
    stim_task = data.rew_task; %[Sound, reward, choice, Evidence, Block];
    stim_xy = data3.rew;
else
    hoge
end

%Going to session
mean_xy = stim_xy(temp_session).matrix.mean_xy;
X_plot_all = stim_xy(temp_session).matrix.X_plot_all;
Y_plot_all = stim_xy(temp_session).matrix.Y_plot_all;
temp_stim = stim_sound(temp_session).matrix;
temp_stim_task = stim_task(temp_session).matrix;
[size_trial,size_neuron] = size(temp_stim);
        
%Each tone cloud
stim_category = temp_stim_task(:,1);
stim_choice = temp_stim_task(:,3);
stim_evi   = temp_stim_task(:,4);
stim_reward = temp_stim_task(:,2);
stim_block = temp_stim_task(:,5);
stim_correct = find(stim_reward == 1);
stim_error   = find(stim_reward == 0);
stim_block_L = find(stim_block == 0);
stim_block_R  = find(stim_block == 1);
        
%BF is determined with activity during reward
for k = 1:6,
    temp_evi = find(stim_evi == k);
    temp_correct = intersect(temp_evi, stim_correct);
    temp_error = intersect(temp_evi, stim_error);
    temp_block_L = intersect(temp_evi, stim_block_L);
    temp_block_R = intersect(temp_evi, stim_block_R);
            
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
end

%Detect BF
for l = 1:size_neuron,
    p_kruskal_stim(l,1) = kruskalwallis(temp_stim(:,l),stim_evi,'off');

    BF_neuron(l,1) = find(median_all(l,:) == max(median_all(l,:)),1);
    BF_neuron(l,2) = find(median_correct(l,:) == max(median_correct(l,:)),1);
    BF_neuron(l,3) = find(median_error(l,:) == max(median_error(l,:)),1);
end
        
X_plot_moto = X_plot_all;
Y_plot_moto = Y_plot_all;
%Get the correlation analysis
temp_stim = temp_stim(:,neuron_number);
mean_xy = mean_xy(neuron_number,:);
BF_neuron = BF_neuron(neuron_number,2);
X_plot_all = X_plot_all(neuron_number);
Y_plot_all = Y_plot_all(neuron_number);

return

