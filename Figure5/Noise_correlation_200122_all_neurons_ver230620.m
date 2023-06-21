%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function Noise_correlation_200122_all_neurons_ver230620
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
all_BF_neuron = [];

all_length_neuron = [];
all_number_thre = [];
all_all = [];
all_same = [];
all_other = [];

all_corr_same = [];
all_p_corr_same = [];
all_b_same = [];
all_p_b_same = [];

all_corr_other = [];
all_p_corr_other = [];
all_b_other = [];
all_p_b_other = [];

mouse_all = [];
mouse_same = [];
mouse_other = [];
        
%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, ...
 moto_sig_kruskal_both, moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all

%Decide which neurons to use
sig_kruskal_sound = moto_sig_kruskal_both;

count = 0;
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    
    temp_filename = filename3(i).name 
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
    mouse_number = i;
    
    clear number_thre length_neuron
    clear corr_same p_corr_same b_same p_b_same
    clear corr_other p_corr_other b_other p_b_other
    length_session = length(stim_sound);
    for j = 1:length_session,
        count = count + 1;
        mean_xy = stim_xy(j).matrix.mean_xy;
        temp_stim = stim_sound(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);

        %Get the sound responsive neurons for testing
        all_size_neuron(count) = size_neuron;
        sum_neuron = sum(all_size_neuron);
        haba_neuron = [sum_neuron - size_neuron + 1 : sum_neuron];
        [use_neuron,neuron_number] = intersect(haba_neuron,sig_kruskal_sound);
        neuron_number
        
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
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear p_kruskal_stim BF_neuron p_RE_neuron
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
        all_BF_neuron = [all_BF_neuron; BF_neuron];
        
        %Get the correlation analysis
        temp_stim = temp_stim(:,neuron_number);
        mean_xy = mean_xy(neuron_number,:);
        
        temp_stim = temp_stim';
        decode(count).matrix = Noise_correlation_200120_process4_save_only...
            (temp_stim,stim_category,stim_evi,stim_choice,stim_reward,stim_block,mean_xy);
        
            [length_neuron(j,1), number_thre(j,:), distance_corr_noise_all, distance_corr_noise_same, distance_corr_noise_other, ...
                corr_same(j,:),  p_corr_same(j,:),  b_same(j,:),  p_b_same(j,:), ...
                corr_other(j,:), p_corr_other(j,:), b_other(j,:), p_b_other(j,:)] = get_noise_session_20200120(decode(count).matrix);
        
        all_all = [all_all; distance_corr_noise_all];
        all_same = [all_same; distance_corr_noise_same];
        all_other = [all_other; distance_corr_noise_other];
        
        %about mouse number
        temp_mouse = ones(size(distance_corr_noise_all,1),1) * mouse_number;
        temp_session = ones(size(distance_corr_noise_all,1),1) * count;
        temp_mouse = [temp_mouse, temp_session];
        mouse_all = [mouse_all; temp_mouse];
        
        temp_mouse = ones(size(distance_corr_noise_same,1),1) * mouse_number;
        temp_session = ones(size(distance_corr_noise_same,1),1) * count;
        temp_mouse = [temp_mouse, temp_session];
        mouse_same = [mouse_same; temp_mouse];
        
        temp_mouse = ones(size(distance_corr_noise_other,1),1) * mouse_number;
        temp_session = ones(size(distance_corr_noise_other,1),1) * count;
        temp_mouse = [temp_mouse, temp_session];
        mouse_other = [mouse_other; temp_mouse];
    end    
    all_length_neuron = [all_length_neuron; length_neuron];
    all_number_thre = [all_number_thre; number_thre];
    
    all_corr_same = [all_corr_same; corr_same];
    all_p_corr_same = [all_p_corr_same; p_corr_same];
    all_b_same = [all_b_same; b_same];
    all_p_b_same = [all_p_b_same; p_b_same];

    all_corr_other = [all_corr_other; corr_other];
    all_p_corr_other = [all_p_corr_other; p_corr_other];
    all_b_other = [all_b_other; b_other];
    all_p_b_other = [all_p_b_other; p_b_other];
end

%Check the Nan component
nan_all = find(isnan(all_all(:,2)) == 1);
if length(nan_all) ~= 0,
    disp('nan all')
    all_all(nan_all,1)
    all_all(nan_all,:) = [];
    mouse_all(nan_all,:) = [];
end
nan_same = find(isnan(all_same(:,2)) == 1);
if length(nan_same) ~= 0,
    disp('nan same')
    all_same(nan_same,1)
    all_same(nan_same,:) = [];
    mouse_same(nan_same,:) = [];
end
nan_other = find(isnan(all_other(:,2)) == 1);
if length(nan_other) ~= 0,
    disp('nan other')
    all_other(nan_other,1)
    all_other(nan_other,:) = [];
    mouse_other(nan_other,:) = [];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

length(all_length_neuron)
sum(all_length_neuron)

close all

[corr_same,  p_corr_same,  b_same,  p_b_same]  = get_noise_robust_fit(all_same);
[corr_other, p_corr_other, b_other, p_b_other] = get_noise_robust_fit(all_other);

plot_all_noise_corr(all_same, all_b_same)
plot_all_noise_corr(all_other, all_b_other)

plot_all_noise_comparison(all_same, all_other, all_all, mouse_same, mouse_other)

%All neuron at once
corr_same
p_corr_same
b_same
p_b_same

corr_other
p_corr_other
b_other
p_b_other

%Neuron in each session
[nanmedian(all_corr_same), nanmedian(all_corr_other)]
[nanmedian(all_b_same), nanmedian(all_b_other)]

[signrank(all_b_same(:,2)), signrank(all_b_same(:,4))]
[signrank(all_b_other(:,2)), signrank(all_b_other(:,4))]

test_nan = find(isnan(all_corr_same(:,1)) == 0);
length(test_nan)

ranksum(all_corr_same(:,2),all_corr_other(:,2))

figure
temp_label = [zeros(length(all_same),1); ones(length(all_other),1)];
temp_plot1 = [all_same(:,2); all_other(:,2)];
temp_plot2 = [all_same(:,3); all_other(:,3)];

subplot(1,2,1)
h = boxplot(temp_plot1, temp_label)
set(h(7,:),'Visible','off');
set(gca,'ylim',[-1 1])
subplot(1,2,2)
h = boxplot(temp_plot2, temp_label)
set(h(7,:),'Visible','off');
set(gca,'ylim',[-0.3 0.5])

size(all_all)
size(all_same)
size(all_other)

for i = 2:3
    ranksum(all_same(:,i), all_other(:,i)) %Signal correlation
    temp_data1 = [all_same(:,i); all_other(:,i)];
    temp_data2 = [zeros(length(all_same(:,i)),1); ones(length(all_other(:,i)),1)];
    temp_data3 = [mouse_same; mouse_other];
%     size(temp_data1)
%     size(temp_data2)
%     size(temp_data3)
    temp_data = [temp_data1, temp_data2, temp_data3];
    [lme,AIC_model,BIC_model,p_AIC_BIC] = fitlme_analysis_20210520_2_ranksum(temp_data);
    p_AIC_BIC
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_all_noise_comparison(all_same, all_other, all_all, mouse_same, mouse_other)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

distance_same = all_same(:,1);
corr_same = all_same(:,2);
noise_same = all_same(:,3);

distance_other = all_other(:,1);
corr_other = all_other(:,2);
noise_other = all_other(:,3);

distance_all = all_all(:,1);
corr_all = all_all(:,2);
noise_all = all_all(:,3);

distance_bin = [0:100:500,5000];
distance_x = [0:100:500] + 50;
%distance_bin = [0:100:500,600];
figure
for i = 1:length(distance_bin)-1,
    temp_all  = find(distance_all >= distance_bin(i) & distance_all < distance_bin(i+1));
    temp_same  = find(distance_same >= distance_bin(i) & distance_same < distance_bin(i+1));
    temp_other = find(distance_other >= distance_bin(i) & distance_other < distance_bin(i+1));
    
    temp_corr_all = corr_all(temp_all);
    temp_noise_all = noise_all(temp_all);
    temp_corr_same = corr_same(temp_same);
    temp_noise_same = noise_same(temp_same);
    temp_corr_other = corr_other(temp_other);
    temp_noise_other = noise_other(temp_other);
    
    p_corr(i) = ranksum(temp_corr_same,temp_corr_other);
    p_noise(i) = ranksum(temp_noise_same,temp_noise_other);
    temp_data3 = [mouse_same(temp_same,:); mouse_other(temp_other,:)];
    %fitlme
    temp_data1 = [temp_corr_same; temp_corr_other];
    temp_data2 = [zeros(length(temp_corr_same),1); ones(length(temp_corr_other),1)];
    temp_data = [temp_data1, temp_data2, temp_data3];
    [lme,AIC_model,BIC_model,p_corr_AIC_BIC(i,:)] = fitlme_analysis_20210520_2_ranksum(temp_data);
    %fitlme
    temp_data1 = [temp_noise_same; temp_noise_other];
    temp_data2 = [zeros(length(temp_noise_same),1); ones(length(temp_noise_other),1)];
    temp_data = [temp_data1, temp_data2, temp_data3];
    [lme,AIC_model,BIC_model,p_noise_AIC_BIC(i,:)] = fitlme_analysis_20210520_2_ranksum(temp_data);
    
    median_corr_all(i).matrix = temp_corr_all;
    median_noise_all(i).matrix = temp_noise_all;
    median_corr_same(i).matrix = temp_corr_same;
    median_noise_same(i).matrix = temp_noise_same;
    median_corr_other(i).matrix = temp_corr_other;
    median_noise_other(i).matrix = temp_noise_other;
    
    length_neuron(i,:) = [length(temp_same), length(temp_other)];
    
    subplot(1,length(distance_bin)-1,i)
    h = boxplot([temp_noise_same;temp_noise_other], [zeros(length(temp_same),1); ones(length(temp_other),1)]);
    set(h(7,:),'Visible','off');
    %set(gca,'ylim',[-1 1])
    %set(gca,'ylim',[-0.4 0.4])
    set(gca,'ylim',[-0.3 0.5])
end
p_corr
p_noise
p_noise(5)
p_noise(6)

disp('p_corr_AIC_BIC')
p_corr_AIC_BIC
for i = 1:length(distance_bin)-1
    p_corr_AIC_BIC(i,2)
end
disp('p_noise_AIC_BIC')
p_noise_AIC_BIC
for i = 1:length(distance_bin)-1
    p_noise_AIC_BIC(i,2)
end

%Correlation all preferred tone cloud
figure

subplot(1,2,1)
plot_median_se_moto_x_axis_matrix(median_corr_all,distance_x,[0 0 0],2)
%plot_median_se_moto_x_axis_matrix(median_corr_all,distance_x,[0 0 0],0)
hold on
plot_median_se_moto_x_axis_matrix(median_corr_same,distance_x,[1 0 0],2)
%plot_median_se_moto_x_axis_matrix(median_corr_same,distance_x,[1 0 0],0)
hold on
plot_median_se_moto_x_axis_matrix(median_corr_other,distance_x,[0 0 1],2)
%plot_median_se_moto_x_axis_matrix(median_corr_other,distance_x,[0 0 1],0)
set(gca,'xlim',[0 650])
subplot(1,2,2)
plot_median_se_moto_x_axis_matrix(median_noise_all,distance_x,[0 0 0],2)
%plot_median_se_moto_x_axis_matrix(median_noise_all,distance_x,[0 0 0],0)
hold on
plot_median_se_moto_x_axis_matrix(median_noise_same,distance_x,[1 0 0],2)
%plot_median_se_moto_x_axis_matrix(median_noise_same,distance_x,[1 0 0],0)
hold on
plot_median_se_moto_x_axis_matrix(median_noise_other,distance_x,[0 0 1],2)
%plot_median_se_moto_x_axis_matrix(median_noise_other,distance_x,[0 0 1],0)
set(gca,'xlim',[0 650])

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_all_noise_corr(all_same, all_b_same)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

distance_all = all_same(:,1);
corr_all = all_same(:,2);
noise_all = all_same(:,3);

figure
subplot(1,2,1)
plot(distance_all, corr_all, '.','color',[0.5 0.5 0.5])
%plot(distance_all, corr_all, 'b.')
hold on
subplot(1,2,2)
plot(distance_all, noise_all, '.','color',[0.5 0.5 0.5])
%plot(distance_all, noise_all, 'r.')
hold on
    
[b_corr,stat_corr]  = robustfit(distance_all,corr_all);
[b_noise,stat_noise] = robustfit(distance_all,noise_all);

x = [min(distance_all), max(distance_all)];
y_corr  = x * b_corr(2) + b_corr(1);
y_noise = x * b_noise(2) + b_noise(1);

%Plot line for each session
x_step = 5;
temp_x = [min(distance_all): x_step : max(distance_all)];

[size_session,~] = size(all_b_same);

use_color = jet(size_session);
%figure
hold on
for i = 1:size_session,
    temp_y = temp_x .* all_b_same(i,2) + all_b_same(i,1);
    y_sound_plot(i,:) = temp_y;

    temp_y = temp_x .* all_b_same(i,4) + all_b_same(i,3);
    y_noise_plot(i,:) = temp_y;
end

subplot(1,2,1)
hold on
plot_mean_se_moto_x_axis(y_sound_plot,temp_x,[1 0 0],2)
subplot(1,2,2)
hold on
plot_mean_se_moto_x_axis(y_noise_plot,temp_x,[0 0 1],2)

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [length_neuron, number_thre_neuron, distance_corr_noise_all, distance_corr_noise_same, distance_corr_noise_other, ...
    corr_same,  p_corr_same,  b_same,  p_b_same, ...
    corr_other, p_corr_other, b_other, p_b_other] = get_noise_session_20200120(stim_session)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norm_residual  = stim_session.norm_residual;
norm_BF_neuron = stim_session.norm_BF_neuron;
norm_mean_xy   = stim_session.norm_mean_xy;
norm_sound     = stim_session.norm_sound;

%ALready changed to um
%Change norm_mean_xy from pixel to um
%x axis 380um <- 512 pixel
%y axis 550um <- 512 pixel
% norm_mean_xy(:,1) = norm_mean_xy(:,1) .* (380/512); %change to um
% norm_mean_xy(:,2) = norm_mean_xy(:,2) .* (550/512); %change to um

rho = stim_session.rho;
rho_sound = stim_session.rho_sound;
BF_map_pre = stim_session.BF_map_pre;
BF_map_post = stim_session.BF_map_post;

%Start analysis
use_rho = triu(rho);
BF_map_pre = triu(BF_map_pre);
BF_map_post = triu(BF_map_post);
%First high noise correlation is shared across same BF or does not matter
thre_correlation = [0.2 : 0.1 : 0.7];
for i = 1:length(thre_correlation),
    temp = find(use_rho >= thre_correlation(i));
    clear temp_neuron
    temp_neuron = [];
    for j = 1:length(temp),
        temp_neuron(j,:) = [BF_map_pre(temp(j)),BF_map_post(temp(j))];
    end
    BF_map_thre(i).matrix = temp_neuron;
    number_thre_neuron(i) = length(temp);
end

length_neuron = length(norm_BF_neuron);
number_thre_neuron
%Yes, same BF neuron share the same noise correlation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NEXT
%It seems that the correlation is determined by the function and not the
%distance between neurons, need to check

%focus on the most high and low neurons
BF_high = find(norm_BF_neuron == 6);
BF_low  = find(norm_BF_neuron == 1);

xy_all  = norm_mean_xy([1:length_neuron],:);
xy_high = norm_mean_xy(BF_high,:);
xy_low  = norm_mean_xy(BF_low,:);

distance_corr_noise_all = plot_noise_correlation([1:length_neuron], [1:length_neuron], xy_all, xy_all, rho_sound, rho);

%Plot the correlation and distance profile
distance_corr_noise_other = plot_noise_correlation(BF_high, BF_low, xy_high, xy_low, rho_sound, rho);
%same for the high->low case
%[b_corr(2,:), p_stat(2,:), corr_median(2,:), p_median(2,:),~] = plot_noise_correlation(BF_low, BF_high, xy_low, xy_high, rho_sound, rho);

distance_corr_noise_high = plot_noise_correlation(BF_high, BF_high, xy_high, xy_high, rho_sound, rho);
distance_corr_noise_low = plot_noise_correlation(BF_low, BF_low, xy_low, xy_low, rho_sound, rho);
distance_corr_noise_same = [distance_corr_noise_high; distance_corr_noise_low];

[corr_same,  p_corr_same,  b_same,  p_b_same]  = get_noise_robust_fit(distance_corr_noise_same);
[corr_other, p_corr_other, b_other, p_b_other] = get_noise_robust_fit(distance_corr_noise_other);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [corr_median, p_median, b_keisu, p_stat] = get_noise_robust_fit(distance_corr_noise)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_neuron,~] = size(distance_corr_noise);

if size_neuron >= 10,
    distance_all = distance_corr_noise(:,1);
    corr_all = distance_corr_noise(:,2);
    noise_all = distance_corr_noise(:,3);

    [b_corr,stat_corr]  = robustfit(distance_all,corr_all);
    [b_noise,stat_noise] = robustfit(distance_all,noise_all);

    corr_median = [nanmedian(corr_all), nanmedian(noise_all)];
    p_median = [signrank(corr_all), signrank(noise_all)];
    b_keisu = [b_corr(1), b_corr(2), b_noise(1), b_noise(2)];
    p_stat = [stat_corr.p(1),stat_corr.p(2), stat_noise.p(1),stat_noise.p(2)];
else
    corr_median = nan(1,2);
    p_median = nan(1,2);
    b_keisu = nan(1,4);
    p_stat = nan(1,4);
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function distance_corr_noise = plot_noise_correlation(BF_high, BF_low, xy_high, xy_low, rho_sound, rho_residual)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

distance_all = [];
corr_all = [];
noise_all = [];

for i = 1:length(BF_high),
    temp_dif  = xy_low  - repmat(xy_high(i,:),length(BF_low),1);
    
    %distance
    temp_dif = sum(temp_dif .* temp_dif,2);
    temp_dif = sqrt(temp_dif);
    
    clear use_correlation use_noise
    use_correlation = [];
    use_noise = [];
    for j = 1:length(BF_low),
        use_correlation(j,1) = rho_sound(BF_high(i),BF_low(j));
        use_noise(j,1) = rho_residual(BF_high(i),BF_low(j));
    end
    
    distance_all = [distance_all; temp_dif];
    corr_all = [corr_all; use_correlation];
    noise_all = [noise_all; use_noise];
end

[~,temp] = unique(distance_all);
[length(temp), length(distance_all)]

distance_all = distance_all(temp);
corr_all = corr_all(temp);
noise_all = noise_all(temp);

distance_corr_noise = [distance_all, corr_all, noise_all];

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_raster_each_neuron_RE(i,all_median_correct,all_median_error,all_BF_freq,sig_low,sig_high,use_color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:6,
        BF_freq(j).matrix = find(all_BF_freq == j);
    end
    BF_freq(1).matrix = intersect(BF_freq(1).matrix,sig_low);
    BF_freq(2).matrix = intersect(BF_freq(2).matrix,sig_low);
    BF_freq(3).matrix = intersect(BF_freq(3).matrix,sig_low);
    BF_freq(4).matrix = intersect(BF_freq(4).matrix,sig_high);
    BF_freq(5).matrix = intersect(BF_freq(5).matrix,sig_high);
    BF_freq(6).matrix = intersect(BF_freq(6).matrix,sig_high);
    
    if length(sig_low) ~= 0,
          plot(all_median_correct(BF_freq(1).matrix,i),all_median_error(BF_freq(1).matrix,i),'o','color',use_color)
          hold on
         plot(all_median_correct(BF_freq(2).matrix,i),all_median_error(BF_freq(2).matrix,i),'d','color',use_color)
         hold on
          plot(all_median_correct(BF_freq(3).matrix,i),all_median_error(BF_freq(3).matrix,i),'+','color',use_color)
          hold on
    end
    if length(sig_high) ~= 0,
         plot(all_median_correct(BF_freq(4).matrix,7-i),all_median_error(BF_freq(4).matrix,7-i),'+','color',use_color)
         hold on
         plot(all_median_correct(BF_freq(5).matrix,7-i),all_median_error(BF_freq(5).matrix,7-i),'d','color',use_color)
         hold on
         plot(all_median_correct(BF_freq(6).matrix,7-i),all_median_error(BF_freq(6).matrix,7-i),'o','color',use_color)
         hold on
    end
return

