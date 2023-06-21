%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_190502_ver_20230602
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

analysis_folder = 'stimulus';
[stim_length_neuron, stim_BF_neuron, stim_behav_data, ...
    stim_Prefer_sabun, stim_norm_Prefer_sabun, ...
    stim_BF_sabun, stim_norm_BF_sabun] = ...
    get_session_based_activity(analysis_folder);

analysis_folder = 'reward';
[rew_length_neuron, rew_BF_neuron, rew_behav_data, ...
    rew_Prefer_sabun, rew_norm_Prefer_sabun, ...
    rew_BF_sabun, rew_norm_BF_sabun] = ...
    get_session_based_activity(analysis_folder);

[sig_kruskal, sig_kruskal_stim, sig_kruskal_rew, ...
 sig_kruskal_both, moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all
%Based on all_mouse_number make the session data
max_session = max(all_mouse_number);
max_session = max_session(2);
all_session_mouse = [];
for i = 1:max_session
    temp = find(all_mouse_number(:,2) == i);
    temp = all_mouse_number(temp,1);
    temp = unique(temp);
    if length(temp) ~= 1
        hoge
    else
        all_session_mouse(i,1) = temp;
    end
end
%all_session_mouse

[stim_Prefer, easy_stim_Prefer, mid_stim_Prefer, dif_stim_Prefer, stim_length, mouse_number] = ...
    get_session_sabun(stim_length_neuron, sig_kruskal_both, stim_BF_neuron, ...
    stim_Prefer_sabun, stim_norm_Prefer_sabun, stim_BF_sabun, stim_norm_BF_sabun, all_mouse_number);

[rew_Prefer, easy_rew_Prefer, mid_rew_Prefer, dif_rew_Prefer, rew_length,~] = ...
    get_session_sabun(rew_length_neuron, sig_kruskal_both, rew_BF_neuron, ...
    rew_Prefer_sabun, rew_norm_Prefer_sabun, rew_BF_sabun, rew_norm_BF_sabun, all_mouse_number);

%Check the mouse number
temp = all_session_mouse == mouse_number;
if min(temp) ~= 1
    temp
    hoge
else
    clear mouse_number
end

sum(stim_length)
sum(rew_length)
[sum(sum(stim_length)), sum(sum(rew_length))]
%1 all tones
%2 norm all tones
%3 BF activity
%4 BF norm activity
use_number = 4; %BF norm 
thre_neuron = 0;

sum_stim_length = sum(stim_length,2);
sum_rew_length =  sum(rew_length,2);

disp('all neurons')
[session_number(1,:), median_modu(1,:), r(1,:), p(1,:), b(1,:), stats_p(1,:)] = ...
    plot_correlation_20230602(stim_Prefer, sum_stim_length, stim_behav_data, ...
                     rew_Prefer, sum_rew_length, rew_behav_data, ...
                     use_number, thre_neuron, all_session_mouse);
disp('easy neurons')
[session_number(2,:), median_modu(2,:), r(2,:), p(2,:), b(2,:), stats_p(2,:)] = ...
    plot_correlation_20230602(easy_stim_Prefer, stim_length(:,1), stim_behav_data, ...
                     easy_rew_Prefer, rew_length(:,1), rew_behav_data, ...
                     use_number, thre_neuron, all_session_mouse);
disp('mid neurons')
[session_number(3,:), median_modu(3,:), r(3,:), p(3,:), b(3,:), stats_p(3,:)] = ...
    plot_correlation_20230602(mid_stim_Prefer, stim_length(:,2), stim_behav_data, ...
                     mid_rew_Prefer, rew_length(:,2), rew_behav_data, ...
                     use_number, thre_neuron, all_session_mouse);
disp('dif neurons')
[session_number(4,:), median_modu(4,:), r(4,:), p(4,:), b(4,:), stats_p(4,:)] = ...
    plot_correlation_20230602(dif_stim_Prefer, stim_length(:,3), stim_behav_data, ...
                     dif_rew_Prefer, rew_length(:,3), rew_behav_data, ...
                     use_number, thre_neuron, all_session_mouse);
                 
session_number
median_modu
r
p
b
stats_p


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [session_number, median_modulation, r, p, beta, stats_p] = ...
    plot_correlation_20230602(stim_Prefer, stim_length, stim_behav_data, ...
                     rew_Prefer, rew_length, rew_behav_data, ...
                     use_number, thre_neuron, all_session_mouse)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stim_Prefer = stim_Prefer(use_number).matrix;
[~,size_x] = size(stim_Prefer);
if size_x == 2,
    stim_Prefer = stim_Prefer(:,1);
    %stim_Prefer = stim_Prefer(:,2);
end
temp1 = find(stim_length > thre_neuron);

rew_Prefer = rew_Prefer(use_number).matrix;
if size_x == 2,
    rew_Prefer = rew_Prefer(:,1);
    %rew_Prefer = rew_Prefer(:,2);
end
temp2 = find(rew_length > thre_neuron);

temp12 = intersect(temp1,temp2);

figure
plot(stim_Prefer(temp12),rew_Prefer(temp12),'b.')
hold on
plot([-0.3 0.5],[-0.3 0.5],'k')
%plot([-0.8 1.2],[-0.8 1.2],'k')
signrank(stim_Prefer(temp12),rew_Prefer(temp12))
[length(temp12),length(stim_Prefer),length(rew_Prefer)]

%lme
lme = fitlme_analysis_20210520_0(stim_Prefer(temp12)-rew_Prefer(temp12),all_session_mouse(temp12));
%lme(1).lme 
lme(2).lme

behave_data = [stim_behav_data(temp1); rew_behav_data(temp2)];
neuron_data = [stim_Prefer(temp1); rew_Prefer(temp2)];

% [r(1),p(1)] = corr(stim_behav_data(temp1), stim_Prefer(temp1));
% [r(2),p(2)] = corr(rew_behav_data(temp2), rew_Prefer(temp2));
% [r(3),p(3)] = corr(behave_data, neuron_data);
[r(1),p(1)] = corr(stim_behav_data(temp1), stim_Prefer(temp1),'type','Spearman');
[r(2),p(2)] = corr(rew_behav_data(temp2), rew_Prefer(temp2),'type','Spearman');
[r(3),p(3)] = corr(behave_data, neuron_data,'type','Spearman');

session_number = [length(temp1), length(temp2)];
median_modulation = [median(stim_behav_data(temp1)), median(rew_behav_data(temp2))];
%r
%p

%Robust fit
[b,stats] = robustfit(behave_data, neuron_data);
beta(1) = b(2);
temp_p = stats.p;
stats_p(1) = temp_p(2);

%Linear model
%mdl = fitlm(behave_data, neuron_data);
mdl = fitlm(behave_data, neuron_data,'linear','RobustOpts','on');
b = mdl.Coefficients.Estimate;
temp_p = mdl.Coefficients.pValue;
beta(2) = b(2);
stats_p(2) = temp_p(2);

figure

subplot(1,2,1)
temp_x = [-0.1:0.05:0.6];
[Ypred,YCI] = predict(mdl, temp_x');
x_fill = [temp_x, fliplr(temp_x)];
y_fill = [YCI(:,1)', fliplr(YCI(:,2)')];
hold on
plot(temp_x', Ypred,'k')
hold on
fill(x_fill, y_fill,[0.5 0.5 0.5],'edgecolor','none','FaceAlpha',0.3)

subplot(1,2,2)
%plot(stim_behav_data(temp1), stim_Prefer(temp1), 'b.')
plot(stim_behav_data(temp1), stim_Prefer(temp1), '.','color',[0 0.7 0])
hold on
%plot(rew_behav_data(temp2), rew_Prefer(temp2), 'r.')
plot(rew_behav_data(temp2), rew_Prefer(temp2), '.','color',[0.7 0 0.7])


return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [median_Prefer, easy_Prefer, mid_Prefer, dif_Prefer, length_neuron, mouse_number] = ...
    get_session_sabun(stim_length_neuron, sig_kruskal_both, stim_BF_neuron, ...
    stim_Prefer_sabun, stim_norm_Prefer_sabun, stim_BF_sabun, stim_norm_BF_sabun, all_mouse_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get the session based plasticity
cumsum_neuron = cumsum(stim_length_neuron);

%[cumsum_neuron(end), length(all_mouse_number)]
if cumsum_neuron(end) ~= length(all_mouse_number)
    cumsum_neuron(end)
    hoge
end

all_easy_sabun = [];
all_mid_sabun = [];
all_dif_sabun = [];
norm_easy_sabun = [];
norm_mid_sabun = [];
norm_dif_sabun = [];
clear mouse_number
for i = 1:length(stim_length_neuron),
    if i == 1,
        use_neuron = [1:cumsum_neuron(i)];
    else
        use_neuron = [cumsum_neuron(i-1)+1:cumsum_neuron(i)];
    end
    
    temp_sig = intersect(use_neuron, sig_kruskal_both);
    sig_BF = stim_BF_neuron(temp_sig);
    sig_Prefer = stim_Prefer_sabun(temp_sig,:);
    sig_norm_Prefer = stim_norm_Prefer_sabun(temp_sig,:);
    sig_BF_sabun = stim_BF_sabun(temp_sig,:);
    sig_norm_BF_sabun = stim_norm_BF_sabun(temp_sig,:);
    
    median_Prefer(1).matrix(i,:) = median(sig_Prefer);
    median_Prefer(2).matrix(i,:) = median(sig_norm_Prefer);
    median_Prefer(3).matrix(i,:) = median(sig_BF_sabun);
    median_Prefer(4).matrix(i,:) = median(sig_norm_BF_sabun);
    
    clear BF_neuron
    for j = 1:6,
        BF_neuron(j).matrix = find(sig_BF == j);
    end
    easy_neuron = [BF_neuron(1).matrix; BF_neuron(6).matrix];
    mid_neuron =  [BF_neuron(2).matrix; BF_neuron(5).matrix];
    dif_neuron =  [BF_neuron(3).matrix; BF_neuron(4).matrix];
    if length(easy_neuron) ~= 0,
        easy_Prefer(1).matrix(i,:) = median(sig_Prefer(easy_neuron,:),1);
        easy_Prefer(2).matrix(i,:) = median(sig_norm_Prefer(easy_neuron,:),1);
        easy_Prefer(3).matrix(i,:) = median(sig_BF_sabun(easy_neuron,:),1);
        easy_Prefer(4).matrix(i,:) = median(sig_norm_BF_sabun(easy_neuron,:),1);
        all_easy_sabun = [all_easy_sabun; sig_BF_sabun(easy_neuron,:)];
        norm_easy_sabun = [norm_easy_sabun; sig_norm_BF_sabun(easy_neuron,:)];
    else
        easy_Prefer(1).matrix(i,:) = nan(1,2);
        easy_Prefer(2).matrix(i,:) = nan(1,2);
        easy_Prefer(3).matrix(i,:) = nan(1,1);
        easy_Prefer(4).matrix(i,:) = nan(1,1);
    end
    if length(mid_neuron) ~= 0,
        mid_Prefer(1).matrix(i,:) = median(sig_Prefer(mid_neuron,:),1);
        mid_Prefer(2).matrix(i,:) = median(sig_norm_Prefer(mid_neuron,:),1);
        mid_Prefer(3).matrix(i,:) = median(sig_BF_sabun(mid_neuron,:),1);
        mid_Prefer(4).matrix(i,:) = median(sig_norm_BF_sabun(mid_neuron,:),1);
        all_mid_sabun = [all_mid_sabun; sig_BF_sabun(mid_neuron,:)];
        norm_mid_sabun = [norm_mid_sabun; sig_norm_BF_sabun(mid_neuron,:)];
    else
        mid_Prefer(1).matrix(i,:) = nan(1,2);
        mid_Prefer(2).matrix(i,:) = nan(1,2);
        mid_Prefer(3).matrix(i,:) = nan(1,1);
        mid_Prefer(4).matrix(i,:) = nan(1,1);
    end
    if length(dif_neuron) ~= 0,
        dif_Prefer(1).matrix(i,:) = median(sig_Prefer(dif_neuron,:),1);
        dif_Prefer(2).matrix(i,:) = median(sig_norm_Prefer(dif_neuron,:),1);
        dif_Prefer(3).matrix(i,:) = median(sig_BF_sabun(dif_neuron,:),1);
        dif_Prefer(4).matrix(i,:) = median(sig_norm_BF_sabun(dif_neuron,:),1);
        all_dif_sabun = [all_dif_sabun; sig_BF_sabun(dif_neuron,:)];
        norm_dif_sabun = [norm_dif_sabun; sig_norm_BF_sabun(dif_neuron,:)];
    else
        dif_Prefer(1).matrix(i,:) = nan(1,2);
        dif_Prefer(2).matrix(i,:) = nan(1,2);
        dif_Prefer(3).matrix(i,:) = nan(1,1);
        dif_Prefer(4).matrix(i,:) = nan(1,1);
    end
    length_neuron(i,:) = [length(easy_neuron), length(mid_neuron), length(dif_neuron)];

    %Mouse number
    temp_mouse = all_mouse_number(temp_sig,:);
    temp_mouse1 = unique(temp_mouse(:,1));
    temp_mouse2 = unique(temp_mouse(:,2));
    if length(temp_mouse2) ~= 1
        hoge
    end
    if length(temp_mouse1) ~= 1
        hoge
    else
        mouse_number(i,1) = temp_mouse1;
    end
end

[median(all_easy_sabun), median(all_mid_sabun), median(all_dif_sabun)]
[median(norm_easy_sabun), median(norm_mid_sabun), median(norm_dif_sabun)]
p_sign(1,1) = signrank(all_easy_sabun);
p_sign(1,2) = signrank(all_mid_sabun);
p_sign(1,3) = signrank(all_dif_sabun);
p(1,1) = ranksum(all_easy_sabun, all_mid_sabun);
p(1,2) = ranksum(all_easy_sabun, all_dif_sabun);
p(1,3) = ranksum(all_mid_sabun, all_dif_sabun);

p_sign(2,1) = signrank(norm_easy_sabun);
p_sign(2,2) = signrank(norm_mid_sabun);
p_sign(2,3) = signrank(norm_dif_sabun);
p(2,1) = ranksum(norm_easy_sabun, norm_mid_sabun);
p(2,2) = ranksum(norm_easy_sabun, norm_dif_sabun);
p(2,3) = ranksum(norm_mid_sabun, norm_dif_sabun);

p_sign
p

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_length_neuron, all_BF_neuron, behav_data, ...
    all_Prefer_sabun, all_norm_Prefer_sabun, ...
    all_BF_sabun, all_norm_BF_sabun] = ...
    get_session_based_activity(analysis_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

stim_behavior = [0.2264
0.1471
0.0719
0.1344
0.0838
0.0235
0.0919
0.0604
0.1343
-0.0027
0.2099
0.304
0.2182
-0.0303
0.1186
0.1811
0.1324
0.1648
0.2535
0.1663
0.0857
0.1667
0.161
0.2391
0.073
0.0836
0.19
0.2715
0.11
0.0999
0.1436
0.0772
0.14
0.1977
0.0803
0.1739
0.0617
0.2251
0.2022
0.0888
0.215
0.1138
0.2425
0.3176
0.165
0.2801
0.1796
0.1312
0.1785
0.23
0.1463
0.1837
0.2796
0.199
0.1908
0.1851
0.1001
0.137
0.0212
0.0658
0.2725
0.0852
0.2058
0.083
-0.0122
0.0423
0.0558
0.2783
0.2039
0.2832
0.0451
0.2134
0.0788
0.1341
0.1247
0.1528
0.1687
-0.031
0.0759
0.0775
0.1546
0.007
0.1123
];

reward_behavior = [0.3049
0.314
0.371
0.4121
0.1811
0.3336
0.2722
0.2603
0.2725
0.4261
0.1577
0.5175
0.3215
0.4599
0.2496
0.4045
0.2907
0.3689
0.4787
0.4389
0.4081
0.5516
0.5479
0.3988
0.4186
0.4964
0.4046
0.5293
0.3001
0.4409
0.4677
0.3461
0.3102
0.3071
0.1998
0.3893
0.3564
0.4917
0.4274
0.3824
0.3702
0.4182
0.3816
0.4207
0.3005
0.2306
0.3176
0.5434
0.3824
0.4076
0.1928
0.5297
0.3442
0.1294
0.4141
0.2433
0.4119
0.2426
0.1339
0.2241
0.5934
0.4395
0.2561
0.3415
0.2611
0.3227
0.0755
0.3165
0.1856
0.3277
0.4245
0.1791
0.4447
0.262
0.3763
0.1492
0.176
0.078
0.3432
0.1482
0.2005
0.0713
0.3068
];

if length(analysis_folder) == 8, %Stimulus
    behav_data = stim_behavior; %df/f
elseif length(analysis_folder) == 6, %reward
    behav_data = reward_behavior;
else
    hoge
end

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

%[filename1, pathname1,findex]=uigetfile('*.mat','Sound_file','Multiselect','on');
%pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_RE_20180610_no_lick/',analysis_folder);
pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/process_190214');
cd(pathname1)
filename1 = dir('*.mat');
%all_sound all_sound_L all_sound_R all_sound_category all_sound_max_time 
%all_sig_sound all_sig_sound_S all_sig_sound_L all_sig_sound_R 
%all_block_L all_block_R 
%all_block_LL all_block_LR all_block_RL all_block_RR 
%all_block_category_L all_block_category_R all_block_max_time
%all_roi_overlap

temp = ['cd ',default_folder];
eval(temp); %move directory
% if length(analysis_folder) == 8, %Stimulus
% % [filename4, pathname4]=uigetfile('*.mat','Sound_file');
% % fpath = fullfile(pathname4, filename4);
% % load(fpath);
%     load('tuning_stimulus_task.mat')
% elseif length(analysis_folder) == 6, %reward
%     load('tuning_reward_task.mat')
% else
%     hoge
% end
% load('tuning_integrate_task.mat')

all_BF_neuron = [];
all_median_low_sound = [];
all_median_high_sound = [];
all_p_block_dif = [];

all_Prefer_sabun = [];
all_norm_Prefer_sabun = [];
all_BF_sabun = [];
all_norm_BF_sabun = [];
all_length_neuron = [];

for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    %'stim_sound','stim_norm','stim_task','stim_order_block','stim_mean','stim_std',
    %'rew_sound','rew_norm','rew_task','rew_order_block','rew_mean','rew_std'
    if length(analysis_folder) == 8, %Stimulus
        stim_data = data.stim_data; %df/f
    elseif length(analysis_folder) == 6, %reward
        stim_data = data.rew_data;
    else
        hoge
    end

    clear length_neuron_BF session_length_neuron
    length_session = length(stim_data.mean_sound);
    for j = 1:length_session,
        
        temp_BF  = stim_data.BF(j).matrix;
        temp_BF = temp_BF(:,2); %use reward trials
        temp_std = stim_data.std_sound(j).matrix;
        temp_p_block_dif  = stim_data.p_block_dif(j).matrix;
        
        temp_low = stim_data.median_sound_low(j).matrix; %L R All (blocks)
        temp_high = stim_data.median_sound_high(j).matrix;
        temp_std_low = stim_data.std_sound_low(j).matrix;
        temp_std_high = stim_data.std_sound_high(j).matrix;
        
        temp_block_L = stim_data.median_block_L(j).matrix;
        temp_block_R = stim_data.median_block_R(j).matrix;
        
        %all_p_kruskal_stim
        %all_median_category_R
        %all_median_category_L
        temp_std = temp_std';
        temp_std = repmat(temp_std,1,2);
        
        all_median_high_sound = [all_median_high_sound; temp_high];
        all_median_low_sound = [all_median_low_sound; temp_low];
        all_p_block_dif = [all_p_block_dif; temp_p_block_dif];
        all_BF_neuron = [all_BF_neuron; temp_BF];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Each session get how much the neurons change
        temp_BF_low = find(temp_BF < 3.5);
        temp_BF_high = find(temp_BF > 3.5);
        [Prefer_sabun, norm_Prefer_sabun] = get_prefer_sabun(temp_low, temp_high, temp_std, temp_BF_low, temp_BF_high);
        %[Prefer-tone, NonPrefer tones] 
        
        %Each prefer tone, get the block difference
        [BF_sabun, norm_BF_sabun] = get_BF_sabun(temp_block_L, temp_block_R, temp_std, temp_BF);
        
        all_Prefer_sabun = [all_Prefer_sabun; Prefer_sabun];
        all_norm_Prefer_sabun = [all_norm_Prefer_sabun; norm_Prefer_sabun];
        all_BF_sabun = [all_BF_sabun; BF_sabun];
        all_norm_BF_sabun = [all_norm_BF_sabun; norm_BF_sabun];
        
        session_length_neuron(j) = length(temp_BF);
    end    
    all_length_neuron = [all_length_neuron; session_length_neuron'];
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

size(all_Prefer_sabun)
median(all_Prefer_sabun)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Prefer_sabun, norm_Prefer_sabun] = get_prefer_sabun(sig_temp_low, sig_temp_high, sig_std, temp_BF_low, temp_BF_high)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        clear prefer_sabun_category norm_prefer_sabun_category
        clear Prefer_sabun NonPrefer_sabun
        clear norm_Prefer_sabun norm_NonPrefer_sabun
        prefer_sabun_category(:,1) = sig_temp_low(:,1) - sig_temp_low(:,2); %Low tones (block_L - block_R)
        prefer_sabun_category(:,2) = sig_temp_high(:,2) - sig_temp_high(:,1); %High tones (block_R - block_L)
        norm_prefer_sabun_category = prefer_sabun_category ./ sig_std;
        
        Prefer_sabun = zeros(length(sig_std),1);
        NonPrefer_sabun = zeros(length(sig_std),1);
        Prefer_sabun(temp_BF_low) = prefer_sabun_category(temp_BF_low,1);
        Prefer_sabun(temp_BF_high) = prefer_sabun_category(temp_BF_high,2);
        NonPrefer_sabun(temp_BF_low) = -1*prefer_sabun_category(temp_BF_low,2);
        NonPrefer_sabun(temp_BF_high) = -1*prefer_sabun_category(temp_BF_high,1);
        Prefer_sabun = [Prefer_sabun, NonPrefer_sabun];
        
        norm_Prefer_sabun = zeros(length(sig_std),1);
        norm_NonPrefer_sabun = zeros(length(sig_std),1);
        norm_Prefer_sabun(temp_BF_low) = norm_prefer_sabun_category(temp_BF_low,1); 
        norm_Prefer_sabun(temp_BF_high) = norm_prefer_sabun_category(temp_BF_high,2);
        norm_NonPrefer_sabun(temp_BF_low) = -1*norm_prefer_sabun_category(temp_BF_low,2)
        norm_NonPrefer_sabun(temp_BF_high) = -1*norm_prefer_sabun_category(temp_BF_high,1);
        norm_Prefer_sabun = [norm_Prefer_sabun, norm_NonPrefer_sabun];
        
        return

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Prefer_sabun, norm_Prefer_sabun] = get_BF_sabun(temp_block_L, temp_block_R, temp_std, temp_BF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Prefer_sabun = zeros(length(temp_std),1);
norm_Prefer_sabun = zeros(length(temp_std),1);

for i = 1:length(temp_BF),
    temp = temp_BF(i);
    if temp < 3.5, %low tones
        sabun = temp_block_L(i,temp) - temp_block_R(i,temp);
    else
        sabun = temp_block_R(i,temp) - temp_block_L(i,temp);
    end
    norm_sabun = sabun ./ temp_std(i,1);
    
    Prefer_sabun(i) = sabun;
    norm_Prefer_sabun(i) = norm_sabun;
end

return

