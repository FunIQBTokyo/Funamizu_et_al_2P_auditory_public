%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_190506_sound_only_compare_181021_ver20230531
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Stim rew order
%0: stim -> rew
%1: rew -> stim
a023_order = [0 0 1 0 1 0 1 0 1 0];
a025_order = [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0];
a026_order = [0 1 0 1 0 1 0 1 0 1 0 1 0 1];
a030_order = [0 1 0 1 0 1 0 1 0 1 0 1];
fr00_order = [0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 0 1];
fr02_order = [1 1 0 1 0 1 0 1 0];
order_session = [a023_order'; a025_order'; a026_order'; a030_order'; fr00_order'; fr02_order'];

%Day difference
a023_day = [1 1 1 2 3 1 2 1 2 1];
a025_day = [1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2];
a026_day = [1 2 1 1 1 2 1 1 3 4 1 2 2 1];
a030_day = [3 1 1 3 2 1 1 1 1 1 1 2];
fr00_day = [1 1 1 2 3 3 1 1 1 2 1 1 1 1 1 1 1];
fr02_day = [1 3 1 2 1 1 1 2 1];
day_session = [a023_day'; a025_day'; a026_day'; a030_day'; fr00_day'; fr02_day'];

currentFolder = pwd;
%analysis_folder = 'stimulus';
%analysis_folder = 'reward';

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
cd(pathname1)
filename1 = dir('*.mat');

pathname2 = strcat('e:/Tone_discri1/all_mouse/overlap_matrix_20181021');
cd(pathname2)
filename2 = dir('*.mat');

pathname3 = strcat('e:/Tone_discri1/all_mouse/single_decode/separate4/regress_0910/lognfit_each_task');
cd(pathname3)
filename3 = dir('*.mat');

[sig_kruskal, sig_kruskal_stim, sig_kruskal_rew, sig_kruskal_both] = ...
                tuning_curve_190429_sound_only_compare_block4_prefer;
close all

cd(default_folder)

clear neuron_number

all_sig_sound_or = [];
all_sig_sound_only = [];

all_stim_sound = [];
all_rew_sound = [];
all_stim_sound_evi = [];
all_rew_sound_evi = [];
all_p_sound = [];
all_p_sound_evi = [];

all_p_stim = [];
all_p_rew = [];
all_sig_sound_number = [];

order_block = [];
thre = 0.01;
count_session = 0;
all_mouse_number = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath
    
    temp_filename = filename1(i).name 
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);

    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);
    
    %'stim_sound','stim_baseline','stim_task','stim_order_block','rew_sound','rew_baseline','rew_task','rew_order_block'
    stim_sound = data.stim_sound; %df/f
    %stim_baseline = data.stim_baseline;
    stim_task = data.stim_task; %[Sound, reward, choice, Evidence, Block];
    
    rew_sound = data.rew_sound;
    %rew_baseline = data.rew_baseline;
    rew_task = data.rew_task; %[Sound, reward, choice, Evidence, Block];

    stim_order_block = data.stim_order_block;
    
    length_session = length(stim_sound);
    clear p_stim p_rew p_stim_evi p_rew_evi temp_sig_sound_number
    clear temp_order_block
    for j = 1:length_session,
        
        temp_order_block(j,1) = stim_order_block(j).matrix;
        
        clear p_all p_evi 
        clear median_stim median_rew
        clear median_stim_evi median_rew_evi
        temp_stim = stim_sound(j).matrix;
        %temp_stim_baseline = stim_baseline(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,~] = size(temp_stim);
        %temp_stim_baseline = repmat(temp_stim_baseline,size_trial,1);
        %temp_stim = temp_stim + temp_stim_baseline;
        
        temp_rew = rew_sound(j).matrix;
        %temp_rew_baseline = rew_baseline(j).matrix;
        temp_rew_task = rew_task(j).matrix;
        [size_trial,size_neuron] = size(temp_rew);
        %temp_rew_baseline = repmat(temp_rew_baseline,size_trial,1);
        %temp_rew = temp_rew + temp_rew_baseline;
        
        %Compare with all activity
        median_stim = median(temp_stim)';
        median_rew  = median(temp_rew)';
        for l = 1:size_neuron,
            p_all(l,1) = ranksum(temp_stim(:,l),temp_rew(:,l));   
        end
        %Each tone cloud
        stim_evi = temp_stim_task(:,4);
        rew_evi  = temp_rew_task(:,4);
        for k = 1:6,
            temp1 = find(stim_evi == k);
            temp2 = find(rew_evi == k);
            for l = 1:size_neuron,
                p_evi(l,k) = ranksum(temp_stim(temp1,l),temp_rew(temp2,l));    
            end
            median_stim_evi(:,k) = median(temp_stim(temp1,:))';
            median_rew_evi(:,k)  = median(temp_rew(temp2,:))';
        end
        all_stim_sound = [all_stim_sound; median_stim];
        all_rew_sound = [all_rew_sound; median_rew];
        all_stim_sound_evi = [all_stim_sound_evi; median_stim_evi];
        all_rew_sound_evi = [all_rew_sound_evi; median_rew_evi];
        all_p_sound = [all_p_sound; p_all];
        all_p_sound_evi = [all_p_sound_evi; p_evi];
        
        %About data2
        %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2];
        temp_stim = data2.stim_box(j).matrix;
        temp_rew  = data2.rew_box(j).matrix;
        temp_sound = [temp_stim(:,2), temp_rew(:,2)];
        all_sig_sound_only = [all_sig_sound_only; temp_sound];

        check_sig = data2.sig_overlap_or(j).matrix;
        check_sig = find(check_sig == 1);
        temp_sig_sound_number(j,1) = length(check_sig);

        %[p_stim(j,1),p_rew(j,1)] = get_p_value_each_session(median_stim,median_rew,temp_sound_neuron,p_all,thre);
        [p_stim(j,1),p_rew(j,1)] = get_p_value_each_session(median_stim,median_rew,check_sig,p_all,thre);

    end   
    order_block = [order_block; temp_order_block];
    all_p_stim = [all_p_stim; p_stim];
    all_p_rew = [all_p_rew; p_rew];
%     all_p_stim_evi = [all_p_stim_evi; p_stim_evi];
%     all_p_rew_evi = [all_p_rew_evi; p_rew_evi];
    all_sig_sound_number = [all_sig_sound_number; temp_sig_sound_number];

    %sig_roi_overlap_matrix = [sig_roi_overlap; sig_roi_overlap_S; sig_roi_overlap_L; sig_roi_overlap_R];
%    all_sig_sound_and =  [all_sig_sound_and; data2.all_sig_sound_and];
     all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
%     all_sig_sound_rew =  [all_sig_sound_rew; data2.all_sig_sound_rew];
%     all_sig_sound_stim = [all_sig_sound_stim; data2.all_sig_sound_stim];

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

[length(all_stim_sound), length(sig_kruskal_both)]

%focus on the activity: Sound neurons
%plot_stim_reward_sound_20230531(thre, all_stim_sound, all_rew_sound, all_p_sound, all_mouse_number, sig_kruskal_both);

%All sig neurons
plot_stim_reward_sound_20230531(thre, all_stim_sound, all_rew_sound, all_p_sound, all_mouse_number, all_sig_sound_or);

%All neurons
%plot_stim_reward_sound(thre, all_stim_sound, all_rew_sound, all_p_sound, [1:length(all_stim_sound)]);


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Each session
% 
% figure
% subplot(1,3,1)
% order0 = find(order_block == 0); %Left -> Right
% order1 = find(order_block == 1);
% plot(all_p_rew(order0), all_p_stim(order0), 'r.');
% hold on
% plot(all_p_rew(order1), all_p_stim(order1), 'b.');
% 
% subplot(1,3,2)
% temp0 = find(order_session == 0); % Stim -> Rew
% temp1 = find(order_session == 1);
% plot(all_p_rew(temp0), all_p_stim(temp0), 'r.');
% hold on
% plot(all_p_rew(temp1), all_p_stim(temp1), 'b.');
% 
% subplot(1,3,3)
% day1 = find(day_session == 1); % Stim -> Rew
% day2 = find(day_session == 2);
% day3 = find(day_session >= 3);
% plot(all_p_rew(day1), all_p_stim(day1), 'r.');
% hold on
% plot(all_p_rew(day2), all_p_stim(day2), 'g.');
% hold on
% plot(all_p_rew(day3), all_p_stim(day3), 'b.');
% 
% figure
% plot(all_p_stim+all_p_rew)
% 
% [median(all_p_stim(temp0)), median(all_p_rew(temp0))]
% [median(all_p_stim(temp1)), median(all_p_rew(temp1))]
% ranksum(all_p_stim(temp0), all_p_stim(temp1))
% ranksum(all_p_rew(temp0), all_p_rew(temp1))
% ranksum(all_p_stim(temp0)-all_p_rew(temp0), all_p_stim(temp1)-all_p_rew(temp1))
% 
% sum(all_sig_sound_number)
% sum(all_p_stim .* all_sig_sound_number)
% sum(all_p_rew .* all_sig_sound_number)

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
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [p_stim,p_rew] = get_p_value_each_session(median_stim,median_rew,temp_sound_neuron,p_all,thre)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        temp_p = p_all(temp_sound_neuron);
        stim_neuron = find(median_stim(temp_sound_neuron) > median_rew(temp_sound_neuron));
        rew_neuron = find(median_stim(temp_sound_neuron) < median_rew(temp_sound_neuron));
        temp_p = find(temp_p < thre);
        p_stim = intersect(temp_p,stim_neuron);
        p_rew  = intersect(temp_p,rew_neuron);
        %all_p_prob = [length(p_stim), length(p_rew)] ./ length(temp_sound_neuron);
        p_stim = length(p_stim) ./ length(temp_sound_neuron);
        p_rew = length(p_rew) ./ length(temp_sound_neuron);
        
        %[p_stim, p_rew]
        return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_stim_reward_sound_20230531(thre, all_stim_sound, all_rew_sound, all_p_sound, mouse_number, sig_neurons)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
all_stim_sound = all_stim_sound(sig_neurons);
all_rew_sound = all_rew_sound(sig_neurons);
all_p_sound = all_p_sound(sig_neurons);
mouse_number = mouse_number(sig_neurons,:);

temp_stim = find(all_stim_sound > all_rew_sound);
temp_rew  = find(all_stim_sound < all_rew_sound);

temp_sig = find(all_p_sound < thre);
temp_sig_stim = intersect(temp_stim, temp_sig);
temp_sig_rew  = intersect(temp_rew, temp_sig);

temp_non_sig = setdiff([1:length(sig_neurons)], temp_sig);

figure
plot(all_rew_sound(temp_non_sig), all_stim_sound(temp_non_sig), 'k.')
hold on
plot(all_rew_sound(temp_sig_rew), all_stim_sound(temp_sig_rew), 'r.')
hold on
plot(all_rew_sound(temp_sig_stim), all_stim_sound(temp_sig_stim), 'b.')
hold on
plot([-1 3.5],[-1 3.5],'k')
set(gca,'xlim',[-1 3.5],'ylim',[-1 3.5])

[length(sig_neurons),length(temp_non_sig),length(temp_sig_stim),length(temp_sig_rew)]
[r,p] = corr(double(all_rew_sound), double(all_stim_sound), 'Type','Spearman')

[r,p] = partialcorr(double(all_rew_sound),double(all_stim_sound),mouse_number(:,1),'Type','Spearman')
[r,p] = partialcorr(double(all_rew_sound),double(all_stim_sound),mouse_number(:,2),'Type','Spearman')

return


