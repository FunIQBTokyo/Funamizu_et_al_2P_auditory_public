%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tuning_curve_200401_block_encode7_with_decoder3_230607
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

default_folder = 'e:/Tone_discri1/all_mouse/sound_20180610';
%analysis_folder = 'stimulus'; %stimulus or reward
analysis_folder = 'reward'; %stimulus or reward

pathname1 = strcat('e:/Tone_discri1/all_mouse/sound_only_20181017/');
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
% [filename2, pathname2,findex]=uigetfile('*.mat','Overlap_file','Multiselect','on');
% filename2

pathname4 = strcat('E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920\keisu_20191120');
cd(pathname4)
filename4 = dir('*.mat');

pathname3 = strcat('e:/Tone_discri1/all_mouse/single_decode/separate4/single_190925/');
cd(pathname3)
filename3 = dir('*.mat');

clear neuron_number
all_length_neuron = [];
all_sig_sound_or = [];

all_median_all = [];
all_median_correct = [];
all_median_error = [];
all_BF_neuron = [];
all_p_RE = [];
all_p_block = [];
all_p_block_correct = [];
all_sabun_block = [];

all_median_block_L = [];
all_median_block_R = [];
all_median_block_L_correct = [];
all_median_block_R_correct = [];
all_neuron_sabun_sound = [];

all_p_kruskal_stim = [];
all_p_prior = [];

all_BF_sound = [];
all_thre_decode = [];
all_thre_sound = [];
all_correct = [];

all_decode_neuron = [];

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, sig_kruskal_sound,...
 moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all
mouse_number = [];
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
    
    %Based on length_neuron, make the mouse session lines
    temp_length_session = ones(length_session,1) * i;
    mouse_number = [mouse_number; temp_length_session];
    
    clear mouse_p_kruskal_stim mouse_BF_stim mouse_sabun_block mouse_std_sound
    clear length_neuron
    for j = 1:length_session,
        temp_stim = stim_sound(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);
        length_neuron(j,1) = size_neuron;
        
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
        clear p_prior sabun_block
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
        %BF is determined with activity during reward
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
            
            %Based on the BF_neuron(2), get the sabun block activity
            if BF_neuron(l,2) < 3.5,
                sabun_block(l,1) = median_block_L(l,BF_neuron(l,2)) - median_block_R(l,BF_neuron(l,2));
            else
                sabun_block(l,1) = median_block_R(l,BF_neuron(l,2)) - median_block_L(l,BF_neuron(l,2));
            end
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
        
        temp_std = std(temp_stim);
        all_sabun_block = [all_sabun_block; sabun_block./temp_std'];
        
        mouse_p_kruskal_stim(j).matrix = p_kruskal_stim;
        mouse_BF_stim(j).matrix = BF_neuron(:,2); %reward only
        mouse_sabun_block(j).matrix = sabun_block;
        mouse_std_sound(j).matrix = std(temp_stim);
    end    
    all_length_neuron = [all_length_neuron; length_neuron];
    
    %Detection of overlap is from overlap mat file
    temp_filename = filename2(i).name 
    temp_path = pathname2;
    fpath = fullfile(temp_path, temp_filename);
    data2 = load(fpath);

    %sig_box = [sig_led, sig_sound, sig_LR, sig_RE, sig_LR2, sig_RE2, sig_sound_ore];
    %Pick only sig_sound
    clear mouse_sig_sound
    length_session = length(data2.stim_box);
    for j = 1:length_session,
        if length(analysis_folder) == 8, %Stimulus
            temp_stim = data2.stim_box(j).matrix;
        elseif length(analysis_folder) == 6, %reward
            temp_stim  = data2.rew_box(j).matrix;
        end
        temp_sound = temp_stim(:,2);
        mouse_sig_sound(j).matrix = temp_sound;
    end
    all_sig_sound_or =   [all_sig_sound_or; data2.all_sig_sound_or];
        
    %Get Population regression
    temp_filename = filename4(i).name 
    temp_path = pathname4;
    fpath = fullfile(temp_path, temp_filename);
    data3 = load(fpath);
    
    temp_filename = filename3(i).name 
    temp_path = pathname3;
    fpath = fullfile(temp_path, temp_filename);
    data4 = load(fpath);

    clear temp_sound_decode_neuron temp_sound_neuron
    clear length_sig_sound session_sabun median_session_thre
    for j = 1:length_session,
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %About decoding
        if length(analysis_folder) == 8, %Stimulus
            temp_regress = data3.stim(j).matrix.b_sound_and;
            %temp_regress = data3.stim(j).matrix.b_sound_or;
            %temp_thre = data4.stim(j).matrix.distri_sound.thre_sound;
            %temp_thre = data4.stim(j).matrix.distri_sound.thre_sound_all;
            temp_correct = data4.stim(j).matrix.distri_sound.correct_rate;
        elseif length(analysis_folder) == 6, %reward
            temp_regress = data3.rew(j).matrix.b_sound_and;
            %temp_regress = data3.rew(j).matrix.b_sound_or;
            %temp_thre = data4.rew(j).matrix.distri_sound.thre_sound;
            %temp_thre = data4.rew(j).matrix.distri_sound.thre_sound_all;
            temp_correct = data4.rew(j).matrix.distri_sound.correct_rate;
        end
        
        %First regression is not for neuron
        %Check with data2.sig_overlap_or(i).matrix
        [~,length_regress] = size(temp_regress);
        check_sig = data2.sig_overlap_or(j).matrix;
        check_sig = find(check_sig == 1);
        if length(check_sig) ~= length_regress,
            [length(check_sig), length_regress]
            hoge
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Integrate the decoding and regression
        temp_decode_neuron = zeros(length(mouse_BF_stim(j).matrix),1);
        temp_regress = min(temp_regress); %Take only the neurons which used in all the 1000 CVs
        use_sig_decode = find(temp_regress ~= 0);
        use_sig_decode = check_sig(use_sig_decode); %decoding neurons, ovelapped neurons number
        %sound_decode_regress = temp_decode_neuron(use_sig_decode,1);
        temp_decode_neuron(use_sig_decode) = 1;
        
        all_decode_neuron = [all_decode_neuron; temp_decode_neuron];
        all_correct = [all_correct; temp_correct(:,1)];
        
    end
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

%BF neuron in sound neurons
cumsum_neuron = cumsum(all_length_neuron);

all_BF_neuron = all_BF_neuron(:,2); %correct trials
sound_BF_neuron = zeros(length(all_BF_neuron),1);
sound_BF_neuron(sig_kruskal_sound) = all_BF_neuron(sig_kruskal_sound);

for i = 1:length(all_BF_neuron),
    all_p_BF(i,1) = all_p_block(i,all_BF_neuron(i));
end
thre = 0.05;
all_p_prior = all_p_BF; %Take the smaller value
% thre = 0.01;
% all_p_prior = min(all_p_prior,[],2); %Take the smaller value

%Sig neurons: all_sig_sound_or

%Decoder neurons: all_decode_neuron
%Sabun activity (Normalized): all_sabun_block
%Correct rate of decoding: all_correct

%Check0
all_length_neuron
size(all_length_neuron)
sum(all_length_neuron)
%Check1
size(sound_BF_neuron)
size(all_sig_sound_or)
size(all_decode_neuron)
size(all_sabun_block)
size(all_correct)

%Check2
temp = find(sound_BF_neuron ~= 0);
[sum(all_sig_sound_or), length(temp), sum(all_decode_neuron)]

%Check3: sound neurons are all in the sig neurons?
temp_sig = find(all_sig_sound_or == 1);
temp_sig = sound_BF_neuron(temp_sig);
temp_sig = find(temp_sig ~= 0);
[length(temp_sig), length(temp)]

%Start analysis
%Difference between the distribution of sound neurons in sig-neurons
%Sound neurons in decoder neurons

for i = 1:length(all_length_neuron),
    if i == 1,
        pre = 1;
    else
        pre = cumsum_neuron(i-1)+1;
    end
    post = cumsum_neuron(i);
    
    %Each session
    temp_sig    = all_sig_sound_or([pre:post]);
    temp_sound  = sound_BF_neuron([pre:post]);
    temp_decode = all_decode_neuron([pre:post]);
    temp_prior = all_p_prior([pre:post]);
    
    %Limit to sig neurons
    temp_sig = find(temp_sig == 1);
    temp_sound = temp_sound(temp_sig);
    temp_decode = temp_decode(temp_sig);
    temp_prior = temp_prior(temp_sig);
    
    %Sound neurons in decoder
    temp_sound  = find(temp_sound ~= 0);
    temp_decode = find(temp_decode ~= 0);
    temp_sound_decode = intersect(temp_sound, temp_decode);
    neuron_sound_decode(i,:) = [length(temp_sound_decode),length(temp_decode),length(temp_sound),length(temp_sig)];
    
    %Prior modulation in decoder
    temp_prior = find(temp_prior < thre);
    temp_sound_prior = intersect(temp_sound, temp_prior);
    temp_sound_prior_decode = intersect(temp_sound_decode, temp_prior);
    neuron_sound_prior(i,:) = [length(temp_sound_prior_decode), length(temp_sound_decode), length(temp_sound_prior), length(temp_sound)];
end
sum(neuron_sound_decode)
sum(neuron_sound_prior)

%Sound neurons or non-sound neurons in decoder
sound_in_decode(:,1) = neuron_sound_decode(:,1) ./ neuron_sound_decode(:,2);
sound_in_decode(:,2) = neuron_sound_decode(:,3) ./ neuron_sound_decode(:,4);
%Modulation neurons or non-modulated neurons in decoder
prior_sound_in_decode(:,1) = neuron_sound_prior(:,1) ./ neuron_sound_prior(:,2);
prior_sound_in_decode(:,2) = neuron_sound_prior(:,3) ./ neuron_sound_prior(:,4);

figure
subplot(1,2,1) %Sound in decoder
boxplot(sound_in_decode)
hold on
plot(sound_in_decode')
subplot(1,2,2) %Prior Sound in decoder
boxplot(prior_sound_in_decode)
hold on
plot(prior_sound_in_decode')

figure
subplot(1,2,1)
plot(sound_in_decode(:,2), sound_in_decode(:,1),'k.')
hold on
plot([0 1], [0 1],'k')
set(gca,'xlim',[0 1],'ylim',[0 1])
subplot(1,2,2)
plot(prior_sound_in_decode(:,2), prior_sound_in_decode(:,1),'k.')
hold on
plot([0 1], [0 1],'k')
set(gca,'xlim',[0 1],'ylim',[0 1])

%Kentei
signrank(sound_in_decode(:,1), sound_in_decode(:,2))
signrank(prior_sound_in_decode(:,1), prior_sound_in_decode(:,2))

%LME
lme = fitlme_analysis_20210520_0(sound_in_decode(:,1)-sound_in_decode(:,2),mouse_number);
lme(2).lme
lme = fitlme_analysis_20210520_0(prior_sound_in_decode(:,1)-prior_sound_in_decode(:,2),mouse_number);
lme(2).lme
