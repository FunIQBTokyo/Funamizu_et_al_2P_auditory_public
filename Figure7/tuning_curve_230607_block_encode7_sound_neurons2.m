%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [all_sig_sound,all_sig_sound_L,all_sig_sound_R,block_L,block_R] = analysis_pipeline_all_sound1_180208_ROC_RE_nan_new
function tuning_curve_230607_block_encode7_sound_neurons2
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

pathname4 = strcat('E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920\keisu_20191120');
cd(pathname4)
filename4 = dir('*.mat');

pathname3 = strcat('e:/Tone_discri1/all_mouse/single_decode/separate4/single_210907/');
cd(pathname3)
filename3 = dir('*.mat');
% correct_rate = [correct_rate_mix, correct_rate_easy, correct_rate_mid, correct_rate_dif, temp_correct2];
% correct_rate2 = [mix_L0, mix_L1, mix_R0, mix_R1];

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
all_sabun_block_BF = [];

all_median_block_L = [];
all_median_block_R = [];
all_median_block_L_correct = [];
all_median_block_R_correct = [];

all_p_kruskal_stim = [];
all_p_prior = [];

all_thre_decode = [];
all_correct = [];

all_decode_neuron = [];

%Get the number of neurons for sound neurons
[moto_sig_kruskal, moto_sig_kruskal_stim, moto_sig_kruskal_rew, sig_kruskal_sound,...
 moto_sig_sound_timing, check_neuron_number, all_mouse_number] = ...
                tuning_curve_230531_sound_only_compare_block4_prefer;
close all

for i = 1 : length(filename1)
    clear data data2 data3 data4 data5 temp_filename temp_pass fpath
    
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
        clear p_prior sabun_block sabun_block_BF
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
                sabun_block_BF(l,1) = median_block_L(l,BF_neuron(l,2)) - median_block_R(l,BF_neuron(l,2));
            else
                sabun_block_BF(l,1) = median_block_R(l,BF_neuron(l,2)) - median_block_L(l,BF_neuron(l,2));
            end
            sabun_block(l,1) = median(temp_stim(block_category1L,l)) - median(temp_stim(block_category2L,l));
            sabun_block(l,2) = median(temp_stim(block_category1R,l)) - median(temp_stim(block_category2R,l));
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
        temp_std = repmat(temp_std',1,2);
        all_sabun_block = [all_sabun_block; sabun_block./temp_std];
        all_sabun_block_BF = [all_sabun_block_BF; sabun_block_BF./temp_std];
        
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
            temp_thre = data4.stim(j).matrix.distri_sound.thre_sound_all;
            temp_correct = data4.stim(j).matrix.distri_sound.max_correct;
        elseif length(analysis_folder) == 6, %reward
            temp_regress = data3.rew(j).matrix.b_sound_and;
            %temp_regress = data3.rew(j).matrix.b_sound_or;
            %temp_thre = data4.rew(j).matrix.distri_sound.thre_sound;
            temp_thre = data4.rew(j).matrix.distri_sound.thre_sound_all;
            temp_correct = data4.rew(j).matrix.distri_sound.max_correct;
        end
        
        %About threshold analysis
        temp_std = mouse_std_sound(j).matrix;
        temp_std = repmat(temp_std',1,3);
        temp_thre = temp_thre ./ temp_std; %SD base, normalized
        all_thre_decode = [all_thre_decode; temp_thre];

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
        all_correct = [all_correct; temp_correct]; %correct rate for block_L, block_R, all_with_one_thre, all_with_two_thre
    end
end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

thre = 0.01;
[all_p_prior,temp_I] = min(all_p_prior,[],2); %Take the smaller value
all_sabun_block = abs(all_sabun_block);
all_sabun_block = all_sabun_block(:,temp_I); %Align to the p-value

%BF neuron in sound neurons
cumsum_neuron = cumsum(all_length_neuron);

all_BF_neuron = all_BF_neuron(:,2); %correct trials
sound_BF_neuron = zeros(length(all_BF_neuron),1);
sound_BF_neuron(sig_kruskal_sound) = all_BF_neuron(sig_kruskal_sound);

%Sig neurons: all_sig_sound_or

%Decoder neurons: all_decode_neuron
%Sabun activity (Normalized): all_sabun_block
%Correct rate of decoding: all_correct

%Decode thre
all_thre_decode = abs(all_thre_decode(:,1) - all_thre_decode(:,2));

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
size(all_thre_decode)

%Check2
temp = find(sound_BF_neuron ~= 0);
[sum(all_sig_sound_or), length(temp), sum(all_decode_neuron)]

%Check3: sound neurons are all in the sig neurons?
temp_sig = find(all_sig_sound_or == 1);
temp_sig = sound_BF_neuron(temp_sig);
temp_sig = find(temp_sig ~= 0);
[length(temp_sig), length(temp)]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Change the sabun block to sabun BF
all_sabun_block = all_sabun_block_BF;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Check the number of neurons 
if length(all_sabun_block) ~= length(all_mouse_number)
    hoge
end

%Focus on the sound neurons
%Start analysis
%threshold in decoding neurons
sound_neuron = find(sound_BF_neuron ~= 0);
thre_sound = all_thre_decode(sound_neuron);
decode_sound = all_correct(sound_neuron,:);
sabun_sound = abs(all_sabun_block(sound_neuron));
sound_neuron_in_decoder = all_decode_neuron(sound_neuron);
sound_p_prior = all_p_prior(sound_neuron);

all_mouse_number = all_mouse_number(sound_neuron,:);
%%%%%%%%%%%%%%%%%%%%%%%%
%First check about the sabun and p_prior
sig_p_sabun = find(sound_p_prior < thre);
temp = [1:length(thre_sound)];
non_sig_p_sabun = setdiff(temp, sig_p_sabun);
temp_x = [zeros(length(sig_p_sabun),1);ones(length(non_sig_p_sabun),1)];
figure
boxplot([sabun_sound(sig_p_sabun); sabun_sound(non_sig_p_sabun)], temp_x)

[length(sig_p_sabun), length(non_sig_p_sabun)]
[median(sabun_sound(sig_p_sabun)), median(sabun_sound(non_sig_p_sabun))]
ranksum(sabun_sound(sig_p_sabun), sabun_sound(non_sig_p_sabun))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Check the decode sound
figure
plot(decode_sound(:,3),decode_sound(:,4),'b.')
hold on
plot([0.5 1],[0.5 1],'k')
constant_decode_sound = double(decode_sound(:,3));
decode_sound = decode_sound(:,4)-decode_sound(:,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fitting dy definied the moving bins

decode_sound = double(decode_sound);
sabun_sound = double(sabun_sound);
thre_sound = double(thre_sound);

SN_in_decoder = find(sound_neuron_in_decoder == 1);
all_neuron = [1:length(sabun_sound)];
SN_not_decoder = setdiff(all_neuron,SN_in_decoder);

median(sabun_sound - thre_sound) %block modulation - thre change
p_sabun_thre = signrank(sabun_sound, thre_sound)
p_sabun_thre_in = signrank(sabun_sound(SN_in_decoder), thre_sound(SN_in_decoder))
p_sabun_thre_out = signrank(sabun_sound(SN_not_decoder), thre_sound(SN_not_decoder))

data = [sabun_sound-thre_sound];
[lme,AIC_model,BIC_model,p_AIC_BIC(1,:)] = fitlme_analysis_20210520_1([data,all_mouse_number]);
data = [sabun_sound(SN_in_decoder)-thre_sound(SN_in_decoder)];
[lme,AIC_model,BIC_model,p_AIC_BIC(2,:)] = fitlme_analysis_20210520_1([data,all_mouse_number(SN_in_decoder,:)]);
data = [sabun_sound(SN_not_decoder)-thre_sound(SN_not_decoder)];
[lme,AIC_model,BIC_model,p_AIC_BIC(3,:)] = fitlme_analysis_20210520_1([data,all_mouse_number(SN_not_decoder,:)]);
p_AIC_BIC
p_AIC_BIC(1,2)
p_AIC_BIC(2,2)
p_AIC_BIC(3,2)

[length(all_neuron), length(SN_in_decoder), length(SN_not_decoder)]
%thre_sound, sabun_sound
figure
subplot(1,3,1)
[rho(1,:),pval(1,:)] = make_linear_robust_fit_20230523(thre_sound, sabun_sound,all_neuron,all_mouse_number,[-0.1 3.5],[-0.1 3.5]);
subplot(1,3,2)
[rho(2,:),pval(2,:)] = make_linear_robust_fit_20230523(thre_sound, sabun_sound,SN_in_decoder,all_mouse_number,[-0.1 3.5],[-0.1 3.5]);
subplot(1,3,3)
[rho(3,:),pval(3,:)] = make_linear_robust_fit_20230523(thre_sound, sabun_sound,SN_not_decoder,all_mouse_number,[-0.1 3.5],[-0.1 3.5]);
rho
pval
pval(1,1)
pval(1,2)
pval(1,3)
pval(2,1)
pval(2,2)
pval(2,3)
pval(3,1)
pval(3,2)
pval(3,3)

%decode_sound, sabun_sound
figure
subplot(1,3,1)
[rho2(1,:),pval2(1,:)] = make_linear_robust_fit_20230523(decode_sound, sabun_sound,all_neuron,all_mouse_number,[-0.001 0.06],[-0.1 3.5]);
subplot(1,3,2)
[rho2(2,:),pval2(2,:)] = make_linear_robust_fit_20230523(decode_sound, sabun_sound,SN_in_decoder,all_mouse_number,[-0.001 0.06],[-0.1 3.5]);
subplot(1,3,3)
[rho2(3,:),pval2(3,:)] = make_linear_robust_fit_20230523(decode_sound, sabun_sound,SN_not_decoder,all_mouse_number,[-0.001 0.06],[-0.1 3.5]);
rho2
pval2
pval2(1,1)
pval2(1,2)
pval2(1,3)
pval2(2,1)
pval2(2,2)
pval2(2,3)
pval2(3,1)
pval2(3,2)
pval2(3,3)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Check the relationship between single correct rate and activity change
%all_correct: 1:left, 2:right or 5:left_single, 6:right_single
prefer_L = 5;
prefer_R = 6;

L_neuron = find(sound_BF_neuron(sound_neuron) < 3.5);
H_neuron = find(sound_BF_neuron(sound_neuron) > 3.5);
%sound_neuron = find(sound_BF_neuron ~= 0);
sabun_sound = double(all_sabun_block(sound_neuron));
%Reset decode sound
decode_sound = double(all_correct(sound_neuron,:));
decode_sound_LR = decode_sound(:,prefer_L) - decode_sound(:,prefer_R); %L - R
decode_sound_LR(H_neuron) = decode_sound(H_neuron,prefer_R) - decode_sound(H_neuron,prefer_L); %L - R

%Put the regression analysis result
%set_x_axis = [-2.5 1.5];
set_x_axis = [-1.5 1.5];
set_y_axis = [-0.25 0.25];

figure
subplot(1,3,1)
[rho3(1,:),pval3(1,:)] = make_linear_robust_fit_20230523(sabun_sound, decode_sound_LR,all_neuron,all_mouse_number,set_x_axis,set_y_axis);
subplot(1,3,2)
[rho3(2,:),pval3(2,:)] = make_linear_robust_fit_20230523(sabun_sound, decode_sound_LR,SN_in_decoder,all_mouse_number,set_x_axis,set_y_axis);
subplot(1,3,3)
[rho3(3,:),pval3(3,:)] = make_linear_robust_fit_20230523(sabun_sound, decode_sound_LR,SN_not_decoder,all_mouse_number,set_x_axis,set_y_axis);
rho3
pval3
pval3(1,1)
pval3(1,2)
pval3(1,3)
pval3(2,1)
pval3(2,2)
pval3(2,3)
pval3(3,1)
pval3(3,2)
pval3(3,3)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function make_linear_robust_fit(sabun_sound, decode_sound,SN_in_decoder,temp_x)
function [rho_not,pval_not] = make_linear_robust_fit_20230523(sabun_sound, decode_sound,SN_in_decoder,all_mouse_number,x_val,y_val)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mdl = fitlm(sabun_sound(SN_in_decoder), decode_sound(SN_in_decoder),'linear','RobustOpts','on');
b = mdl.Coefficients.Estimate;
temp_p = mdl.Coefficients.pValue;
beta(2) = b(2);
stats_p(2) = temp_p(2);

%figure
%subplot(1,2,1)
plot(sabun_sound(SN_in_decoder), decode_sound(SN_in_decoder), '.','color',[0 0 0])
hold on

temp_step = (max(sabun_sound) - min(sabun_sound)) / 20;
temp_x = [min(sabun_sound) : temp_step : max(sabun_sound)];

%subplot(1,2,2)
[Ypred,YCI] = predict(mdl, temp_x');
x_fill = [temp_x, fliplr(temp_x)];
y_fill = [YCI(:,1)', fliplr(YCI(:,2)')];
hold on
plot(temp_x', Ypred,'r')
% plot(temp_x', Ypred,'k')
% hold on
% fill(x_fill, y_fill,[0.5 0.5 0.5],'edgecolor','none','FaceAlpha',0.3)
set(gca,'xlim',x_val,'ylim',y_val)

[rho_not(1),pval_not(1)] = corr(sabun_sound(SN_in_decoder), decode_sound(SN_in_decoder),'Type','Spearman');
[rho_not(2),pval_not(2)] = partialcorr(sabun_sound(SN_in_decoder), decode_sound(SN_in_decoder),all_mouse_number(SN_in_decoder,1),'Type','Spearman');
[rho_not(3),pval_not(3)] = partialcorr(sabun_sound(SN_in_decoder), decode_sound(SN_in_decoder),all_mouse_number(SN_in_decoder,2),'Type','Spearman');

return

