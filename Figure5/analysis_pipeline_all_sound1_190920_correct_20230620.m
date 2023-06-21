%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function analysis_pipeline_all_sound1_190920_correct_20230620
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currentFolder = pwd;

analysis_folder = 'stimulus'; %stimulus or reward
%analysis_folder = 'reward'; %stimulus or reward

pathname1 = 'E:\Tone_discri1\all_mouse\single_decode\separate4\population_190920';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

pathname3 = strcat('e:/Tone_discri1/all_mouse/behave_data_20190731');
cd(pathname3)
filename3 = dir('*.mat');

all_correct_random = [];
all_correct_CV = [];

max_neuron = 140;
all_correct_rate1 = [];
all_correct_easy = [];
all_correct_mid = [];
all_correct_dif = [];
all_r_original = [];
all_r_shuffle = [];

S_correct_rate1 = [];
S_correct_easy = [];
S_correct_mid = [];
S_correct_dif = [];
S_r_original = [];
S_r_shuffle = [];

L_correct_rate1 = [];
L_correct_easy = [];
L_correct_mid = [];
L_correct_dif = [];
L_r_original = [];
L_r_shuffle = [];
for i = 1 : length(filename1)
    clear data data2 data3 data4 temp_filename temp_pass fpath

    temp_filename = filename1(i) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    load(fpath);
    if length(analysis_folder) == 8,
        use_stim = stim;
        use_S = stim_S;
        use_L = stim_L;
    else
        use_stim = rew;
        use_S = rew_S;
        use_L = rew_L;
    end
    [all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle] ...
        = update_correct_rate(use_stim,all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle,max_neuron);
    [S_correct_rate1,S_correct_easy,S_correct_mid,S_correct_dif,S_r_original,S_r_shuffle] ...
        = update_correct_rate(use_S,S_correct_rate1,S_correct_easy,S_correct_mid,S_correct_dif,S_r_original,S_r_shuffle,max_neuron);
    [L_correct_rate1,L_correct_easy,L_correct_mid,L_correct_dif,L_r_original,L_r_shuffle] ...
        = update_correct_rate(use_L,L_correct_rate1,L_correct_easy,L_correct_mid,L_correct_dif,L_r_original,L_r_shuffle,max_neuron);

end

currentFolder = ['cd ',currentFolder];
eval(currentFolder); %move directory

test = S_r_original == L_r_original;
test = find(test == 0);
if length(test) ~= 0,
    hoge
end
figure
plot([S_r_original, S_r_shuffle, L_r_shuffle]')
hold on
boxplot([S_r_original, S_r_shuffle, L_r_shuffle])
[median(S_r_original),median(S_r_shuffle),median(L_r_shuffle)]

plot_correct_rate(all_correct_rate1,S_correct_rate1,L_correct_rate1)
plot_correct_rate(all_correct_easy,S_correct_easy,L_correct_easy)
plot_correct_rate(all_correct_mid,S_correct_mid,L_correct_mid)
plot_correct_rate(all_correct_dif,S_correct_dif,L_correct_dif)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_correct_rate(all_correct_rate1,S_correct_rate1,L_correct_rate1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(1,3,1)
%plot(nanmean(all_correct_rate1),'k')
plot_mean_se_moto(all_correct_rate1,[0 0 0],2)
hold on
%plot(nanmean(S_correct_rate1),'r')
plot_mean_se_moto(S_correct_rate1,[1 0 0],2)
hold on
%plot(nanmean(L_correct_rate1),'b')
plot_mean_se_moto(L_correct_rate1,[0 0 1],2)
set(gca,'xlim',[0 140],'ylim',[0.5 1])
subplot(1,3,2)
%plot(nanmean(all_correct_rate1),'k')
plot_mean_se_moto(all_correct_rate1,[0.5 0.5 0.5],2)
hold on
%plot(nanmean(S_correct_rate1),'r')
plot_mean_se_moto(S_correct_rate1,[0 0 0],2)
set(gca,'xlim',[0 140],'ylim',[0.5 1])


%Add some analysis
%Focus on the all_correct_rate1
[size_session,~] = size(all_correct_rate1);

for i = 1:size_session
    temp_correct = all_correct_rate1(i,:);
    %size(temp_correct)
    max_correct = max(temp_correct);
    correct95 = 0.95 * max_correct;
    
    temp = find(temp_correct >= max_correct,1);
    temp95 = find(temp_correct >= correct95,1);
    
    max_N_number(i,:) = [temp95, temp];
end
%max_N_number
median_N = median(max_N_number)

plot_median = 0.2 * median_N + 0.5;
%hist_count = [0:2:20,inf];
hist_count = [0:5:140,inf];
hist_max = histcounts(max_N_number(:,2),hist_count);
hist95 = histcounts(max_N_number(:,1),hist_count);

subplot(1,3,3)
plot(hist_max,'color',[0.5 0.5 0.5])
hold on
plot(hist95,'color',[0 0 0])
hold on
plot([plot_median(1),plot_median(1)],[0 10],'color',[0 0 0])
hold on
plot([plot_median(2),plot_median(2)],[0 10],'color',[0.5 0.5 0.5])
set(gca,'ylim',[0 45])
%bar(hist95)

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle] ...
     = update_correct_rate(stim,all_correct_rate1,all_correct_easy,all_correct_mid,all_correct_dif,all_r_original,all_r_shuffle,max_neuron)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clear correct_rate1 correct_easy correct_mid correct_dif correct_rate2
    clear correct_L0 correct_L1 correct_R0 correct_R1
    clear reward_all reward_easy reward_mid reward_dif
    clear r_original r_shuffle
    length_session = length(stim); 
    %Sound choice integrate
%     correct_rate(count,:) = [temp_correct1, temp_easy, temp_mid, temp_dif, temp_correct2];
%     correct_rate2(count,:) = [mix_L0, mix_L1, mix_R0, mix_R1];
%     reward_rate(count,:) = get_4_reward_rate(sound_mix(:,temp_mix), Sound, block_L, block_R, Sound_Evi, block_rew);
    r_original = [];
    r_shuffle = [];
    for j = 1:length_session,
            temp_correct1  = stim(j).matrix.correct1;
            temp_correct2  = stim(j).matrix.correct2;
            temp_reward_rate1  = stim(j).matrix.reward_rate1;
            
            if isfield(stim(j).matrix,'r_original'),
                r_original(j,1) = median(stim(j).matrix.r_original);
                r_shuffle(j,1)  = median(stim(j).matrix.r_shuffle);
            end
        %%%%
        correct_rate1(j,:) = align_correct_rate(temp_correct1(:,1)', max_neuron);
        correct_easy(j,:) =  align_correct_rate(temp_correct1(:,2)', max_neuron);
        correct_mid(j,:) =   align_correct_rate(temp_correct1(:,3)', max_neuron);
        correct_dif(j,:) =   align_correct_rate(temp_correct1(:,4)', max_neuron);
        correct_rate2(j,:) = align_correct_rate(temp_correct1(:,5)', max_neuron);
        
        correct_L0(j,:) = align_correct_rate(temp_correct2(:,1)', max_neuron);
        correct_L1(j,:) = align_correct_rate(temp_correct2(:,2)', max_neuron);
        correct_R0(j,:) = align_correct_rate(temp_correct2(:,3)', max_neuron);
        correct_R1(j,:) = align_correct_rate(temp_correct2(:,4)', max_neuron);

        reward_all(j,:) = align_correct_rate(temp_reward_rate1(:,1)', max_neuron);
        reward_easy(j,:) = align_correct_rate(temp_reward_rate1(:,2)', max_neuron);
        reward_mid(j,:) = align_correct_rate(temp_reward_rate1(:,3)', max_neuron);
        reward_dif(j,:) = align_correct_rate(temp_reward_rate1(:,4)', max_neuron);
    end
    all_correct_rate1 = [all_correct_rate1; correct_rate1];
    all_correct_easy = [all_correct_easy; correct_easy];
    all_correct_mid = [all_correct_mid; correct_mid];
    all_correct_dif = [all_correct_dif; correct_dif];

    all_r_original = [all_r_original; r_original];
    all_r_shuffle = [all_r_shuffle; r_shuffle];

    return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function correct_rate_max = align_correct_rate(correct_rate, max_neuron)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        size_neuron = length(correct_rate); 
        size_neuron2 = min(size_neuron, max_neuron);
        correct_rate_max = nan(1,max_neuron);
        correct_rate_max(1,[1:size_neuron2]) = correct_rate(1,[1:size_neuron2]);

        return
        