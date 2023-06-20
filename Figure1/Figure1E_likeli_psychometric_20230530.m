%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Figure1E_likeli_psychometric_20230530
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

current_folder = pwd;

stimulus_folder = 'E:\Tone_discri1\all_mouse_behavior\regress_results\block5';
reward_folder = 'E:\Tone_discri1\all_mouse_behavior\regress_results\reward5';

%Number of parameter for 1 to 7:
%1 2 3 3 4 3 4

disp('stimulus task')
log_likeli_model_analysis(stimulus_folder,current_folder);
disp('reward task')
log_likeli_model_analysis(reward_folder,current_folder);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function log_likeli_model_analysis(stimulus_folder,current_folder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(stimulus_folder)
filename1 = dir('*.mat');
%filename1
    
all_likeli = [];
for i = 1 : length(filename1)
    temp_filename = filename1(i).name; 
    fpath = fullfile(stimulus_folder, temp_filename);
    data = load(fpath);
    temp = data.log_likeli_all;
    temp = temp(:,[2:7]); %2 3 3 4 3 4
    
    [mouse_session(i),~] = size(temp);
    all_likeli = [all_likeli; temp];
    
    ave_likeli(i,:) = mean(temp);
end

cd(current_folder)

size(all_likeli)
size(ave_likeli)

%Based on average log likeli per session
disp('average per session')
plot_log_likeli_analysis(all_likeli)

%Based on average log likeli per mouse
disp('average per mouse')
plot_log_likeli_analysis(ave_likeli)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_log_likeli_analysis(ave_likeli)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_stim1 = mean(ave_likeli);

test_stim1 = mean_stim1(3) - mean_stim1(1); %parameter 3 - 2
test_stim2 = mean_stim1(4) - mean_stim1(3); %parameter 4 - 3
test_stim3 = mean_stim1(6) - mean_stim1(3); %parameter 4 - 3

1 - chi2cdf(2*test_stim1,1)
1 - chi2cdf(2*test_stim2,1)
1 - chi2cdf(2*test_stim3,1)

sabun_stim(:,1) = ave_likeli(:,1) - ave_likeli(:,1);
sabun_stim(:,2) = ave_likeli(:,3) - ave_likeli(:,1);
sabun_stim(:,3) = ave_likeli(:,4) - ave_likeli(:,3);
sabun_stim(:,4) = ave_likeli(:,6) - ave_likeli(:,3);

figure
subplot(1,3,1)
boxplot(ave_likeli)

%plot dot
rand_dot = (rand(length(sabun_stim),1) - 0.5) .* 0.2;

%plot line
subplot(1,3,2)
boxplot(sabun_stim)
hold on
plot(sabun_stim')

subplot(1,3,3)
boxplot(sabun_stim)
for i = 1:4,
    hold on
    plot(i+rand_dot,sabun_stim(:,i),'b.')
end

return

