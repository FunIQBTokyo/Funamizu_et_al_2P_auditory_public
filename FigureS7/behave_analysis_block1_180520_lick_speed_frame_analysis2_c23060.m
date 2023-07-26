%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
%Sparse Logistic regression to analyze sound frequency
----------------------------------------------------------------------------
%}
function behave_analysis_block1_180520_lick_speed_frame_analysis2_c23060
currentFolder = pwd;

% [filename1, pathname1,findex]=uigetfile('*.mat','behave_Decode_file','Multiselect','on');
% filename1
%Stimulus
pathname1 = 'E:\Tone_discri1\all_mouse\behave_decode_180520\reward_time\stimulus';
%Reward
%pathname1 = 'E:\Tone_discri1\all_mouse\behave_decode_180520\reward_time\reward';
cd(pathname1)
temp = dir('*.mat');
cd(currentFolder)
for i = 1:length(temp)
    filename1{i} = temp(i).name;
end

all_speed01 = [];
all_speed11 = [];
all_speed00 = [];
all_speed10 = [];
all_lick_L01 = [];
all_lick_L00 = [];
all_lick_L11 = [];
all_lick_L10 = [];
all_lick_C01 = [];
all_lick_C00 = [];
all_lick_C11 = [];
all_lick_C10 = [];
all_lick_R01 = [];
all_lick_R00 = [];
all_lick_R11 = [];
all_lick_R10 = [];

for i = 1:6,
    all_frame(i).matrix = [];
    all_frame0(i).matrix = [];
    all_frame1(i).matrix = [];
end
hist_x = [1:178]; %frame for each event

for filename = 1 : length(filename1)
    clear data temp_filename temp_pass fpath
    
    temp_filename = filename1(filename) 
    temp_filename = cell2mat(temp_filename);
    temp_path = pathname1;
    fpath = fullfile(temp_path, temp_filename);
    data = load(fpath);
    %sound choice reward Sound_Evi 
    %Block_L Block_R frame_speed frame_lick_L frame_lick_C frame_lick_R
    
    number_session = length(data.sound);

    clear mean_speed01 mean_speed11 mean_speed10 mean_speed00
    clear mean_lick_L01 mean_lick_L11 mean_lick_L10 mean_lick_L00
    clear mean_lick_C01 mean_lick_C11 mean_lick_C10 mean_lick_C00
    clear mean_lick_R01 mean_lick_R11 mean_lick_R10 mean_lick_R00
    for i = 1:number_session,
        sound = data.Sound_Evi(i).matrix;
        choice = data.choice(i).matrix;
        reward = data.reward(i).matrix;
        Block_L = data.Block_L(i).matrix;
        Block_R = data.Block_R(i).matrix;
    
        frame_speed = data.frame_speed(i).matrix;
        frame_lick_L = data.frame_lick_L(i).matrix;
        frame_lick_C = data.frame_lick_C(i).matrix;
        frame_lick_R = data.frame_lick_R(i).matrix;
        frame_all = data.frame_all_start(i).matrix;
        
        Block = [Block_L; Block_R];
        Block = sort(Block);
        sound = sound(Block);
        choice = choice(Block);
        reward = reward(Block);
        frame_speed = frame_speed(Block,:);
        frame_lick_L = frame_lick_L(Block,:);
        frame_lick_C = frame_lick_C(Block,:);
        frame_lick_R = frame_lick_R(Block,:);
        frame_all = frame_all(Block,:);
        
        temp_reward1 = find(reward == 1);
        temp_reward0 = find(reward == 0);
        temp_choice1 = find(choice == 1);
        temp_choice0 = find(choice == 0);
        temp_sound1  = find(sound > 3.5);
        temp_sound0  = find(sound < 3.5);
                
        temp_sound0_reward1 = intersect(temp_sound0, temp_reward1);
        temp_sound0_reward0 = intersect(temp_sound0, temp_reward0);
        temp_sound1_reward1 = intersect(temp_sound1, temp_reward1);
        temp_sound1_reward0 = intersect(temp_sound1, temp_reward0);
        
        mean_speed01(i,:) = mean(frame_speed(temp_sound0_reward1,:));
        mean_speed00(i,:) = mean(frame_speed(temp_sound0_reward0,:));
        mean_speed11(i,:) = mean(frame_speed(temp_sound1_reward1,:));
        mean_speed10(i,:) = mean(frame_speed(temp_sound1_reward0,:));
        
        mean_lick_L01(i,:) = mean(frame_lick_L(temp_sound0_reward1,:));
        mean_lick_L00(i,:) = mean(frame_lick_L(temp_sound0_reward0,:));
        mean_lick_L11(i,:) = mean(frame_lick_L(temp_sound1_reward1,:));
        mean_lick_L10(i,:) = mean(frame_lick_L(temp_sound1_reward0,:));

        mean_lick_C01(i,:) = mean(frame_lick_C(temp_sound0_reward1,:));
        mean_lick_C00(i,:) = mean(frame_lick_C(temp_sound0_reward0,:));
        mean_lick_C11(i,:) = mean(frame_lick_C(temp_sound1_reward1,:));
        mean_lick_C10(i,:) = mean(frame_lick_C(temp_sound1_reward0,:));
        
        mean_lick_R01(i,:) = mean(frame_lick_R(temp_sound0_reward1,:));
        mean_lick_R00(i,:) = mean(frame_lick_R(temp_sound0_reward0,:));
        mean_lick_R11(i,:) = mean(frame_lick_R(temp_sound1_reward1,:));
        mean_lick_R10(i,:) = mean(frame_lick_R(temp_sound1_reward0,:));
        
        all_frame = get_hist_event_all(frame_all, hist_x,all_frame);
        [all_frame0, all_frame1] = get_hist_event(frame_all, temp_sound0,temp_sound1,hist_x,all_frame0,all_frame1);
    end
    all_speed01 = [all_speed01; mean_speed01];
    all_speed00 = [all_speed00; mean_speed00];
    all_speed11 = [all_speed11; mean_speed11];
    all_speed10 = [all_speed10; mean_speed10];

    all_lick_L01 = [all_lick_L01; mean_lick_L01];
    all_lick_L00 = [all_lick_L00; mean_lick_L00];
    all_lick_L11 = [all_lick_L11; mean_lick_L11];
    all_lick_L10 = [all_lick_L10; mean_lick_L10];

    all_lick_C01 = [all_lick_C01; mean_lick_C01];
    all_lick_C00 = [all_lick_C00; mean_lick_C00];
    all_lick_C11 = [all_lick_C11; mean_lick_C11];
    all_lick_C10 = [all_lick_C10; mean_lick_C10];

    all_lick_R01 = [all_lick_R01; mean_lick_R01];
    all_lick_R00 = [all_lick_R00; mean_lick_R00];
    all_lick_R11 = [all_lick_R11; mean_lick_R11];
    all_lick_R10 = [all_lick_R10; mean_lick_R10];
end

[size_frame,size_time] = size(all_lick_L01)

%%%%%C_reward is frame 81
%%%%%sound_start is frame 55
%C_reward is frame 82
%sound_start is frame 56

figure
%reward
subplot(3,1,1)
plot_mean_se_moto(all_lick_L01,[0 0 1],2)
hold on
plot_mean_se_moto(all_lick_C01,[0 0 0],2)
hold on
plot_mean_se_moto(all_lick_R01,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,2)
plot_mean_se_moto(all_lick_L11,[0 0 1],2)
hold on
plot_mean_se_moto(all_lick_C11,[0 0 0],2)
hold on
plot_mean_se_moto(all_lick_R11,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,3)
plot_mean_se_moto(all_speed01,[0 0 1],2)
hold on
plot_mean_se_moto(all_speed11,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
%error
figure
subplot(3,1,1)
plot_mean_se_moto(all_lick_L00,[0 0 1],2)
hold on
plot_mean_se_moto(all_lick_C00,[0 0 0],2)
hold on
plot_mean_se_moto(all_lick_R00,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,2)
plot_mean_se_moto(all_lick_L10,[0 0 1],2)
hold on
plot_mean_se_moto(all_lick_C10,[0 0 0],2)
hold on
plot_mean_se_moto(all_lick_R10,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,3)
plot_mean_se_moto(all_speed00,[0 0 1],2)
hold on
plot_mean_se_moto(all_speed10,[1 0 0],2)
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])

[size_frame,size_time] = size(all_frame(1).matrix)

use_color = winter(6);
figure
subplot(3,1,1)
for i = 1:6,
    plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),2);
    hold on
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,2)
for i = 1:6,
    plot_mean_se_moto(all_frame0(i).matrix,use_color(i,:),2);
    hold on
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])
subplot(3,1,3)
for i = 1:6,
    plot_mean_se_moto(all_frame1(i).matrix,use_color(i,:),2);
    hold on
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])

use_color = winter(5);

%Sound on off
figure
for i = 1:5,
    if i ~= 2,
        %plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),2);
        plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),0);
        hold on
    end
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])

%Choice plot
figure
for i = 1:5,
    if i ~= 4,
%        plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),2);
        plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),0);
        hold on
    end
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])

%Reward plot
figure
for i = 1:5,
    if i ~= 5,
%        plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),2);
        plot_mean_se_moto(all_frame(i).matrix,use_color(i,:),0);
        hold on
    end
end
set(gca,'xlim',[1 size_time])
set(gca,'xtick',[1:9:size_time])

%Frame 27 is the sound offset
%Frame time 82 is the sound offset

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function all_frame0 = get_hist_event_all(frame_all, hist_x,all_frame0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_y,size_x] = size(frame_all);

for i = 1:size_x,
    temp0 = hist(frame_all(:,i),hist_x);
    temp0 = temp0 ./ size_y;
    
    all_frame0(i).matrix = [all_frame0(i).matrix; temp0];
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [all_frame0, all_frame1] = get_hist_event(frame_all, temp_sound0,temp_sound1,hist_x,all_frame0,all_frame1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,size_x] = size(frame_all);
frame0 = frame_all(temp_sound0,:);
frame1 = frame_all(temp_sound1,:);

for i = 1:size_x,
    temp0 = hist(frame0(:,i),hist_x);
    temp1 = hist(frame1(:,i),hist_x);
    temp0 = temp0 ./ length(temp_sound0);
    temp1 = temp1 ./ length(temp_sound1);
    
    all_frame0(i).matrix = [all_frame0(i).matrix; temp0];
    all_frame1(i).matrix = [all_frame1(i).matrix; temp1];
end

return



