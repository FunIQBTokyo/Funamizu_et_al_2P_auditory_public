%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Figure2C_Image_analysis_180123_sound2

[filename1, pathname1]=uigetfile('*.mat','Delat_Image_mat');
load(filename1)
%delta_trace roi_map Y_plot_all X_plot_all

[filename2, pathname2]=uigetfile('*.mat','Block_tif');
load(filename2)
%length_trial frame_sound stim_order

[filename3, pathname3]=uigetfile('*.mat','ROI_overlap');
load(filename3) %roi_overlap

%Detect sound with only high intensity
pure_tone = find(stim_order(:,1) == 1);
pure_int  = stim_order(pure_tone,3);

%pure_int1 = find(pure_int == 1);
pure_int2 = find(pure_int == 2);
%pure_tone1 = pure_tone(pure_int1);
pure_tone2 = pure_tone(pure_int2);
pure_freq = stim_order(pure_tone2,2);

for i = 1:max(pure_freq), %max18
    num_freq(i,:) = find(pure_freq == i);
end
for i = 1:max(pure_freq)/2, %now 9
    temp1 = num_freq(2*i-1,:);
    temp2 = num_freq(2*i,:);
    num_freq2(i,:) = sort([temp1,temp2]);
    num_freq2_all(num_freq2(i,:)) = i;
end

%Neural activity analysis
[trace_number,frame_length] = size(delta_trace);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalized the delta_trace_activity
mean_trace = mean(delta_trace');
mean_trace = mean_trace';
std_trace  = std(delta_trace');
std_trace = std_trace';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time_window = 45; %1.0 sec
temp_pre = 45;
temp_post = 180;

sig_thre = 2;

kijyun_sound  = sig_trace_get_pre(delta_trace, pure_tone2, frame_sound, time_window);
mean_sound  = sig_trace_get(delta_trace, pure_tone2, frame_sound, time_window);

norm_sound = mean_sound;

%Kentei with signrank test
for i = 1:trace_number,
    %sound (all, sound_left, sound_right)
    p_sound(i,1) = signrank(mean_sound(i,:), kijyun_sound(i,:), 'tail','right');

    %Get the best active frequency by 2 freq integrating
    clear freq_trace freq_trace2
    for j = 1:max(pure_freq), %now 9
        temp_num = num_freq(j,:);
        temp_temp_trace = mean_sound(i,temp_num);
        freq_trace(j) = mean(temp_temp_trace);
    end
    for j = 1:max(pure_freq)/2, %now 9
        temp_num = num_freq2(j,:);
        temp_temp_trace = mean_sound(i,temp_num);
        temp_kijyun_trace = kijyun_sound(i,temp_num);
        freq_trace2(j) = mean(temp_temp_trace);
        %get the p value for each frequency
        p_sound_freq(i,j) = signrank(temp_temp_trace, temp_kijyun_trace, 'tail','right');
    end
    max_freq(i)  = find(freq_trace == max(freq_trace),1);
    max_freq2(i) = find(freq_trace2 == max(freq_trace2),1);
end

%Get the significance in traces
%Sound onset
Sound_trace = sig_trace_analysis_sound(delta_trace, pure_tone2, frame_sound, trace_number, temp_pre, temp_post);

p_sound_freq2 = min(p_sound_freq');
p_sound_freq2 = p_sound_freq2';

%sig_thre for figures
sig_thre = 2;
%sig_thre = 3;
temp_xtick = 45/4;
%sound_trace is z-score of activity
sig_sound_neuron = find(-log10(p_sound) >= sig_thre)
%sig_sound_neuron_freq = find(-log10(p_sound_freq2) >= sig_thre+1)
sig_sound_neuron_freq = find(p_sound_freq2 < 0.005)
use_window = [temp_pre+1:temp_pre+time_window];
use_color = jet(max(pure_freq)/2);

plot_sig_thre = find(-log10(p_sound_freq2) >= 4.3); %Find nice tuning curve
%sort(-log10(p_sound_freq2(sig_sound_neuron_freq)))

sig_sound_neuron = sig_sound_neuron_freq
length(sig_sound_neuron)
plot_sig_neuron = plot_sig_thre(17)
%sig_sound_neuron = intersect(sig_sound_neuron, sig_sound_neuron_freq)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for i = 1:length(sig_sound_neuron),
%     temp_neuron = sig_sound_neuron(i);
for i = 1:length(plot_sig_neuron),
    temp_neuron = plot_sig_neuron(i);
    temp_trace = Sound_trace(temp_neuron).matrix;
    temp_ave = norm_sound(temp_neuron,:);

    clear freq_trace
    figure
    subplot(1,2,1)
    for j = 1:max(pure_freq)/2, %now 9
        temp_num = num_freq2(j,:);
        temp_temp_trace = temp_trace(temp_num,:);
        freq_trace(j,:) = mean(temp_temp_trace);
        plot_mean_se_moto(temp_temp_trace,use_color(j,:),2); %se
        hold on
    end
    %set(gca,'xlim',[1,temp_pre + temp_post+1])
    set(gca,'xlim',[1,135])
    set(gca,'xtick',[0:temp_xtick:temp_pre + temp_post+1])
    
    
    %Bar graph is better to explain the High Low and Reward Error relation
    subplot(1,2,2)
    boxplot(temp_ave, num_freq2_all)
end 

%Plot all neuron map with tuning property
count = 0;
map_freq = roi_map;
map_sig = zeros(size(roi_map));
for i = 1:trace_number
    temp_place = find(roi_map == i);
    map_freq(temp_place) = max_freq2(i);
    
    %Plot with sig neuron
    temp_sig = find(sig_sound_neuron == i);
    if length(temp_sig) ~= 0,
        count = count + 1;
        map_sig(temp_place) = max_freq2(i);
        %map_sig(temp_place) = count; % For position of neuron
    end
end

[size_y,size_x] = size(roi_map);
figure
imagesc(roi_map)
colormap('jet')
figure
subplot(1,2,1)
imagesc(map_freq)
colormap('jet')
caxis([0 max(pure_freq)/2])
subplot(1,2,2)
imagesc(map_sig)
colormap('jet')
caxis([0 max(pure_freq)/2])

%Make the color bar
figure
for i = 1:max(pure_freq)/2,
    fill([1,1,2,2],[i-1,i,i,i-1],use_color(i,:),'edgecolor','none')
    hold on
end
axis off

%Plot based on boundary
%ONLY SIG Neurons
clear max_freq
figure
%for i = 1:length(plot_sig_neuron),
%    temp_neuron = plot_sig_neuron(i);
for i = 1:length(sig_sound_neuron),
    temp_neuron = sig_sound_neuron(i);
    fill(X_plot_all(temp_neuron).matrix,Y_plot_all(temp_neuron).matrix,use_color(max_freq2(temp_neuron),:),'edgecolor','none');
    hold on
end
%Plot edge for all neurons
for i = 1:trace_number,
    plot(X_plot_all(i).matrix,Y_plot_all(i).matrix,'k');
    hold on
end
set(gca,'xlim',[1,size_x],'ylim',[1,size_y])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sound_trace = sig_trace_analysis_sound(delta_trace, Choice_trial, frame_sound, trace_number, temp_pre, temp_post)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sound_trace = [];
for neuron = 1:trace_number,
    [neuron, trace_number]
    
    temp_sound  = zeros(length(Choice_trial),temp_pre + temp_post);
    for i = 1:length(Choice_trial);
        %Sound onset
        temp_frame = frame_sound(Choice_trial(i));
        if temp_frame-temp_pre+1 > 0,
            temp_sound(i,:) = delta_trace(neuron, temp_frame-temp_pre+1:temp_frame+temp_post);
        else
            temp_start_frame = -(temp_frame-temp_pre+1);
            temp_sound(i,temp_start_frame+2:temp_pre + temp_post) = delta_trace(neuron, 1:temp_frame+temp_post);
        end
    end
    sound_trace(neuron).matrix = temp_sound;
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function kijyun_sound = sig_trace_get(delta_trace, Choice_trial, frame_RE, time_window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(Choice_trial);
    temp_frame = frame_RE(Choice_trial(i));
    temp_trace = delta_trace(:, [temp_frame : temp_frame+time_window-1]);
    temp_trace = mean(temp_trace');
    kijyun_sound(:,i) = temp_trace'; 
end
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function kijyun_sound = sig_trace_get_pre(delta_trace, Choice_trial, frame_RE, time_window)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(Choice_trial);
    temp_frame = frame_RE(Choice_trial(i));
    if temp_frame-time_window > 0,
        temp_trace = delta_trace(:, [temp_frame-time_window : temp_frame-1]);
    else
        temp_frame
        temp_trace = delta_trace(:, [1 : temp_frame-1]);
    end
    temp_trace = mean(temp_trace');
    kijyun_sound(:,i) = temp_trace'; 
end
return
    

