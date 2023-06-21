%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
function Image_analysis_230620_sig_neurons_ROI_map

%analysis_task = 'stimulus'
analysis_task = 'reward'

%Get the sound only activity
filename2 = 'Block_tif_170317_6.mat';

filename1 = dir('Delta_180605_1_*'); %Only one 
if length(filename1) == 1,
    filename1 = filename1.name;
else
    filename1
    hoge
end
filename3 = dir('roi_overlap*'); %Only one 
if length(filename3) == 1,
    filename3 = filename3.name;
else
    filename3
    hoge
end

filename4 = dir('overlap181021_*'); %Only one 
if length(filename4) == 1,
    filename4 = filename4.name;
else
    filename4
    hoge
end
data2 = load(filename4);
if length(analysis_task) == 8,
    temp_stim_box = data2.stim_box;
else
    temp_stim_box = data2.rew_box;
end
stim_box(1).matrix = temp_stim_box;
sig_overlap_or = data2.sig_overlap_or;

load(filename1) %Data and XY plot
load(filename3) %roi_overlap

[stim_sound(1).matrix,stim_norm(1).matrix,stim_task(1).matrix,stim_order_block,stim_mean,stim_std] = ...
    Image_analysis_181017_sound_only_pline(filename1,filename2,filename3);

[mean_sound, median_sound, std_sound, ...
 sig_sound, BF, median_sound_low, median_sound_high, ...
 std_sound_low, std_sound_high, length_trial, ...
 median_block_L, median_block_R,...
 median_block_L_correct, median_block_R_correct, ...
 mean_sound_low, mean_sound_high, ...
 logn_mean_low, logn_mean_high, ...
 logn_std_low, logn_std_high, ...
 median_correct, median_error] = ...
      get_sound_activity_matrix(stim_sound, stim_task, stim_box);    
 
BF = BF.matrix(:,2); %Correct trials only
sig_sound = sig_sound.matrix;
%sig_sound,BF is only overlapped ROIs.

%Get only the overlapped ROI.
for i = 1:length(roi_overlap),
    temp_neuron = roi_overlap(i);
    X_plot_overlap(i).matrix = X_plot_all(temp_neuron).matrix;
    Y_plot_overlap(i).matrix = Y_plot_all(temp_neuron).matrix;
end

sig_neuron = find(sig_overlap_or == 1);
neuron_number(1) = length(X_plot_all);
neuron_number(2) = length(roi_overlap);
neuron_number(3) = length(sig_neuron);
neuron_number(4) = length(sig_sound);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot all neuron map with tuning property
number_color = 9;
use_color = jet(number_color);
% % use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
% %              use_color(number_color-2,:); use_color(number_color-1,:); use_color(number_color,:)];
use_color = [use_color(1,:); use_color(2,:); use_color(3,:);...
             use_color(number_color-1,:); use_color(number_color,:); 1 0 0];
[size_y,size_x] = size(roi_map);

figure
%all_sig_neurons
for i = 1:length(sig_neuron),
    temp_neuron = sig_neuron(i);
    fill(X_plot_overlap(temp_neuron).matrix,Y_plot_overlap(temp_neuron).matrix,[0.7 0.7 0.7],'edgecolor','none');
    hold on
end
%all_BF_neuron
for i = 1:length(sig_sound),
    temp_neuron = sig_sound(i);
    temp_BF = BF(temp_neuron);
    fill(X_plot_overlap(temp_neuron).matrix,Y_plot_overlap(temp_neuron).matrix,use_color(temp_BF,:),'edgecolor','none');
    %fill(X_plot_overlap(temp_neuron).matrix,Y_plot_overlap(temp_neuron).matrix,[0.2 0.2 0.2],'edgecolor','none');
    hold on
end
%All edges
for i = 1:length(roi_overlap),
    %fill(X_plot_overlap(i).matrix,Y_plot_overlap(i).matrix,use_color(max_freq2(temp_neuron),:),'edgecolor','none');
    line(X_plot_overlap(i).matrix,Y_plot_overlap(i).matrix,'color',[0 0 0]);
    hold on
end
set(gca,'xlim',[1 size_x],'ylim',[1 size_y])

%Make the color bar
figure
for i = 1:6,
    fill([1,1,2,2],[i-1,i,i,i-1],use_color(i,:),'edgecolor','none')
    hold on
end
axis off

neuron_number

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mouse_mean_sound, mouse_median_sound, mouse_std_sound, ...
          mouse_sig_sound, mouse_BF, mouse_median_sound_low, mouse_median_sound_high, ...
          mouse_std_sound_low, mouse_std_sound_high, mouse_length_trial, ...
          mouse_median_block_L, mouse_median_block_R,...
          mouse_median_block_L_correct, mouse_median_block_R_correct, ...
          mouse_mean_sound_low, mouse_mean_sound_high, ...
          mouse_logn_mean_low, mouse_logn_mean_high, ...
          mouse_logn_std_low, mouse_logn_std_high, ...
          mouse_median_correct, mouse_median_error] = ...
          get_sound_activity_matrix(stim_sound, stim_task, temp_stim_box)    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    length_session = length(stim_sound);
    clear mouse_p_kruskal_stim mouse_BF_stim mouse_sabun_block mouse_std_sound
    clear length_sig_sound mouse_sig_sound mouse_sabun_block_non
    clear mouse_std_block mouse_std_block_non
    
    clear mouse_mean_sound mouse_median_sound mouse_std_sound
    clear mouse_sig_sound mouse_BF
    clear mouse_median_sound_low mouse_median_sound_high
    clear mouse_std_sound_low mouse_std_sound_high
    clear mouse_length_trial
    clear mouse_median_block_L mouse_median_block_R
    clear mouse_median_block_L_correct mouse_median_block_R_correct
        
    clear mouse_logn_mean_low mouse_logn_mean_high
    clear mouse_logn_std_low mouse_logn_std_high
    clear mouse_median_correct mouse_median_error 
    for j = 1:length_session,
        temp_stim = stim_sound(j).matrix;
        temp_stim_task = stim_task(j).matrix;
        [size_trial,size_neuron] = size(temp_stim);

        %Each tone cloud
        stim_category = temp_stim_task(:,1);
        stim_evi   = temp_stim_task(:,4);
        stim_reward = temp_stim_task(:,2);
        stim_block = temp_stim_task(:,5);
        stim_correct = find(stim_reward == 1);
        stim_error   = find(stim_reward == 0);
        stim_block_L = find(stim_block == 0);
        stim_block_R  = find(stim_block == 1);
        stim_category0 = find(stim_category == 0);
        stim_category1 = find(stim_category == 1);
        
        stim_category0L = intersect(stim_category0,stim_block_L);
        stim_category1L = intersect(stim_category1,stim_block_L);
        stim_category0R = intersect(stim_category0,stim_block_R);
        stim_category1R = intersect(stim_category1,stim_block_R);
        
        clear mean_sound_low mean_sound_high
        clear median_sound_low median_sound_high
        clear std_sound_low std_sound_high
        clear length_trial
        mean_sound_low(:,1) = mean(temp_stim(stim_category0L,:),1)'; %trial x neuron -> neuron x sound
        mean_sound_low(:,2) = mean(temp_stim(stim_category0R,:),1)'; %trial x neuron
        mean_sound_low(:,3) = mean(temp_stim(stim_category0,:),1)'; %trial x neuron
        mean_sound_high(:,1) = mean(temp_stim(stim_category1L,:),1)'; %trial x neuron
        mean_sound_high(:,2) = mean(temp_stim(stim_category1R,:),1)'; %trial x neuron
        mean_sound_high(:,3) = mean(temp_stim(stim_category1,:),1)'; %trial x neuron
        median_sound_low(:,1) = median(temp_stim(stim_category0L,:),1)'; %trial x neuron -> neuron x sound
        median_sound_low(:,2) = median(temp_stim(stim_category0R,:),1)'; %trial x neuron
        median_sound_low(:,3) = median(temp_stim(stim_category0,:),1)'; %trial x neuron
        median_sound_high(:,1) = median(temp_stim(stim_category1L,:),1)'; %trial x neuron
        median_sound_high(:,2) = median(temp_stim(stim_category1R,:),1)'; %trial x neuron
        median_sound_high(:,3) = median(temp_stim(stim_category1,:),1)'; %trial x neuron
        std_sound_low(:,1) = std(temp_stim(stim_category0L,:),1)'; %trial x neuron -> neuron x sound
        std_sound_low(:,2) = std(temp_stim(stim_category0R,:),1)'; %trial x neuron
        std_sound_low(:,3) = std(temp_stim(stim_category0,:),1)'; %trial x neuron
        std_sound_high(:,1) = std(temp_stim(stim_category1L,:),1)'; %trial x neuron
        std_sound_high(:,2) = std(temp_stim(stim_category1R,:),1)'; %trial x neuron
        std_sound_high(:,3) = std(temp_stim(stim_category1,:),1)'; %trial x neuron
        
        clear logn_mean_low logn_mean_high
        clear logn_std_low logn_std_high
        min_temp_stim = min(temp_stim,[],1);
        min_temp_stim = repmat(min_temp_stim,size_trial,1);
        temp_stim_log = temp_stim - min_temp_stim + 1; %minimum became 1
        for l = 1:size_neuron,
            temp_activ = temp_stim_log(:,l);
            temp0L = lognfit(temp_activ(stim_category0L));
            temp0R = lognfit(temp_activ(stim_category0R));
            temp0  = lognfit(temp_activ(stim_category0));
            temp1L = lognfit(temp_activ(stim_category1L));
            temp1R = lognfit(temp_activ(stim_category1R));
            temp1  = lognfit(temp_activ(stim_category1));
            logn_mean_low(l,:) =  [temp0L(1), temp0R(1), temp0(1)];
            logn_mean_high(l,:) = [temp1L(1), temp1R(1), temp1(1)];
            logn_std_low(l,:) =  [temp0L(2), temp0R(2), temp0(2)];
            logn_std_high(l,:) = [temp1L(2), temp1R(2), temp1(2)];
        end
        
        length_trial(1,1) = length(stim_category0L);
        length_trial(1,2) = length(stim_category0R);
        length_trial(1,3) = length(stim_category0);
        length_trial(2,1) = length(stim_category1L);
        length_trial(2,2) = length(stim_category1R);
        length_trial(2,3) = length(stim_category1);
%         std_sound_L(:,1) = std_sound_L(:,1) ./ sqrt(length(stim_category0L));
%         std_sound_L(:,2) = std_sound_L(:,2) ./ sqrt(length(stim_category1L));
%         std_sound_R(:,1) = std_sound_R(:,1) ./ sqrt(length(stim_category0R));
%         std_sound_R(:,2) = std_sound_R(:,2) ./ sqrt(length(stim_category1R));

        %BF is determined with activity during reward
        clear median_all median_correct median_error 
        clear median_block_L median_block_R
        clear median_block_L_correct median_block_R_correct
        clear p_kruskal_stim BF_neuron p_RE_neuron p_block_neuron p_block_correct
        clear sabun_block sabun_block_non
        clear std_block std_block_non
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
        end
        
        %About data2 for sig_neurons
        temp_sound = temp_stim_box(j).matrix(:,2);
        if length(temp_sound) ~= length(p_kruskal_stim),
            hoge
        end
        
        %Sound neuron side
        temp_kruskal = find(p_kruskal_stim < 0.01);
        temp_sig = find(temp_sound == 1);
        temp_sig = intersect(temp_sig,temp_kruskal);
        
        %For saving the values
        
        %mouse_p_kruskal_stim(j).matrix = p_kruskal_stim;
        mouse_mean_sound(j).matrix = mean(temp_stim);
        mouse_median_sound(j).matrix = median(temp_stim);
        mouse_std_sound(j).matrix = std(temp_stim);

        mouse_sig_sound(j).matrix = temp_sig;
        mouse_BF(j).matrix = BF_neuron; %reward only
            
        mouse_mean_sound_low(j).matrix = mean_sound_low; 
        mouse_mean_sound_high(j).matrix = mean_sound_high; %trial x neuron
        mouse_median_sound_low(j).matrix = median_sound_low; 
        mouse_median_sound_high(j).matrix = median_sound_high; %trial x neuron
        mouse_std_sound_low(j).matrix = std_sound_low; %trial x neuron -> neuron x sound
        mouse_std_sound_high(j).matrix = std_sound_high; %trial x neuron
        mouse_length_trial(j).matrix = length_trial;

        mouse_logn_mean_low(j).matrix = logn_mean_low;
        mouse_logn_mean_high(j).matrix = logn_mean_high;
        mouse_logn_std_low(j).matrix = logn_std_low;
        mouse_logn_std_high(j).matrix = logn_std_high;
        
        mouse_median_block_L(j).matrix = median_block_L;
        mouse_median_block_R(j).matrix = median_block_R;
        mouse_median_block_L_correct(j).matrix = median_block_L_correct;
        mouse_median_block_R_correct(j).matrix = median_block_R_correct;

        mouse_median_correct(j).matrix = median_correct;
        mouse_median_error(j).matrix = median_error;
    end
    
    return
    



