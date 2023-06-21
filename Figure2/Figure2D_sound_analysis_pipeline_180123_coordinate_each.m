%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}

function Figure2D_sound_analysis_pipeline_180123_coordinate_each

[filename1, pathname1,findex]=uigetfile('*.mat','Pure_tone_file');
filename1
%all_max_freq all_max_freq2 all_sig_sound all_roi_overlap

[filename2, pathname2,findex]=uigetfile('*.mat','Coordinate_file');
filename2
%all_xyz all_radius all_ref_xyz all_green all_red

%First L is for block, Second L is for sound
% Block_LL = intersect(Block_L,Sound_L);
% Block_LR = intersect(Block_L,Sound_R);
% Block_RL = intersect(Block_R,Sound_L);
% Block_RR = intersect(Block_R,Sound_R);

clear neuron_number
all_max_freq = [];
all_max_freq2 = [];
    
all_sig_sound = [];

all_xyz = [];
%for i = 1 : length(filename1)
    clear data data2 temp_filename temp_pass fpath
    
    fpath1 = fullfile(pathname1, filename1);
    data  = load(fpath1)
    fpath2 = fullfile(pathname2, filename2);
    data2 = load(fpath2)
    
    all_xyz = [all_xyz; data2.all_xyz];
    
    [neuron_number,~] = size(data.all_max_freq);
    
    %Correct for all sessions
    all_max_freq  = [all_max_freq; data.all_max_freq];
    all_max_freq2 = [all_max_freq2; data.all_max_freq2];
    
    if i > 1,
        all_sig_sound = [all_sig_sound; data.all_sig_sound + sum(neuron_number(1:i-1))];
    else
        all_sig_sound =   data.all_sig_sound;
    end
%end
    
[length(all_sig_sound) length(all_max_freq)] %all sig neuron

sig_xyz = all_xyz(all_sig_sound,:);

sig_freq  = all_max_freq(all_sig_sound);
sig_freq2 = all_max_freq2(all_sig_sound);

Figure2D_plot_3D_2D_auditory_map_gauss(sig_freq2, sig_freq2, sig_xyz, 9); %for sig_freq2


