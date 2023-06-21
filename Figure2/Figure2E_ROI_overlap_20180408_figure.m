%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Figure2E_ROI_overlap_20180408_figure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder_name1 = uigetdir('');
filename_strc1 = dir([folder_name1 filesep 'Delta_Neuropil*.mat']);
filename_strc2 = dir([folder_name1 filesep 'roi_overlap*.mat']);

folder_name2 = uigetdir('');
filename_strc3 = dir([folder_name2 filesep 'Delta_Neuropil*.mat']);
filename_strc4 = dir([folder_name2 filesep 'roi_overlap*.mat']);

filename1=[folder_name1 filesep filename_strc1.name];
filename2=[folder_name1 filesep filename_strc2.name];
filename3=[folder_name2 filesep filename_strc3.name];
filename4=[folder_name2 filesep filename_strc4.name];

data1 = load(filename1);
data2 = load(filename2);
data3 = load(filename3);
data4 = load(filename4);

%%%%%%%%%%%%%%%%%%
roi_map1 = data1.roi_map;
X_plot1 = data1.X_plot_all;
Y_plot1 = data1.Y_plot_all;
roi_overlap1 = data2.roi_overlap;

roi_map2 = data3.roi_map;
X_plot2 = data3.X_plot_all;
Y_plot2 = data3.Y_plot_all;
roi_overlap2 = data4.roi_overlap;

%%%%%%%%%%%%%%%%%
%Draw the roi_map

[length(roi_overlap1), length(Y_plot1), length(roi_overlap2), length(Y_plot2)]
%[sort(roi_overlap1),sort(roi_overlap2)]

%Based on the file1 re-number the roi map
for i = 1:length(roi_overlap1)
    temp = roi_overlap1(i);
    y_overlap(i) = mean(Y_plot1(temp).matrix);
end
[y,sort_y] = sort(y_overlap);
roi_overlap_new1 = roi_overlap1(sort_y);
roi_overlap_new2 = roi_overlap2(sort_y);

overlap_map1 = roi_map_overlab(roi_map1, X_plot1, Y_plot1, roi_overlap1);
overlap_map2 = roi_map_overlab(roi_map2, X_plot2, Y_plot2, roi_overlap2);

figure
subplot(1,2,1)
imagesc(overlap_map1)
subplot(1,2,2)
imagesc(overlap_map2)
colormap jet

overlap_map1 = roi_map_overlab(roi_map1, X_plot1, Y_plot1, roi_overlap_new1);
overlap_map2 = roi_map_overlab(roi_map2, X_plot2, Y_plot2, roi_overlap_new2);

figure
subplot(1,2,1)
imagesc(overlap_map1)
subplot(1,2,2)
imagesc(overlap_map2)
colormap jet

roi_map_overlab_figure(roi_map1, X_plot1, Y_plot1, roi_overlap_new1)
roi_map_overlab_figure(roi_map2, X_plot2, Y_plot2, roi_overlap_new2)
roi_map_overlab_figure_non_color(roi_map1, X_plot1, Y_plot1, roi_overlap_new1)
roi_map_overlab_figure_non_color(roi_map2, X_plot2, Y_plot2, roi_overlap_new2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function roi_map_overlab_figure(roi_map, X_plot_all, Y_plot_all, roi_overlap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_y,size_x] = size(roi_map);
non_overlap = setdiff([1:length(X_plot_all)],roi_overlap);

%use_color = jet(length(roi_overlap));
use_color = cool(length(roi_overlap));
color_none = [1 1 1];

clear max_freq
figure
for i = 1:length(roi_overlap),
    temp_neuron = roi_overlap(i);
    fill(X_plot_all(temp_neuron).matrix,Y_plot_all(temp_neuron).matrix,use_color(i,:),'edgecolor','none');
    hold on
end
for i = 1:length(non_overlap),
    temp_neuron = non_overlap(i);
    fill(X_plot_all(temp_neuron).matrix,Y_plot_all(temp_neuron).matrix,color_none,'edgecolor','none');
    hold on
end
%Plot edge for all neurons
for i = 1:length(X_plot_all),
    plot(X_plot_all(i).matrix,Y_plot_all(i).matrix,'k');
    hold on
end
set(gca,'xlim',[1,size_x],'ylim',[1,size_y])

axis ij

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function roi_map_overlab_figure_non_color(roi_map, X_plot_all, Y_plot_all, roi_overlap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_y,size_x] = size(roi_map);
non_overlap = setdiff([1:length(X_plot_all)],roi_overlap);

%use_color = jet(length(roi_overlap));
use_color = cool(length(roi_overlap));
color_none = [1 1 1];

clear max_freq
figure
for i = 1:length(roi_overlap),
    temp_neuron = roi_overlap(i);
    plot(X_plot_all(temp_neuron).matrix,Y_plot_all(temp_neuron).matrix,'color',use_color(i,:));
    hold on
end
set(gca,'xlim',[1,size_x],'ylim',[1,size_y])

axis ij

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function overlap_map = roi_map_overlab(roi_map, X_plot, Y_plot, roi_overlap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[size_y,size_x] = size(roi_map);
overlap_map = zeros(size_y,size_x);

for i = 1:length(roi_overlap),
    temp_neuron = roi_overlap(i);
    temp_roi = find(roi_map == temp_neuron);
    overlap_map(temp_roi) = i;
end

return
