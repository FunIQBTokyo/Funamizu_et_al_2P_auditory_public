%{
----------------------------------------------------------------------------
Analyzing behavioral data
At least for the correct rate
----------------------------------------------------------------------------
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Figure2D_plot_3D_2D_auditory_map_gauss(sig_freq, sig_freq_color, sig_xyz, number_color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%sig_freq: Value that you want to plot
%sig_xyz:  Coordiante of each value
%number_color: from 1 to max, the color for sig_freq

use_color = jet(number_color); %for sig_freq2
%use_color = [0 0 1; 1 0 0]; %for sig_freq2

%Get the colormap (imagesc) for tuning map
edges_x = [-inf,-1000:100:700,inf];  %x
edges_y = [inf,1400:-100:-300,-inf]; %y
edges_x_fig = [-1100:100:700]  + 50;
edges_y_fig = [1500:-100:-300] - 50;
temp_x_tick = [1:2:length(edges_x_fig)];
temp_y_tick = [1:2:length(edges_y_fig)];

gauss_std = 50; %100um

L_edges1 = length(edges_x) - 1;
L_edges2 = length(edges_y) - 1;

map_xy_freq = zeros(L_edges2,L_edges1);

for i = 1:length(edges_y)-1,
    temp_y1 = find(sig_xyz(:,2) < edges_y(i) + gauss_std);
    temp_y2 = find(sig_xyz(:,2) >= edges_y(i+1) - gauss_std);
    temp_y = intersect(temp_y1,temp_y2);
    for j = 1:length(edges_x)-1,
        temp_x1 = find(sig_xyz(:,1) > edges_x(j) - gauss_std);
        temp_x2 = find(sig_xyz(:,1) <= edges_x(j+1) + gauss_std);
        temp_x = intersect(temp_x1,temp_x2);
        temp_xy = intersect(temp_y,temp_x);
        
        if length(temp_xy) ~= 0,
            map_xy_freq(i,j) = mean(sig_freq(temp_xy));
        end
    end
end

%Map for freq2 (9 freq)
%Plot 3D
figure
for i = 1:length(sig_freq),
    plot3(sig_xyz(i,1),sig_xyz(i,2),sig_xyz(i,3),'.','color',use_color(sig_freq_color(i),:))
    hold on
end
xlabel('Dorsal-Medial (um)')
ylabel('Posterior-Anterior (um)')
zlabel('Depth (um)')

figure
imagesc(map_xy_freq)
colormap('jet')
set(gca,'xtick',temp_x_tick,'ytick',temp_y_tick)
set(gca,'xticklabel',edges_x_fig(temp_x_tick),'yticklabel',edges_y_fig(temp_y_tick))
xlabel('Dorsal-Medial (um)')
ylabel('Posterior-Anterior (um)')
