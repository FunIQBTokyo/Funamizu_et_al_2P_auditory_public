%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Figure2B_fast_iilizer_save_file_20200128
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


folder_name = uigetdir('');

filename_strc = dir([folder_name filesep '*.tiff']);
%bfile_strc = dir([folder_name filesep '*.mat']);

filenames={filename_strc.name};
filenames = char(filenames);

[bname,bpath]=uigetfile('')
bcontrol_file_name=[bpath,bname];
load(bcontrol_file_name)

for ii=1:length(all_data)-1;
    freq(ii,1)=all_data(ii).freq;
    db(ii,1)=all_data(ii).volume;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Select the file to analyze
%filenumber = [14 17 22];
filenumber = find(freq < 5000) %pick4kHz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filenumber = [2 14 17 22] %pick4kHz

baseline1 = 1;
baseline2 = 8;
stimulus1 = 9;
stimulus2 = 16;
%winsize = 5;
winsize = 15;
%winsize = 20;

sessionname = folder_name(max(strfind(folder_name,'\'))+1:end);

for t=1:length(filenumber)
    
    filename=[folder_name filesep sessionname '_R' num2str(filenumber(t)) '.tiff'];
    filename
    
    for bb=baseline1:baseline2
        BB(:,:,bb-baseline1+1)=imread(filename,bb);
    end
    
    for ss=stimulus1:stimulus2
        SS(:,:,ss-stimulus1+1)=imread(filename,ss);
    end
    
    SS_over_BB(:,:,t)=(mean(BB,3)-mean(SS,3))./mean(BB,3);

end

%case 'gausian'
convwin=gausswin(winsize)*gausswin(winsize)';
convwin=convwin./(sum(winsize(:)));

mean_SS_over_BB= mean(SS_over_BB,3);
conved_SS_BB=conv2(mean_SS_over_BB,convwin);

figure
imagesc(conved_SS_BB);
%colorbar;
colormap(gray);

size(conved_SS_BB)
min_image = min(min(conved_SS_BB));
max_image = max(max(conved_SS_BB));
tif_SS_BB = (conved_SS_BB - min_image) ./ (max_image - min_image);

% figure
% imagesc(tif_SS_BB);
% colormap(gray);

save_file_name = [folder_name, '_4kHz_20200128.tif'];
%imwrite(conved_SS_BB,save_file_name)
imwrite(tif_SS_BB,save_file_name)
%save(save_file_name, 'conved_SS_BB');

% thre = 0.01;
% temp = find(conved_SS_BB > max_image - thre);
% conved_SS_BB(temp) = max_image - thre;
% min_image = min(min(conved_SS_BB));
% max_image = max(max(conved_SS_BB));
% 
% tif_SS_BB2 = (conved_SS_BB - min_image) ./ (max_image - min_image);
% 
% save_file_name2 = [folder_name, '_4kHz_2.tif'];
% %imwrite(conved_SS_BB,save_file_name)
% imwrite(tif_SS_BB2,save_file_name2)
% %save(save_file_name, 'conved_SS_BB');

