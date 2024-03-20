% summary: label connected components and find out the proper threshold for
% dialation, empirically
% 
% UCL Institute of Neurology
% Tianbo XU
% init. 24.11.2015
function optimise_threhold_dilation_20151125115718()

clc
clear
close all

%% load prepared data matrices
load('zhead1333.mat');
head = zhead1333;

load('zeta1333_log.mat');
data = zeta1333_log;

% variables
thresh_vx = 1 : 30;
dim = head{1}.dim;

% downsample sizes
ds = [6,8];

%% label the binary images 
for i = 1 : size(data,1)
   
    tmp_dat = data(i, :);
    tmp_dat = reshape(tmp_dat, dim);
    
    % clustering: 26 connections
    [label, clus] = bwlabeln(tmp_dat);
    
    for j = 1 : clus
        
        tmp_idx = find(label == j);
        
        tmp_clus_vol = length(tmp_idx);
        
        tmp_clus_dat(j, :, :, :) = tmp_dat;
        tmp_clus_dat(j, ~tmp_idx) = 0;
        
        if le(tmp_clus_vol, thresh_vx)
            
            % dilating the cluster
            fprintf('Dilating: %d - %d >> %d\n', i, j, tmp_clus_vol);
            tmp_clus_dat(j,:,:,:) = imdilate(squeeze(tmp_clus_dat(j,:,:,:)),NHOOD);
        end
        
        clear tmp_idx tmp_clus_vol
    end
    
    dat_dil(i, :, :, :) = sum(tmp_clus_dat, 1);
    
    clear tmp_dat label clus
end


%% end of this function
end