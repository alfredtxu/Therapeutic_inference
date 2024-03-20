% Summary. this function aims to visulise the lesions comparing to the
% weighting masks
% 
% Author. Tianbo XU
% 
% Organisation. Institute of Neurology, UCL
% 
% init. 22.10.2015
% 
function random_lesion_review()

% begging of this function
clc
clear
% close all

% loading data
load('zhead1333_6mm.mat');
head = zhead1333_6mm;

% load the prepared data matrices
load('zeta1333_6mm_bin.mat');
data = zeta1333_6mm_bin;

% sample path 
simu_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015/2_simulations';
sam_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015/2_simulations/f_post_samples';

% the number of samples
cnt = 10;

% random index of dataset
ran_idx = randperm(size(head, 2));

% sampled index
sam_idx = ran_idx(1 : cnt);

for i = 1 : cnt
    
    temp_d = squeeze(data(sam_idx(i), :, :, :));
    temp_h = head{sam_idx(i)};
    
    cd(sam_path);
    spm_write_vol(temp_h, temp_d);
    cd(simu_path);
    
end

% end of this function
end

