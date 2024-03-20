% Summary: based on the optimised the weights masks, calculate the mean
% gaze deviation on each voxel
% 
% Author. Tianbo Xu
% Institution. Institute of Neurology, UCL.
% 
% init. 12.11.2015
function mean_gaze_on_weight_voxel_20151112151839()

clc
clear 
% close all

%% variables
% SVMLin package path
% simulation section folder
% weights masks data folder
svml_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015';
simu_eff_folder = '2_simulations';
wei_folder = 'b_pre_weights_ref_mask';
mask_folder = 'gaze_mask';

% dimensions of image data
dims = [31 37 31];

% threshold
thresh = 0.1;

% prefix of output weights masks
prefix = 'wm_gaze';

%% loading data
load('zhead1333_6mm.mat');
head = zhead1333_6mm;

cd([svml_path '/' simu_eff_folder '/' wei_folder]);
load(['op_wei_dat_' num2str(thresh) '.mat']);
wdata = op_wei_dat;

load('CT_vx_gaze_kde.mat');
CT_R = CT_vx_gaze_R;
CT_L = CT_vx_gaze_L;
CT_mu = CT_vx_gaze_mu;

load('MRI_vx_gaze_kde.mat');
MRI_R = MRI_vx_gaze_R;
MRI_L = MRI_vx_gaze_L;
MRI_mu = MRI_vx_gaze_mu;

cd(svml_path)

for i = 1 : size(wdata, 1)
    
    % weighting data and index of weighted voxeles
    tmp_w = wdata(i, :);
    tmp_idx = find(tmp_w ~= 0);   
    
    % image data for output
    tmp_dat_CT_R = zeros(length(tmp_w), 1);
    tmp_dat_CT_R(tmp_idx, :) = CT_R(tmp_idx, :);
    
    tmp_dat_CT_L = zeros(length(tmp_w), 1);
    tmp_dat_CT_L(tmp_idx, :) = CT_L(tmp_idx, :);
    
    tmp_dat_CT_mu = zeros(length(tmp_w), 1);
    tmp_dat_CT_mu(tmp_idx, :) = CT_mu(tmp_idx, :);
    
    tmp_dat_MRI_R = zeros(length(tmp_w), 1);
    tmp_dat_MRI_R(tmp_idx, :) = MRI_R(tmp_idx, :);
    
    tmp_dat_MRI_L = zeros(length(tmp_w), 1);
    tmp_dat_MRI_L(tmp_idx, :) = MRI_L(tmp_idx, :);
    
    tmp_dat_MRI_mu = zeros(length(tmp_w), 1);
    tmp_dat_MRI_mu(tmp_idx, :) = MRI_mu(tmp_idx, :);
    
    % write weights masks
    tmp_h_CT_R = head{1};
    tmp_h_CT_R.fname = [sprintf('%02d', i) '_' prefix '_CT_R_' num2str(thresh) '.nii'];
    
    tmp_h_CT_L = head{1};
    tmp_h_CT_L.fname = [sprintf('%02d', i) '_' prefix '_CT_L_' num2str(thresh) '.nii'];
    
    tmp_h_CT_mu = head{1};
    tmp_h_CT_mu.fname = [sprintf('%02d', i) '_' prefix '_CT_mu_' num2str(thresh) '.nii'];
    
    tmp_h_MRI_R = head{1};
    tmp_h_MRI_R.fname = [sprintf('%02d', i) '_' prefix '_MRI_R_' num2str(thresh) '.nii'];
    
    tmp_h_MRI_L = head{1};
    tmp_h_MRI_L.fname = [sprintf('%02d', i) '_' prefix '_MRI_L_' num2str(thresh) '.nii'];
    
    tmp_h_MRI_mu = head{1};
    tmp_h_MRI_mu.fname = [sprintf('%02d', i) '_' prefix '_MRI_mu_' num2str(thresh) '.nii'];
    
    cd([svml_path '/' simu_eff_folder '/' wei_folder]);
    
    if ~isdir([mask_folder '_' num2str(thresh)])
        system(['mkdir ' mask_folder '_' num2str(thresh)]);
    end
    
    cd([mask_folder '_' num2str(thresh)])
    
    spm_write_vol(tmp_h_CT_R, reshape(tmp_dat_CT_R, dims));
    spm_write_vol(tmp_h_CT_L, reshape(tmp_dat_CT_L, dims));
    spm_write_vol(tmp_h_CT_mu, reshape(tmp_dat_CT_mu, dims));
    
    spm_write_vol(tmp_h_MRI_R, reshape(tmp_dat_MRI_R, dims));
    spm_write_vol(tmp_h_MRI_L, reshape(tmp_dat_MRI_L, dims));
    spm_write_vol(tmp_h_MRI_mu, reshape(tmp_dat_MRI_mu, dims));
    
    clear tmp_w tmp_idx tmp_dat_*
    clear temp_h_* 
end


%% end of this function
end

