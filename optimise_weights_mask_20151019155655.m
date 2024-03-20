% Summary: check through the strength of weights to decide the threshold
% and optimise the weights masks
% 
% Author. Tianbo Xu
% Institution. Institute of Neurology, UCL.
% 
% init. 19.10.2015
% 
% comm. 20.10.2015
%       * based on the weighting data, remove zeros and the voxels those
%       weaker than pre-set threshold (10^-2 / 10^-3)
function optimise_weights_mask_20151019155655()

clc
clear 
close all

%% variables
% SVMLin package path
% simulation section folder
% weights masks data folder
svml_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015';
simu_eff_folder = '2_simulations';
wei_folder = 'b_pre_weights_ref_mask';

% dimensions of image data
dims = [31 37 31];

% prefix of output weights masks
prefix = 'op_wm';
key_str = '20150917165945_ctmri_pre_20150917170318_1';

%% loading data
load('zhead1333_6mm.mat');
head = zhead1333_6mm;

cd([svml_path '/' simu_eff_folder '/' wei_folder]);
load('wei_dat.mat');
wdata = wei_dat;
cd(svml_path)

%% processing
thresh = 10^-3;

for i = 1 : size(wdata, 1)
    
    tmp_w = wdata(i, :, :, :);
    
    % the voxels those are weaker than the threshold: clear to be zeros
    tmp_w(find(abs(tmp_w) < thresh)) = 0;
    
    % normalise the strength
    tmp_w_nonzero = tmp_w;
    
    tmp_w_nonzero(find(tmp_w_nonzero == 0)) = [];
    stre_min = min(abs(tmp_w_nonzero));
    
    tmp_w = tmp_w / stre_min;
    
    tmp_h = head{1};
    tmp_h.fname = [sprintf('%02d', i) '_' prefix '_' num2str(thresh) '_' key_str '.nii'];
    
    op_wei_dat(i, :) = tmp_w;
    
    cd([svml_path '/' simu_eff_folder '/' wei_folder]);
    spm_write_vol(tmp_h, reshape(abs(tmp_w), dims));
    
    clear tmp_w tmp_w_nonzero stre_min tmp_h
end

save(['op_wei_dat_' num2str(thresh) '.mat'], 'op_wei_dat');
cd(svml_path);


%% end of this function
end

%% temporal backup 
% [A idx] = sort(lab_size,'descend');
% 
% vxx = [];
% 
% for n = 584 : 916
%     vxx = [vxx, find(lab == idx(n))];
% end
% 
% ss = tmp_w;
% topv = ss(vxx);
% 
% outdat = zeros(1, 35557);
% outdat(vxx) = topv;
% 
% tmp_h = head{1};
% tmp_h.fname = 'wei1_clus_1n2.nii';
% spm_write_vol(tmp_h, reshape(abs(outdat), dims));

