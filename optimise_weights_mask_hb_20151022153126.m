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
% 
%       * half brain
function optimise_weights_mask_hb_20151022153126()

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
thresh = 10^-2;

for i = 1 : size(wdata, 1)
    
    temp_w = wdata(i, :, :, :);
    
    % the voxels those are weaker than the threshold: clear to be zeros
    temp_w(find(abs(temp_w) < thresh)) = 0;
    
    % remove positive valued voxels
    temp_w(find(temp_w > 0)) = 0;
    
    % normalise the strength
    temp_w_nonzero = temp_w;
    temp_w_nonzero(find(temp_w_nonzero == 0)) = [];
    temp_w = temp_w / min(abs(temp_w_nonzero));
    
    % binarise data
    temp_w_bin = logical(temp_w);
    
    op_wei_dat(i, :) = temp_w;
    op_wei_dat_bin(i, :) = temp_w_bin;
    
    % write weights masks
    temp_h1 = head{1};
    temp_h1.fname = [sprintf('%02d', i) '_' prefix '_' num2str(thresh) '_' key_str '.nii'];
    
    temp_h2 = head{1};
    temp_h2.fname = [sprintf('%02d', i) '_' prefix '_' num2str(thresh) '_' key_str '_bin.nii'];
    
    cd([svml_path '/' simu_eff_folder '/' wei_folder]);
    spm_write_vol(temp_h1, reshape(abs(temp_w), dims));
    spm_write_vol(temp_h1, reshape(temp_w, dims));
    spm_write_vol(temp_h2, reshape(temp_w_bin, dims));
    
    clear temp_w temp_w_bin temp_w_nonzero
    clear temp_h1 temp_h2
end

save(['op_wei_dat_L_' num2str(thresh) '.mat'], 'op_wei_dat');
save(['op_wei_dat_L_bin_' num2str(thresh) '.mat'], 'op_wei_dat_bin');
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

