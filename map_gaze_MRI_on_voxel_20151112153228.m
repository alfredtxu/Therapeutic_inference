% gaze mask: mapping mean gaze on each voxel 
% 
% Author. Tianbo XU
% Institution. Institute of Neurology, UCL
% 
% init. 12.11.2015
% 

function [MRI_vx_gaze_R, MRI_vx_gaze_L, MRI_vx_gaze_mu] = map_gaze_MRI_on_voxel_20151112153228()

clc
clear
% close all 

%% adjustment of gaze deviation for MRI
% R: 7.0269; 
% L: -3.4669
% mean: 0.9807

% kde_R = 7.0269;
% kde_L = -3.4669;
% kde_mu = 0.9807;

kde_R = 0;
kde_L = 0;
kde_mu = 0;

%% load the prepared data matrices
load('zeta1333_6mm_bin.mat');
data = zeta1333_6mm_bin;
dims = size(data);
zeta = reshape(data, [dims(1) prod(dims(2:4))]);

%% load information struct of patients
load('stroke_info_light.mat');
info = stroke_info_light;

%% refine the cases those appear as proper size of leisons by the thresholds
% threshold: minimum / maximum volume of lesions
vol_min = 1;
vol_max = 200;

% counter: the number of cases within thresholds
c_vol = 0;

% filtering cases
for i = 1 : dims(1)
    
    tmpvol = sum(zeta(i,:));
    
    if ge(tmpvol, vol_min) && le(tmpvol, vol_max)
        
        c_vol = c_vol + 1;
        
        tmpzeta = reshape(zeta(i, :), dims(2:4));
        data_v(c_vol, :, :, :) = tmpzeta;
        info_v(c_vol) = info(i);
        
        clear tmpzeta
    end
    
    clear tmpvol
end

%% extract gazes
cnt = 0;
for i = 1 : length(info_v)
    
    tmp_info = info_v(i);
    
    if tmp_info.ctgazeF == 1 && tmp_info.t2gazeF == 1
        
        cnt = cnt + 1;
        
        % refine data
        data_gaze(cnt, :, :, :) = data_v(i, :, :, :);
        
        % refine gaze
        mri_R(cnt, :) = tmp_info.t2resc_calc(1);
        mri_L(cnt, :) = tmp_info.t2resc_calc(2);
        mri_mu(cnt, :) = mean(tmp_info.t2resc_calc);
        
        % judgement of laterality
        if strcmpi(tmp_info.lat, 'left')
            lat(cnt)=-1;
        elseif strcmpi(tmp_info.lat, 'right')
            lat(cnt)=1;
        else
            lat(cnt)=0;
        end
    end
    
    clear tmp_info 
end

% corrected gaze deviation
mri_R_corr = mri_R - kde_R;
mri_L_corr = mri_L - kde_L;
mri_mu_corr = mri_mu - kde_mu;

%% mean gaze hit on each voxels
dims_g = size(data_gaze);
zeta_g = reshape(data_gaze, [dims_g(1) prod(dims_g(2:4))]);

for i = 1 : size(zeta_g, 2)
   
    tmp_vx = zeta_g(:, i);
    tmp_idx = find(tmp_vx == 1);
    
    if ~isempty(tmp_idx )
        MRI_vx_gaze_R(i, :) = mean(mri_R_corr(tmp_idx, :));
        MRI_vx_gaze_L(i, :) = mean(mri_L_corr(tmp_idx, :));
        MRI_vx_gaze_mu(i, :) = mean(mri_mu_corr(tmp_idx, :));
    else
        MRI_vx_gaze_R(i, :) = 0;
        MRI_vx_gaze_L(i, :) = 0;
        MRI_vx_gaze_mu(i, :) = 0;
    end
    
    clear tmp_vx tmp_idx
end


%% end of this function
end


