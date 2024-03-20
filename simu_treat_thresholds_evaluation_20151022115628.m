% Summary. using best trained classifier in the 'precases' as training results to yield outcomes.
%          simulating treatment on partial cases to see optimise parameters.
%
% Author. Tianbo XU
%
% Institution. Institute of Neurology, UCL
%
% init. 22.10.2015
%

%% beginning of this script
clc
clear
close all

%% load general data matrices
% load the prepared data matrices
load('zeta1333_6mm_bin.mat');
data = zeta1333_6mm_bin;

dims = size(data);
zeta = reshape(data, [dims(1) prod(dims(2:4))]);

% load information struct of patients
load('stroke_info_light.mat');
info = stroke_info_light;

%% refine the cases those appear as proper size of leisons by the thresholds
% threshold: minimum volume of lesions
vol_min = 1;

% threshold: maximum volume of lesions
vol_max = 200;

% counter: the number of cases within thresholds
c_vol = 0;

% processing in loop
for i = 1 : dims(1)
    
    temp_vol = sum(zeta(i,:));
    
    if ge(temp_vol, vol_min) && le(temp_vol, vol_max)
        
        c_vol = c_vol + 1;
        
        temp_zeta = reshape(zeta(i, :), dims(2:4));
        data_v(c_vol, :, :, :) = temp_zeta;
        info_v(c_vol) = info(i);
        
        clear temp_zeta
    end
    
    clear temp_vol
end

%% paths
% svmlin realted paths
time_stamp = datestr(now,'yyyymmddHHMMSS');

% mode name in the pre-cases as references
refmod = '20150917165945_ctmri_pre_20150917170318_1';
refmod_c = strsplit(refmod, '_');

% processing mode name (LS: lesion shink)
mod_name = [time_stamp '_postLS_' refmod];

% SVMLin path and pattern section name
svml_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015';
pattern = '2_simulations';

% refined trained classifer (weights files) with high accuracy
refwei_folder = 'b_pre_weights_ref';

% refined trained classifer (weights files) with high accuracy
refwei_mask_folder = 'b_pre_weights_ref_mask';

% p-values folder
thresh_folder = 'e_post_thresholds';

% full paths
refwei_path = [svml_path '/' pattern '/' refwei_folder];
refwei_mask_path = [svml_path '/' pattern '/' refwei_mask_folder];
pval_path = [svml_path '/' pattern '/' thresh_folder];

%% create data / results folders
% threshold evaluation
cd(pval_path)
if ~isdir(mod_name)
    system(['mkdir ' mod_name]);
end
cd(svml_path)

%% set the range of parameters realated to the refined trained classifer
% 'trdat_20150917165945_ctmri_pre_20150917170318_1_12_3_2_p89_n36_A2_W10000_U1.1_S53_R0.712_1'
hit = 1;
devia = 12;
norma = 3;
rev = 2;

% the ratio of test
tsRatio = 1;

%% variables
% treatment portions: this is the indicator that shows how much the lesion
% is eroded
shink_ratio = 0.05 : 0.05 : 0.95;

% threshold of simulated recovery
thresh_recov = 0.05 : 0.01 : 1;

%% remove fueatures: dimensionality reduction (general hit rate)
% data_v: refined dataset by the thresholds of lesion volumes
% info_v: refined information struture by the thresholds of lesion volumes
[data_red, info_red, fil_idx] = ion20150820125448_reduce_dimensionality_II(data_v, info_v, hit);

%% call function: ctgaze_kde
% the corrected centre of CT gazes on both eyes
% left: 2.3266; right: -3.1486
[corrgaze_ct_l, corrgaze_ct_r] = ctgaze_kde();

%% call function: mrigaze_kde
% the corrected centre of CT gazes on both eyes
% left: 7.0269; right: -3.4669
[corrgaze_mri_l, corrgaze_mri_r] = mrigaze_kde();

%% refine the gaze deviation of CT for the filtered cased above
for i = 1 : length(info_red)
    
    tempi = info_red(i);
    
    if tempi.ctgazeF == 1 && tempi.t2gazeF == 1
        ctgaze(i, 1) = tempi.ctresc_clmp_calc(1) - corrgaze_ct_l;
        ctgaze(i, 2) = tempi.ctresc_clmp_calc(2) - corrgaze_ct_r;
        
        mrigaze(i, 1) = tempi.t2resc_calc(1) - corrgaze_mri_l;
        mrigaze(i, 2) = tempi.t2resc_calc(2) - corrgaze_mri_r;
    else
        ctgaze(i, :) = [nan, nan];
        mrigaze(i, :) = [nan, nan];
    end
    
    clear tempi
end

%% collect the pre-yielded clssifers with best performance
cd([refwei_path '/' refmod]);
weif = dir('*.weights');
cd(svml_path);

%% load pre-optimised weighting masks
cd(refwei_mask_path);
load('op_wei_dat_0.01.mat')
weig_data = op_wei_dat;
cd(svml_path);

% normalise the weighting data to be in range of [-1 1]
for i = 1 : size(weig_data, 1)
    temp_wdat = weig_data(i, :) / max(abs(weig_data(i, :)));
    weig_data_norm(i, :) = temp_wdat(fil_idx);
    
    clear temp_wdat
end

%% variables: labelling
% vector: signs of labellled cases
label = [];

% labelling: -1 - both gazes deviated
%             1 - both gazes within the normal range
%             0 - unkonow gaze deviation for tranductive processing
%             9 - none of categories above
for i = 1 : length(ctgaze)
    
    if (le(ctgaze(i, 1), devia * -1) && le(ctgaze(i, 2), devia * -1)) ...
            && (le(mrigaze(i, 1), devia * -0.5) || le(mrigaze(i, 2), devia * -0.5))
        
        label(i) = -1;
    elseif le(abs(mrigaze(i, 1)), norma) && le(abs(mrigaze(i, 2)), norma)
        
        label(i) = 1;
    elseif isnan(ctgaze(i)) || isnan(mrigaze(i))
        
        label(i) = 0;
    else
        
        label(i) = 9;
    end
end

% refine label & data
labidx = find(label ~= 9);
sublabel = label(labidx);
subdata = data_red(labidx, :);

% indeices of positive, negative labels and as well as the unlabelled data
posidx = find(sublabel == 1);
negidx = find(sublabel == -1);
neuidx = find(sublabel == 0);

% remove the voxeles / cases with dedicated hit rate
subdata_pos = subdata(posidx, :);
subdata_neg = subdata(negidx, :);
subdata_neu = subdata(neuidx, :);

subdata_pos_s = sum(subdata_pos, 1);
subdata_neg_s = sum(subdata_neg, 1);
subdata_neu_s = sum(subdata_neu, 1);

subdata_pos(:, find(subdata_pos_s <= rev))=0;
subdata_neg(:, find(subdata_neg_s <= rev))=0;
subdata_neu(:, find(subdata_neu_s <= rev))=0;

re_pos = find(sum(subdata_pos,2) == 0);
re_neg = find(sum(subdata_neg,2) == 0);
re_neu = find(sum(subdata_neu,2) == 0);

posidx_rev = posidx;
posidx_rev(re_pos) = [];

negidx_rev = negidx;
negidx_rev(re_neg) = [];

neuidx_rev = neuidx;
neuidx_rev(re_neu) = [];

subdata_pos_rev = subdata_pos;
subdata_pos_rev(re_pos, :) = [];

subdata_neg_rev = subdata_neg;
subdata_neg_rev(re_neg, :) = [];

subdata_neu_rev = subdata_neu;
subdata_neu_rev(re_neu, :) = [];

tot_idx = [posidx_rev,negidx_rev,neuidx_rev];

sublabel_rev = sublabel(sort(tot_idx));
subdata_rev = subdata(sort(tot_idx),:);

posidx_rev2 = find(sublabel_rev == 1);
negidx_rev2 = find(sublabel_rev == -1);
neuidx_rev2 = find(sublabel_rev == 0);

% the number of positive, negative labelled data, and unlabelled data, respectively;
posnum = length(posidx_rev2);
negnum = length(negidx_rev2);
neunum = length(neuidx_rev2);

% standard number in order to select equal number of positive and negative data for training
if ge(posnum, negnum)
    stdnum = negnum;
else
    stdnum = posnum;
end

%% set up SVMLin modes
% a set of portions for simulating the treatments
% the set of refined weights files
for we = 1 : length(weif)
    
    fprintf('Going through weights > %d ... \n\n', we);
    
    % weight file and weight data in pair
    temp_weig = weif(we).name;
    temp_weig_dat = weig_data_norm(we, :);
    
    for sr = 1 : length(shink_ratio)
        
        fprintf('    Going through shink ratio > %d ... \n\n', sr);
        
        temp_shink = shink_ratio(sr);
        
        for th = 1 : length(thresh_recov)
            
            fprintf('        Going through threshold > %d ... \n\n', th);
            
            temp_thresh = thresh_recov(th);
            
            if gt(posnum, negnum)
                
                temp = randperm(posnum);
                temp = temp(1 : stdnum);
                
                % test positive labels
                ts_posidx = posidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % extra part of positive labels (posnum-negnum)
                posidx_cov = posidx_rev2;
                posidx_cov(temp) = [];
                
                % convert extra positive labels to neutral
                temp_sublabel_rev = sublabel_rev;
                temp_sublabel_rev(posidx_cov) = 0;
                
                % empty / clear temp
                clear temp
                
                temp = randperm(negnum);
                temp = temp(1 : stdnum);
                
                % training and test negative labels
                ts_negidx = negidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear temp
                clear temp
                
                % combined training and test indeices
                ts_idx=[ts_posidx, ts_negidx];
                
            elseif posnum == negnum
                
                % training and test positive labels
                temp = randperm(posnum);
                temp = temp(1 : stdnum);
                
                ts_posidx = posidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear temp
                clear temp
                
                % training and test negative labels
                temp = randperm(negnum);
                temp = temp(1 : stdnum);
                
                ts_negidx = negidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear temp
                clear temp
                
                temp_sublabel_rev = sublabel_rev;
                
                % combined training and test indeices
                ts_idx=[ts_posidx, ts_negidx];
                
            elseif lt(posnum, negnum)
                
                % training and test positive labels
                temp = randperm(posnum);
                temp = temp(1:stdnum);
                
                ts_posidx = posidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear temp
                clear temp
                
                % training and test negative labels
                temp = randperm(negnum);
                temp = temp(1 : stdnum);
                
                ts_negidx = negidx_rev2(temp(1 : floor(stdnum * tsRatio)));
                
                % extra part of negative labels (negnum-posnum)
                negind_cov = negidx_rev2;
                negind_cov(temp) = [];
                
                % empty / clear temp
                clear temp
                
                temp_sublabel_rev = sublabel_rev;
                temp_sublabel_rev(negind_cov) = 0;
                
                % combined training and test indeices
                ts_idx = [ts_posidx, ts_negidx];
            end
            
            % test data with the extra column of normalised age as a weighting factor
            tslab = temp_sublabel_rev(ts_idx);
            tsdat = subdata_rev(ts_idx, :);
            
            % simulating treated cases (in term of lesion shinkage)
            temp_idx = randperm(length(tslab));
            temp_treated_idx = temp_idx(1 : ceil(length(tslab) / 2));
            temp_treated_dat = tsdat(temp_treated_idx, :);
            
            % test label after treatments
            tslab_treat = tslab;
            
            % shink the lesions by dedicated ratios
            recov_cnt = 0;
            
            if ~eq(temp_shink, 0)
                
                for tr = 1 : length(temp_treated_idx)
                    
                    temp_input = temp_treated_dat(tr, :);
                    temp_output = simu_lesion_erosion_20151016115549(temp_input, temp_shink);
                    
                    temp_weig_dat_overlap = temp_weig_dat(find(temp_output == 1));
                    
                    % treated as recovered
                    if lt(max(abs(temp_weig_dat_overlap)), thresh_recov)
                        
                        if eq(tslab_treat(temp_treated_idx(tr)), -1)
                            recov_cnt = recov_cnt + 1;
                        end
                        
                        % key area in weighting mask erosed: recovered
                        tslab_treat(temp_treated_idx(tr)) = 1;
                    end
                    
                    clear temp_input temp_output
                    clear temp_weig_dat_partial
                end
            end
            
            thresh_eva(we, sr, th) = recov_cnt;
            
            clear temp_idx temp_treated_idx temp_treated_dat
            clear posidx_cov negidx_cov
            clear ts_* tsdat* tslab*
            clear temp_sublabel_rev fid* valid validT accu accuT
            clear temp_thresh
        end
        clear temp_shink
    end
    clear temp_weig
end

