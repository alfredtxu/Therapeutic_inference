% Summary. using best trained classifier in the 'precases' as training results to yield outcomes.
%          simulating treatment on partial cases to see how senstive the treated cases affecting the accuracy.
%
% Author. Tianbo XU
%
% Institution. Institute of Neurology, UCL
%
% init. 16.10.2015
%
% comm. 21.10.2015
%       * involve pre-optimised weighting masks
%       * decide empirical threshold of overlap between shinked lesion and weighting mask
%         --> remaining sick vs recovered

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

% processing data folder
data_folder = 'a_proce_data';

% refined trained classifer (weights files) with high accuracy
refwei_folder = 'b_pre_weights_ref';

% refined trained classifer (weights files) with high accuracy
refwei_mask_folder = 'b_pre_weights_ref_mask';

% p-values folder
pval_folder = 'd_post_pval_lesion';

% full paths
data_path = [svml_path '/' pattern '/' data_folder];
refwei_path = [svml_path '/' pattern '/' refwei_folder];
refwei_mask_path = [svml_path '/' pattern '/' refwei_mask_folder];
pval_path = [svml_path '/' pattern '/' pval_folder];

%% create data / results folders
% training / test data files
cd(data_path)
if ~isdir(mod_name)
    system(['mkdir ' mod_name]);
end
cd(svml_path)

% p-value
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

algor = 2;
wW = 10^4;
uU = 1.1;
portion = 1;

%% variables
% the number of iterations
times = 10;

% the ratio of test
tsRatio = 1;

% treatment portions: this is the indicator that shows how much the lesion
% is eroded
shink_ratio = 0.1 : 0.1 : 0.9;

% threshold of simulated recovery
thresh_recov = 0.1 : 0.03 : 0.4;

% glmfit distribution
distr = 'poisson';

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

% -R  positive class fraction of unlabeled data  (0.5)
rR = posnum ./ (posnum+negnum);

% -S  maximum number of switches in TSVM (default 0.5*number of unlabeled examples)
sS = floor(portion .* abs(posnum-negnum));

%% set up SVMLin modes
% a set of portions for simulating the treatments
% the set of refined weights files
for we = 1 : length(weif)
    
    % weight file and weight data in pair
    temp_weig = weif(we).name;
    temp_weig_dat = weig_data_norm(we, :);
    
    for sr = 1 : length(shink_ratio)
        
        temp_shink = shink_ratio(sr);
        
        for th = 1 : length(thresh_recov)
            
            temp_thresh = thresh_recov(th);
            
            % looping: iterations
            for iter = 1 : times
                
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
                        
                        temp_weig_dat_partial = temp_weig_dat(find(temp_output == 1));
                        
                        if lt(max(temp_weig_dat_partial), thresh_recov*-1)
                            
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
                
                treat_val = zeros(1, length(tslab));
                treat_val(temp_idx) = 1;
                
                clear temp_idx temp_treated_idx temp_treated_dat
                
                % file: test data
                tsdat_fn = ['tsdat_' mod_name '_' num2str(hit) ...
                                 '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                 '_p' num2str(posnum) '_n' num2str(negnum) ...
                                 '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                 '_S' num2str(sS) '_R' num2str(rR) ...
                                 '_wei' num2str(we) '_tr' num2str(temp_shink) '_th' num2str(temp_thresh) ...
                                 '_recov' num2str(recov_cnt) ...
                                 '_' num2str(iter)];
                
                tsdat_fullfile = [data_path '/' mod_name '/' tsdat_fn];
                
                % file: test label
                tslab_fn = ['tslab_' mod_name '_' num2str(hit) ...
                                 '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                 '_p' num2str(posnum) '_n' num2str(negnum) ...
                                 '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                 '_S' num2str(sS) '_R' num2str(rR) ...
                                 '_wei' num2str(we) '_tr' num2str(temp_shink) '_th' num2str(temp_thresh) ...
                                 '_recov' num2str(recov_cnt) ...
                                 '_' num2str(iter)];
                
                tslab_fullfile = [data_path '/' mod_name '/' tslab_fn];
                
                % file: test label (treated)
                tslabT_fn = ['tslabT_' mod_name '_' num2str(hit) ...
                                   '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                   '_p' num2str(posnum) '_n' num2str(negnum) ...
                                   '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                   '_S' num2str(sS) '_R' num2str(rR) ...
                                   '_wei' num2str(we) '_tr' num2str(temp_shink) '_th' num2str(temp_thresh) ...
                                   '_recov' num2str(recov_cnt) ...
                                   '_' num2str(iter)];
                
                tslabT_fullfile = [data_path '/' mod_name '/' tslabT_fn];
                
                % open the data files for writing
                fid1 = fopen(tsdat_fullfile, 'w+');
                fid2 = fopen(tslab_fullfile, 'w+');
                fid3 = fopen(tslabT_fullfile, 'w+');
                
                % writing test data files
                for i = 1 : length(tslab)
                    
                    tempVol = find(tsdat(i, :));
                    
                    for j = 1 : length(tempVol)
                        if j == 1
                            datastr = [mat2str(tempVol(j)) ':' num2str(tsdat(i, tempVol(j)))];
                        else
                            datastr = [datastr ' ' mat2str(tempVol(j)) ':' num2str(tsdat(i, tempVol(j)))];
                        end
                    end
                    
                    fprintf(fid1, '%s\n', datastr);
                    fprintf(fid2, '%s\n', num2str(tslab(i)));
                    fprintf(fid3, '%s\n', num2str(tslab_treat(i)));
                    
                    clear tempVol datastr
                end
                
                % close file ids after writing
                fclose(fid1);
                fclose(fid2);
                fclose(fid3);
                
                % execute SVMLin modes under SVMLin package path
                cd(svml_path);
                
                % SVMLin test: simulated test label
                system(['./svmlin -f ' refwei_path '/' refmod '/' temp_weig ...
                    ' ' pattern '/' data_folder '/' mod_name '/' tsdat_fn ...
                    ' ' pattern '/' data_folder '/' mod_name '/' tslabT_fn]);
                
                % load the outputs and check the performance
                validT = sign(dlmread([tsdat_fn '.outputs']));
                system(['mv ' tsdat_fn '.outputs ' tsdat_fn(1:5) 'T' tsdat_fn(6:end) '.outputs']);
                
                % SVMLin test: original test label
                system(['./svmlin -f ' refwei_path '/' refmod '/' temp_weig ...
                    ' ' pattern '/' data_folder '/' mod_name '/' tsdat_fn ...
                    ' ' pattern '/' data_folder '/' mod_name '/' tslab_fn]);
                
                % load the outputs and check the performance
                valid = sign(dlmread([tsdat_fn '.outputs']));
                
                % display the results
                accu = sum(tslab == valid') ./ length(tslab);
                accuT = sum(tslab_treat == validT') ./ length(tslab_treat);
                
                clc
                fprintf('*** %s: recovered ... \n', num2str(recov_cnt));
                fprintf('Comparing the accuracy non-treatment: %s vs treated: %s\n\n', num2str(accu), num2str(accuT));
                
                % use the prediction in a glm model that helps us isolate the treatment effect
                [b, dev, stats] = glmfit([tslab_treat(:) valid(:)], treat_val(:), distr);
                
                pv1(iter, th, sr, we)=stats.p(2);
                pv2(iter, th, sr, we)=stats.p(3);
                
                disp(['pv1: ' num2str(stats.p(2))])
                disp(['pv2: ' num2str(stats.p(3))])
                
                clear b dev stats
                
                % compare without the prediction
                [b, dev, stats] = glmfit(tslab_treat(:), treat_val(:), distr);
                
                p(iter, th, sr, we)=stats.p(2);
                
                disp(['p: ' num2str(stats.p(2))])
                
                clear b dev stats
                
                % clear / move the weights & outputs files
                svmfiles1 = dir(['*_' mod_name '_*.weights']);
                svmfiles2 = dir(['*_' mod_name '_*.outputs']);
                
                for sf1 = 1 : length(svmfiles1)
                    system(['rm ' svmfiles1(sf1).name]);
                end
                
                for sf2 = 1 : length(svmfiles2)
                    system(['rm ' svmfiles2(sf2).name]);
                end
                
                clear posidx_cov negidx_cov
                clear ts_* tsdat* tslab*
                clear temp_sublabel_rev fid* valid validT accu accuT
            end
            
            clear temp_thresh
        end
        
        clear temp_shink
    end
    
    clear temp_weig
end

% save the whole p-values
save([pval_path '/' mod_name '/pv1.mat'], 'pv1');
save([pval_path '/' mod_name '/pv2.mat'], 'pv2');
save([pval_path '/' mod_name '/p.mat'], 'p');