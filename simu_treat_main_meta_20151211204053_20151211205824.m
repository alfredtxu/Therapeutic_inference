% Summary. using best trained classifier in the 'precases' as training results to yield outcomes.
%          simulating treatment on partial cases to see how senstive the treated cases affecting the accuracy.
%
% Author. Tianbo XU
%
% Institution. Institute of Neurology, UCL
%
% init. 11.12.2015
%
% comm. 11.12.2015
%       * select half of test data as treating group, and shink by
%       dedicated amount;
%       * the other half of lesions as untreating group and leave them as
%       what they were;
% 
%       * compare predicted outcome with and without lesion shrinkage by runnign
%         against already trained classifier (from which the weights image as created)
% 
%       * put predicted treatment responders and non-treatment esponders in glmfit model
% 
%       11.12.2015
%       * separate p-value results by each weights file
%         
%       11.12.2015
%       * exp(beta): odds-ratio in Logistic regression

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
    
    tmp_vol = sum(zeta(i,:));
    
    if ge(tmp_vol, vol_min) && le(tmp_vol, vol_max)
        
        c_vol = c_vol + 1;
        
        tmp_zeta = reshape(zeta(i, :), dims(2:4));
        data_v(c_vol, :, :, :) = tmp_zeta;
        
        vol_v(c_vol) = tmp_vol;
        info_v(c_vol) = info(i);
        
        clear tmp_zeta
    end
    
    clear tmp_vol
end

%% paths
% svmlin realted paths
time_stamp = datestr(now,'yyyymmddHHMMSS');

% mode name in the pre-cases as references
ref_mod = '20150917165945_ctmri_pre_20151126182450_1';

% processing mode name (LS: lesion shink)
modname = [time_stamp '_postLS_' ref_mod];

% SVMLin path and pattern section name
svml_path = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015';

% sub-folders -> simulations
simu_folder = '2_simulations';

% sub-folders -> simulations -> processing data folder
data_folder = 'a_proce_data';

% sub-folders -> simulations -> trained classifer (weights files) in highest accuracy
refwei_folder = 'b_pre_weights_ref';

% sub-folders -> simulations -> p-values folder
pval_folder = 'd_post_beta_lesion';

% full paths
data_path = [svml_path '/' simu_folder '/' data_folder];
refwei_path = [svml_path '/' simu_folder '/' refwei_folder];
pval_path = [svml_path '/' simu_folder '/' pval_folder];

%% create data / results folders
% training / test data files
cd(data_path)
if ~isdir(modname)
    system(['mkdir ' modname]);
end
cd(svml_path)

% p-value
cd(pval_path)
if ~isdir(modname)
    system(['mkdir ' modname]);
end
cd(svml_path)

%% set the range of parameters realated to the refined trained classifer
% 'trdat_20150917165945_ctmri_pre_20150917170318_1_12_3_2_p89_n36_A2_W10000_U1.1_S53_R0.712_1'
% 'trdat_20150917165945_ctmri_pre_20151126182450_1_8.75_3.5_2_p107_n53_A2_W1000_U1.1_S54_R0.66875_1'
hit = 1;
devia = 8.75;
norma = 3.5;
rev = 2;

algor = 2;
wW = 10^3;
uU = 1.1;
portion = 1;

%% variables
% the number of iterations
times = 600;

% the ratio of test
tsRatio = 1;

% treatment portions: this is the indicator that shows how much the lesion
% is eroded
shink_ratio = 0.1 : 0.1 : 0.90;

% glmfit distribution
distr = 'poisson';

%% remove fueatures: dimensionality reduction (general hit rate)
% data_v: refined dataset by the thresholds of lesion volumes
% info_v: refined information struture by the thresholds of lesion volumes
[data_red, info_red, vol_red] = ion20151118175945_reduce_dimensionality_vol(data_v, info_v, vol_v, hit);

% change volume data to be a vector
vol_red = vol_red';

%% call function: ctgaze_kde
% the corrected centre of CT gazes on both eyes
% left: 2.3266; right: -3.1486
[corrgaze_ct_r, corrgaze_ct_l] = ctgaze_kde();

%% call function: mrigaze_kde
% the corrected centre of CT gazes on both eyes
% left: 7.0269; right: -3.4669
[corrgaze_mri_r, corrgaze_mri_l] = mrigaze_kde();

%% refine the gaze deviation of CT for the filtered cased above
for i = 1 : length(info_red)
    
    tempi = info_red(i);
    
    if tempi.ctgazeF == 1 && tempi.t2gazeF == 1
        ctgaze(i, 1) = tempi.ctresc_clmp_calc(1) - corrgaze_ct_r;
        ctgaze(i, 2) = tempi.ctresc_clmp_calc(2) - corrgaze_ct_l;
        
        mrigaze(i, 1) = tempi.t2resc_calc(1) - corrgaze_mri_r;
        mrigaze(i, 2) = tempi.t2resc_calc(2) - corrgaze_mri_l;
    else
        ctgaze(i, :) = [nan, nan];
        mrigaze(i, :) = [nan, nan];
    end
    
    clear tempi
end

%% collect the pre-yielded clssifers with best performance
cd([refwei_path '/' ref_mod]);
weif = dir('*.weights');
cd(svml_path);

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
subvol = vol_red(labidx, :);

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
subvol_rev = subvol(sort(tot_idx),:);

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
    
    temp_weig = weif(we).name;
    
    for te = 1 : length(shink_ratio)
        
        temp_shink = shink_ratio(te);
            
            % looping: iterations
            for trial = 1 : times
                
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
                
                tsvol = subvol_rev(ts_idx, :);
                
                % simulating treated cases (in term of lesion shinkage)
                temp_rand_idx = randperm(length(tslab));
                temp_treated_idx = temp_rand_idx(1 : ceil(length(tslab) / 2));
                
                % test data after treatments
                tsdat_treat = tsdat;
                
                % shink the lesions by dedicated ratios
                if ~eq(temp_shink, 0)
                    
                    for tr = 1 : length(temp_treated_idx)
                        
                        % simulation of lesion shinkage
                        temp_input = tsdat_treat(temp_treated_idx(tr), :);
                        temp_output = simu_lesion_erosion_20151016115549(temp_input, temp_shink);
                        tsdat_treat(temp_treated_idx(tr), :) = temp_output;
                        
                        clear temp_input temp_output
                    end
                end
                
                treat_val = zeros(1, length(tslab));
                treat_val(temp_treated_idx) = 1;
                
                % file: test data
                tsdat_fn = ['tsdat_' modname '_' num2str(hit) ...
                                 '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                 '_p' num2str(posnum) '_n' num2str(negnum) ...
                                 '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                 '_S' num2str(sS) '_R' num2str(rR) ...
                                 '_wei' num2str(we) '_sr' num2str(temp_shink) ...
                                 '_' num2str(trial)];
                
                tsdat_fullfile = [data_path '/' modname '/' tsdat_fn];
                
                % file: test treated data
                tsdatT_fn = ['tsdatT_' modname '_' num2str(hit) ...
                                   '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                   '_p' num2str(posnum) '_n' num2str(negnum) ...
                                   '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                   '_S' num2str(sS) '_R' num2str(rR) ...
                                   '_wei' num2str(we) '_sr' num2str(temp_shink) ...
                                   '_' num2str(trial)];
                
                tsdatT_fullfile = [data_path '/' modname '/' tsdatT_fn];
                
                % file: test label
                tslab_fn = ['tslab_' modname '_' num2str(hit) ...
                                 '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                                 '_p' num2str(posnum) '_n' num2str(negnum) ...
                                 '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                                 '_S' num2str(sS) '_R' num2str(rR) ...
                                 '_wei' num2str(we) '_sr' num2str(temp_shink) ...
                                 '_' num2str(trial)];
                
                tslab_fullfile = [data_path '/' modname '/' tslab_fn];
                
                % open the data files for writing
                fid1 = fopen(tsdat_fullfile, 'w+');
                fid2 = fopen(tsdatT_fullfile, 'w+');
                fid3 = fopen(tslab_fullfile, 'w+');
                
                % writing test data files
                for i = 1 : length(tslab)
                    
                    % writing test data
                    tempVol = find(tsdat(i, :));
                    
                    datastr = '';
                    datastr_T = '';
                
                    for j = 1 : length(tempVol)
                        if j == 1
                            datastr = [mat2str(tempVol(j)) ':' num2str(tsdat(i, tempVol(j)))];
                        else
                            datastr = [datastr ' ' mat2str(tempVol(j)) ':' num2str(tsdat(i, tempVol(j)))];
                        end
                    end
                    
                    fprintf(fid1, '%s\n', datastr);
                    
                    % writing test treated data
                    tempVol_T = find(tsdat_treat(i, :));
                    
                    for j = 1 : length(tempVol_T)
                        if j == 1
                            datastr_T = [mat2str(tempVol_T(j)) ':' num2str(tsdat_treat(i, tempVol_T(j)))];
                        else
                            datastr_T = [datastr_T ' ' mat2str(tempVol_T(j)) ':' num2str(tsdat_treat(i, tempVol_T(j)))];
                        end
                    end
                    
                    fprintf(fid2, '%s\n', datastr_T);
                    
                    % writing test label
                    fprintf(fid3, '%s\n', num2str(tslab(i)));
                    
                    clear tempVol tempVol_T
                end
                
                % close file ids after writing
                fclose(fid1);
                fclose(fid2);
                fclose(fid3);
                
                % execute SVMLin modes under SVMLin package path
                cd(svml_path);
                
                % SVMLin test: original data
                system(['./svmlin -f ' refwei_path '/' ref_mod '/' temp_weig ...
                                   ' ' simu_folder '/' data_folder '/' modname '/' tsdat_fn ...
                                   ' ' simu_folder '/' data_folder '/' modname '/' tslab_fn]);
                
                % load the outputs and check the performance
                valid = sign(dlmread([tsdat_fn '.outputs']));
                
                % SVMLin test: treated test data
                system(['./svmlin -f ' refwei_path '/' ref_mod '/' temp_weig ...
                                   ' ' simu_folder '/' data_folder '/' modname '/' tsdatT_fn ...
                                   ' ' simu_folder '/' data_folder '/' modname '/' tslab_fn]);
                
                % load the outputs and check the performance
                valid_T = sign(dlmread([tsdatT_fn '.outputs']));
                
                % display the results
                clc
                accu = sum(tslab == valid') ./ length(tslab);
                accuT = sum(tslab == valid_T') ./ length(tslab);
                fprintf('Comparing the accuracy non-treatment: %s vs treated: %s\n\n', num2str(accu), num2str(accuT));
                
                % use the prediction in a glm model that helps us isolate the treatment effect
                [b, dev, stats] = glmfit([valid_T(:) valid(:)], treat_val(:), distr);
                
                beta_v{trial, te} = b;
                dev_v{trial, te} = dev;
                stats_v{trial, te} = stats;
                
                clear b dev stats
                
                % use the prediction in a glm model that helps us isolate the treatment + volume effect
                [b, dev, stats] = glmfit([valid_T(:) valid(:) tsvol(:)], treat_val(:), distr);
                
                beta_vm{trial, te} = b;
                dev_vm{trial, te} = dev;
                stats_vm{trial, te} = stats;
                
                clear b dev stats
                
                % use the prediction in a glm model that helps us isolate the volume effect
                [b, dev, stats] = glmfit([valid_T(:) tsvol(:)], treat_val(:), distr);
                
                beta_m{trial, te} = b;
                dev_m{trial, te} = dev;
                stats_m{trial, te} = stats;
                
                clear b dev stats
                
                % compare without the prediction
                [b, dev, stats] = glmfit(valid_T(:), treat_val(:), distr);
                
                beta{trial, te} = b;
                dev_plain{trial, te} = dev;
                stats_plain{trial, te} = stats;
                
                clear b dev stats
                
                % clear / move the weights & outputs files
                svmfiles1 = dir(['*_' modname '_*.weights']);
                svmfiles2 = dir(['*_' modname '_*.outputs']);
                
                for sf1 = 1 : length(svmfiles1)
                    system(['rm ' svmfiles1(sf1).name]);
                end
                
                for sf2 = 1 : length(svmfiles2)
                    system(['rm ' svmfiles2(sf2).name]);
                end
                
                clear temp_rand_idx temp_treated_idx
                clear posidx_cov negidx_cov
                clear ts_* tsdat* tslab*
                clear temp_sublabel_rev fid* valid validT accu accuT
            end
        
        clear temp_shink
    end
    
    % save the whole p-values
    save([pval_path '/' modname '/beta_v_' num2str(we) '.mat'], 'beta_v');
    save([pval_path '/' modname '/beta_vm_' num2str(we) '.mat'], 'beta_vm');
    save([pval_path '/' modname '/beta_m_' num2str(we) '.mat'], 'beta_m');
    save([pval_path '/' modname '/beta_' num2str(we) '.mat'], 'beta');
    
    save([pval_path '/' modname '/dev_v_' num2str(we) '.mat'], 'dev_v');
    save([pval_path '/' modname '/dev_vm_' num2str(we) '.mat'], 'dev_vm');
    save([pval_path '/' modname '/dev_m_' num2str(we) '.mat'], 'dev_m');
    save([pval_path '/' modname '/dev_' num2str(we) '.mat'], 'dev_plain');
    
    save([pval_path '/' modname '/stats_v_' num2str(we) '.mat'], 'stats_v');
    save([pval_path '/' modname '/stats_vm_' num2str(we) '.mat'], 'stats_vm');
    save([pval_path '/' modname '/stats_m_' num2str(we) '.mat'], 'stats_m');
    save([pval_path '/' modname '/stats_' num2str(we) '.mat'], 'stats_plain');
    
    clear beta_v dev_v stats_v
    clear beta_vm dev_vm stats_vm
    clear beta_m dev_m stats_m
    clear beta dev_plain stats_plain
    
    clear temp_weig
end


