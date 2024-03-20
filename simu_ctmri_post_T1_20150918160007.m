% Summary: using best trained classifier in the 'precases' as training results to yield outcomes. 
%          simulating treatment on partial cases to see how senstive the treated cases affecting the accuracy. 
%
% Author. Tianbo XU
%
% Organisation. Institute of Neurology, UCL
%
% init. 14.09.2015
%
% comm.
%       * select treating cases from the entire test data
%       * loop: portions - weights - iterations
% upd. 16.09.2015
% 
% comm. 
%       * MRI gaze data involved
%  

clc
clear
close all

%% load the prepared data matrices
load('zeta1333_6mm_bin.mat');
data = zeta1333_6mm_bin;

dims = size(data);
zeta = reshape(data, [dims(1) prod(dims(2:4))]);

%% load information struct of patients
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

%% svmlin realted paths
time_stamp = datestr(now,'yyyymmddHHMMSS');

% mode name in the pre-cases as references
refmod = '20150916164125_ctmri_pre_20150916164421_1';
refmod_c = strsplit(refmod, '_');

% processing mode name
modname = [time_stamp '_post_' refmod_c{1}];

% SVMLin path and pattern section name
svmlinp = '/media/txu/2TB_1/z_matlab_workspace/pro_Therapeutic/f1_svmlin_2015';
pattern = '2_simu_gaze_recov';

% processing data folder
dataf = 'a_proce_data';

% refined trained classifer (weights files) with high accuracy
refweif = 'b_pre_weights_ref';

% p-values folder
pvalf = 'd_post_pval';

% full paths
datap = [svmlinp '/' pattern '/' dataf];
refweip = [svmlinp '/' pattern '/' refweif];
pvalp = [svmlinp '/' pattern '/' pvalf];

system(['mkdir ' datap '/' modname]);
system(['mkdir ' pvalp '/' modname]);

%% set the range of parameters realated to the refined trained classifer
% 'trdat_20150916164421_ctLvN_min1_max200_pre_1_15_3_2_p89_n27_A2_W10000_U1.1_S62_R0.76724_X'
hit = 1;
devia = 15;
norma = 3;
rev = 2;

algor = 2;
wW = 10^4;
uU = 1.1;
portion = 1;

% the number of iterations
times = 300;

% the ratio of test
tsRatio = 1;

% treatment portions
treatp = 0.1 : 0.1 : 0.9;

% significat p-value
% sp = 0.05;

% glmfit distribution
distr = 'poisson';

%% remove fueatures: dimensionality reduction (general hit rate)
% data_v: refined dataset by the thresholds of lesion volumes
% info_v: refined information struture by the thresholds of lesion volumes
[data_red, info_red] = ion20150820125448_reduce_dimensionality(data_v, info_v, hit);

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
    
    tmpi = info_red(i);
    
    if tmpi.ctgazeF == 1 && tmpi.t2gazeF == 1
        ctgaze(i, 1) = tmpi.ctresc_clmp_calc(1) - corrgaze_ct_l;
        ctgaze(i, 2) = tmpi.ctresc_clmp_calc(2) - corrgaze_ct_r;
        
        mrigaze(i, 1) = tmpi.t2resc_calc(1) - corrgaze_mri_l;
        mrigaze(i, 2) = tmpi.t2resc_calc(2) - corrgaze_mri_r;
    else
        ctgaze(i, :) = [nan, nan];
        mrigaze(i, :) = [nan, nan];
    end
    
    clear tmpi
end

% collect the pre-yielded clssifer
cd([refweip '/' refmod]);
weif = dir('*.weights');
cd(svmlinp);

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
for te = 1 : length(treatp)
    
    tmp_treat = treatp(te);
    
    % the set of refined weights files
    for we = 1 : length(weif)
        
        tmp_wei = weif(we).name;

        % looping: iterations
        for iter = 1 : times
            
            if gt(posnum, negnum)
                
                tmp = randperm(posnum);
                tmp = tmp(1 : stdnum);
                
                % test positive labels
                ts_posidx = posidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % extra part of positive labels (posnum-negnum)
                posidx_cov = posidx_rev2;
                posidx_cov(tmp) = [];
                
                % convert extra positive labels to neutral
                tmp_sublabel_rev = sublabel_rev;
                tmp_sublabel_rev(posidx_cov) = 0;
                
                % empty / clear tmp
                clear tmp
                
                tmp = randperm(negnum);
                tmp = tmp(1 : stdnum);
                
                % training and test negative labels
                ts_negidx = negidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear tmp
                clear tmp
                
                % combined training and test indeices
                ts_idx=[ts_posidx, ts_negidx];
                
            elseif posnum == negnum
                
                % training and test positive labels
                tmp = randperm(posnum);
                tmp = tmp(1 : stdnum);
                
                ts_posidx = posidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear tmp
                clear tmp
                
                % training and test negative labels
                tmp = randperm(negnum);
                tmp = tmp(1 : stdnum);
                
                ts_negidx = negidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear tmp
                clear tmp
                
                tmp_sublabel_rev = sublabel_rev;
                
                % combined training and test indeices
                ts_idx=[ts_posidx, ts_negidx];
                
            elseif lt(posnum, negnum)
                
                % training and test positive labels
                tmp = randperm(posnum);
                tmp = tmp(1:stdnum);
                
                ts_posidx = posidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % empty / clear tmp
                clear tmp
                
                % training and test negative labels
                tmp = randperm(negnum);
                tmp = tmp(1 : stdnum);
                
                ts_negidx = negidx_rev2(tmp(1 : floor(stdnum * tsRatio)));
                
                % extra part of negative labels (negnum-posnum)
                negind_cov = negidx_rev2;
                negind_cov(tmp) = [];
                
                % empty / clear tmp
                clear tmp
                
                tmp_sublabel_rev = sublabel_rev;
                tmp_sublabel_rev(negind_cov) = 0;
                
                % combined training and test indeices
                ts_idx = [ts_posidx, ts_negidx];
            end
            
            % test data with the extra column of normalised age as a weighting factor
            tslab = tmp_sublabel_rev(ts_idx);
            tsdat = subdata_rev(ts_idx, :);
            
            % simulating treated cases
            tmplab = randperm(length(tslab));
            tmplab = tmplab(1:ceil(length(tslab) / 2));
            
            tslab_treat = tslab;
            
            if ~eq(tmp_treat, 0)
                tmp_treated = tmplab(1 : ceil(length(tmplab) * tmp_treat));
                tslab_treat(tmp_treated) = 1;    
            end
            
            treat_val = zeros(1, length(tslab));
            treat_val(tmplab) = 1;
                                    
            clear tmplab tmp_treated
            
            % file: test data
            tsdat_fn = ['tsdat_' modname '_' num2str(hit) ...
                             '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                             '_p' num2str(posnum) '_n' num2str(negnum) ...
                             '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                             '_S' num2str(sS) '_R' num2str(rR) ...
                             '_wei' num2str(we) '_tr' num2str(tmp_treat) ...
                             '_' num2str(iter)];
            
            tsdat_fullfile = [datap '/' modname '/' tsdat_fn];
            
            % file: test label
            tslab_fn = ['tslab_' modname '_' num2str(hit) ...
                             '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                             '_p' num2str(posnum) '_n' num2str(negnum) ...
                             '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                             '_S' num2str(sS) '_R' num2str(rR) ...
                             '_wei' num2str(we) '_tr' num2str(tmp_treat) ...
                             '_' num2str(iter)];
            
            tslab_fullfile = [datap '/' modname '/' tslab_fn];
            
            % file: test label (treated)
            tslabT_fn = ['tslabT_' modname '_' num2str(hit) ...
                             '_' num2str(devia) '_' num2str(norma) '_' num2str(rev) ...
                             '_p' num2str(posnum) '_n' num2str(negnum) ...
                             '_A' num2str(algor) '_W' num2str(wW) '_U' num2str(uU) ...
                             '_S' num2str(sS) '_R' num2str(rR) ...
                             '_wei' num2str(we) '_tr' num2str(tmp_treat) ...
                             '_' num2str(iter)];
            
            tslabT_fullfile = [datap '/' modname '/' tslabT_fn];
            
            % open the data files for writing
            fid1 = fopen(tsdat_fullfile, 'w+');
            fid2 = fopen(tslab_fullfile, 'w+');
            fid3 = fopen(tslabT_fullfile, 'w+');
            
            % writing test data files
            for i = 1 : length(tslab)
                
                tmpVol = find(tsdat(i, :));
                
                for j = 1 : length(tmpVol)
                    if j == 1
                        datastr = [mat2str(tmpVol(j)) ':' num2str(tsdat(i, tmpVol(j)))];
                    else
                        datastr = [datastr ' ' mat2str(tmpVol(j)) ':' num2str(tsdat(i, tmpVol(j)))];
                    end
                end
                
                fprintf(fid1, '%s\n', datastr);
                fprintf(fid2, '%s\n', num2str(tslab(i)));
                fprintf(fid3, '%s\n', num2str(tslab_treat(i)));
                
                clear tmpVol datastr
            end
            
            % close file ids after writing
            fclose(fid1);
            fclose(fid2);
            fclose(fid3);
            
            % execute SVMLin modes under SVMLin package path
            cd(svmlinp);
           
            % SVMLin test: simulated test label
            system(['./svmlin -f ' refweip '/' refmod '/' tmp_wei ...
                               ' ' pattern '/' dataf '/' modname '/' tsdat_fn ...
                               ' ' pattern '/' dataf '/' modname '/' tslabT_fn]);

            % load the outputs and check the performance
            validT = sign(dlmread([tsdat_fn '.outputs']));
            system(['mv ' tsdat_fn '.outputs ' tsdat_fn(1:5) 'T' tsdat_fn(6:end) '.outputs']);
            
            % SVMLin test: original test label
            system(['./svmlin -f ' refweip '/' refmod '/' tmp_wei ...
                               ' ' pattern '/' dataf '/' modname '/' tsdat_fn ...
                               ' ' pattern '/' dataf '/' modname '/' tslab_fn]);
                            
            % load the outputs and check the performance
            valid = sign(dlmread([tsdat_fn '.outputs']));
            
            % display the results
            accu = sum(tslab == valid') ./ length(tslab);
            accuT = sum(tslab_treat == validT') ./ length(tslab_treat);
            
            clc
            fprintf('Comparing the accuracy non-treatment: %s vs treated: %s\n\n', num2str(accu), num2str(accuT));
            
            % use the prediction in a glm model that helps us isolate the treatment effect
            [b, dev, stats] = glmfit([tslab_treat(:) valid(:)], treat_val(:), distr);
            
            pv1(iter, we, te)=stats.p(2);
            pv2(iter, we, te)=stats.p(3);
            
            disp(['pv1: ' num2str(stats.p(2))])
            disp(['pv2: ' num2str(stats.p(3))])
            
            clear b dev stats
            
            % compare without the prediction
            [b, dev, stats] = glmfit(tslab_treat(:), treat_val(:), distr);
            
            p(iter, we, te)=stats.p(2);
            
            disp(['p: ' num2str(stats.p(2))])
            
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
            
            clear posidx_cov negidx_cov
            clear ts_* tsdat* tslab*
            clear tmp_sublabel_rev fid* valid validT accu accuT
        end
           
        clear tmp_wei
    end
    
    clear tmp_treat
end

% save the whole p-values
save([pvalp '/' modname '/pv1.mat'], 'pv1');
save([pvalp '/' modname '/pv2.mat'], 'pv2');
save([pvalp '/' modname '/p.mat'], 'p');

