% summary: classify the relationshops between CT gaze deviation and the location of lesions via SVMLin
% ** for each patient, the gaze deviation is treated as the mean of left and right deviation;
%
% ** Left deviation v.s. Right deviation
%
% Tianbo XU
% init.date: 01.05.2015
function ion20150501152255_ctgaze_mod_LvR()

%% comments
% I. structure of stroke info:
%    serial
%    lat
%    coreid
%    zetaname
%    zetahead
%    zetaname_6mm
%    zetahead_6mm
%    zetadata_6mm_6mm
%    ctgazeF
%    ctgazeDupF
%    t2gazeF
%    t2gazeDupF
%    duration
%    ctname
%    ctmat
%    ctres
%    ctresc
%    ctresc_clmp_calc
%    t2name
%    t2mat
%    t2res
%    t2resc
%    t2resc_calc
%    age
%
% II. Fast  Linear  SVM  Solvers  for Supervised  and Semi-supervised  Learning
% Usage
%      Train: svmlin [options] training_examples training_labels
%       Test: svmlin -f weights_file test_examples
%   Evaluate: svmlin -f weights_file test_examples test_labels
%
%  Options:
%  -A  algorithm : set algorithm (default 1)
%          0 -- Regularized Least Squares (RLS) Classification
%          1 -- SVM (L2-SVM-MFN) (default choice)
%          2 -- Multi-switch Transductive SVM (using L2-SVM-MFN)
%          3 -- Deterministic Annealing Semi-supervised SVM (using L2-SVM-MFN)
%  -W  regularization parameter lambda   (default 1)
%  -U  regularization parameter lambda_u (default 1)
%  -S  maximum number of switches in TSVM (default 0.5*number of unlabeled examples)
%  -R  positive class fraction of unlabeled data  (0.5)
%  -f  weights_filename (Test Mode: input_filename is the test file)
%  -Cp relative cost for positive examples (only available with -A 1)
%  -Cn relative cost for negative examples (only available with -A 1)
%
% III. the set of variables to optimise the performance
% r_dev:
% r_nor:
% algor:
% uU: 1.3
% portion:
% rR:
% per_neu:
% tRatio:
% * times:
% cases in total

clc
clear
close all

%% load the prepared data matrices
%
disp('Loading data & information struct...');
tic
load('zeta1333_6mm_bin.mat');
load('stroke_info_light.mat');
load('age_all.mat');
data=zeta1333_6mm_bin;
info=stroke_info_light;
age=age_all./max(age_all);
toc

%% variables
% key mode feature
feature='LvR';

% threshold of hit rate
hit=3;

% threshold of labelled / unlabelled dataset
% * the threshold is estimated by the probability distribution function: circ_ksdensity
% * refer to the figures of 'circ_ksd_bin50.fig' & 'circ_ksd_bin100.fig'
% threshold of the severe deviated
devpair=[-10 10];

% vector for labels
label=[];

% svmlin realted paths
svmlin_p='/media/2TB_1/pro_Predict_stroke/workspace/functions_matlab/ml/svmlin2015';
data_folder='mod_data';
res_folder='mod_results';
mask_folder='mod_masks';

svmlin_datap=[svmlin_p '/' data_folder];
svmlin_resp=[svmlin_p '/' res_folder];
svmlin_mask=[svmlin_p '/' mask_folder];

for h=1:length(hit)
    
    tmphit=hit(h);
    
    %% call function: ion20150430110732_reduce_dimensionality(data,hits)
    [data_red, age_red, info_red]=ion20150430110732_reduce_dimensionality(data,age,info,tmphit);
    
    %% based on the set of thresholds, label the data and get the refined data
    % the range of severe deviation
    for l=1:size(devpair,1)
        
        tmpl=devpair(l,1);
        tmpr=devpair(l,2);
        
        % summarise key information of currently running mode
        str_thresh=['CT_' feature '_[inf_' num2str(tmpl) ']_[' num2str(tmpr) '_inf]'];
        summary=[datestr(now,'yyyymmddHHMMSS') '_' str_thresh];
        
        % mod: severe deviation on the left vs mild deviation
        for i=1:length(info_red)
            
            ctgazeF=info_red(i).ctgazeF;
            
            if ctgazeF==1
                
                tgaze=mean(info_red(i).ctresc_clmp_calc);
                
                if le(tgaze,tmpl)
                    label(i,:)=1;
                elseif ge(tgaze,tmpr)
                    label(i,:)=-1;
                else
                    label(i,:)=9;
                end
            else
                label(i,:)=0;
            end
            
            clear ctgazeF tgaze
        end
        
        % labelled data (gaze dev on the left) and the unlabelled data (unknown gaze dev)
        labind=find(label~=9);
        sublabel=label(labind);
        subdata=data_red(labind,:);
        subage=age_red(labind);
        
        % index of the hit voxels (the voxels involved into the lesions at least once)
        data_sum=sum(subdata,1);
        hit_ind=find(data_sum>0);
        
        % indeices of positive, negative labels and as well as the unlabelled data
        posind=find(sublabel==1);
        negind=find(sublabel==-1);
        neuind=find(sublabel==0);
        
        % the number of positive, negative labelled data, and unlabelled data, respectively;
        posnum=length(posind);
        negnum=length(negind);
        neunum=length(neuind);
        
        % standard number in order to select equal number of positive and negative data for training
        if ge(posnum,negnum)
            stdnum=negnum;
        else
            stdnum=posnum;
        end
        
        % distrbution of labels as use of an indicator
        distri_lab=[num2str(posnum) '_' num2str(negnum) '_' num2str(neunum)];
        
        % analyzing mode name
        mod=[summary '_' distri_lab];
        
        % create folder to hold date files and results
        system(['mkdir ' svmlin_datap '/' mod]);
        system(['mkdir ' svmlin_resp '/' mod]);
        system(['mkdir ' svmlin_mask '/' mod]);
        
        %% pre-set the range of parameters in SVMLin
        % -A algorithm : set algorithm (default 1)
        %    0 -- Regularized Least Squares (RLS) Classification
        %    1 -- SVM (L2-SVM-MFN) (default choice)
        %    2 -- Multi-switch Transductive SVM (using L2-SVM-MFN)
        %    3 -- Deterministic Annealing Semi-supervised SVM (using L2-SVM-MFN)
        algor=2;
        
        % -W  regularization parameter lambda   (default 1)
        wW=[10 10^2 10^3 10^4 10^5 10^6 10^7 10^8 10^9 10^10];
        
        % -U  regularization parameter lambda_u (default 1)
        uU=[1.3 1.4 10 10^2 10^3 10^4 10^5 10^6 10^7 10^8 10^9 10^10];
        
        % -S  maximum number of switches in TSVM (default 0.5*number of unlabeled examples)
        portion=1;
        
        % -R  positive class fraction of unlabeled data  (0.5)
        % set -R parameter to be the same as the ratio of labelled set.
        rR_std=length(posind)./(length(posind)+length(negind));
        
        %         if le(rR_std,0.15)
        %             rR1=rR_std-0.01:-0.01:rR_std-0.02;
        %             rR2=rR_std:0.03:rR_std+0.15;
        %         elseif gt(rR_std,0.15) && lt(rR_std,0.70)
        %             rR1=rR_std-0.02:-0.02:rR_std-0.04;
        %             rR2=rR_std:0.05:rR_std+0.25;
        %         elseif ge(rR_std,0.70) && lt(rR_std,0.90)
        %             rR1=rR_std-0.05:-0.05:rR_std-0.15;
        %             rR2=rR_std:0.03:rR_std+0.09;
        %         elseif ge(rR_std,0.90)
        %             rR1=rR_std-0.03:-0.03:rR_std-0.06;
        %             rR2=rR_std:0.02:0.95;
        %         end
        
        rR=rR_std;
        % rR=[0.5, rR1, rR2];
        
        % vary the portion of neutral data
        per_neu=0;
        
        %% classification via SVMLin
        % the number of iterations
        times=10;
        
        percent=10;
        
        % the number of labelled data in training
        tRatio=0.80;
        
        % a range of loop to attempt the optimal parameters for good performance
        for ar=1:length(algor)
            
            tmp_algor=algor(ar);
            
            for w=1:length(wW)
                
                tmp_wW=wW(w);
                
                for lu=1:length(uU)
                    
                    tmp_uU=uU(lu);
                    
                    for po=1:length(portion)
                        
                        tmp_portion=portion(po);
                        
                        for fr=1:length(rR)
                            
                            tmp_rR=rR(fr);
                            
                            for ne=1:length(per_neu)
                                
                                tmp_per_neu=per_neu(ne);
                                
                                for ra=1:length(tRatio)
                                    
                                    tmp_tRatio=tRatio(ra);
                                    
                                    for iter=1:times
                                        
                                        sS=floor(tmp_portion.*(neunum+(abs(posnum-negnum))).*tmp_per_neu);
                                        
                                        if gt(posnum,negnum)
                                            
                                            tmp=randperm(posnum);
                                            tmp=tmp(1:stdnum);
                                            
                                            % training and test positive labels
                                            training_posind=posind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_posind=posind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % extra part of positive labels (posnum-negnum)
                                            posind_cov=posind;
                                            posind_cov(tmp)=[];
                                            
                                            % convert extra positive labels to neutral
                                            label_tmp=sublabel;
                                            label_tmp(posind_cov)=0;
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            tmp=randperm(negnum);
                                            tmp=tmp(1:stdnum);
                                            
                                            % training and test negative labels
                                            training_negind=negind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_negind=negind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            % vary the number of unlabelled data
                                            tmp1=randperm(length(posind_cov));
                                            tmp2=randperm(length(neuind));
                                            
                                            training_posind_cov=posind_cov(tmp1(1:floor(length(posind_cov)*tmp_per_neu)));
                                            training_neuind=neuind(tmp2(1:floor(length(neuind)*per_neu)));
                                            
                                            % empty / clear tmp*
                                            clear tmp1 tmp2
                                            
                                            % combined training indeices
                                            training_index=[training_posind;training_negind;training_posind_cov;training_neuind];
                                            
                                        elseif posnum==negnum
                                            
                                            % training and test positive labels
                                            tmp=randperm(posnum);
                                            tmp=tmp(1:stdnum);
                                            training_posind=posind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_posind=posind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            % training and test negative labels
                                            tmp=randperm(negnum);
                                            tmp=tmp(1:stdnum);
                                            training_negind=negind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_negind=negind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            % vary the number of unlabelled data
                                            tmp=randperm(length(neuind));
                                            training_neuind=neuind(tmp(1:floor(length(neuind)*tmp_per_neu)));
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            % combined training indeices
                                            training_index=[training_posind;training_negind;training_neuind];
                                            
                                        elseif lt(posnum,negnum)
                                            
                                            % training and test positive labels
                                            tmp=randperm(posnum);
                                            tmp=tmp(1:stdnum);
                                            training_posind=posind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_posind=posind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            % training and test negative labels
                                            tmp=randperm(negnum);
                                            tmp=tmp(1:stdnum);
                                            training_negind=negind(tmp(1:floor(stdnum*tmp_tRatio)));
                                            test_negind=negind(tmp(floor(stdnum*tmp_tRatio)+1:end));
                                            
                                            % extra part of negative labels (negnum-posnum)
                                            negind_cov=negind;
                                            negind_cov(tmp)=[];
                                            
                                            % empty / clear tmp
                                            clear tmp
                                            
                                            label_tmp=sublabel;
                                            label_tmp(negind_cov)=0;
                                            
                                            % vary the number of unlabels
                                            tmp1=randperm(length(negind_cov));
                                            tmp2=randperm(length(neuind));
                                            
                                            training_negind_cov=negind_cov(tmp1(1:floor(length(negind_cov)*tmp_per_neu)));
                                            training_neuind=neuind(tmp2(1:floor(length(neuind)*tmp_per_neu)));
                                            
                                            % empty / clear tmp*
                                            clear tmp1 tmp2
                                            
                                            % combined training indeices
                                            training_index=[training_posind;training_negind;training_negind_cov;training_neuind];
                                        end
                                        
                                        % training data with the extra column of normalised age as a weighting factor
                                        training_label=label_tmp(training_index,:);
                                        training_data=subdata(training_index,:);
                                        
                                        training_age=subage(training_index,:);
                                        %training_data_corr=[training_data training_age];
                                        training_data_corr=training_data;
                                        
                                        % test data with the extra column of normalised age as a weighting factor
                                        test_index=[test_posind;test_negind];
                                        test_label=label_tmp(test_index,:);
                                        test_data=subdata(test_index,:);
                                        
                                        test_age=subage(test_index,:);
                                        %test_data_corr=[test_data test_age];
                                        test_data_corr=test_data;
                                        
                                        % the nunber of training and test data
                                        len_training=length(training_posind) + length(training_negind);
                                        len_test=length(test_label);
                                        
                                        % data files
                                        training_data_fname=['training_data' ...
                                            '_' mod ...
                                            '_' num2str(len_training) ...
                                            '_' num2str(len_test) ...
                                            '_' num2str(tmp_algor) ...
                                            '_' num2str(tmp_wW) ...
                                            '_' num2str(tmp_uU) ...
                                            '_' num2str(tmp_portion) ...
                                            '_' num2str(tmp_rR) ...
                                            '_' num2str(tmp_per_neu) ...
                                            '_' num2str(iter) ...
                                            '_' num2str(tmp_tRatio)];
                                        file_training_data=[svmlin_datap '/' mod '/' training_data_fname];
                                        
                                        training_label_fname=['training_label' ...
                                            '_' mod ...
                                            '_' num2str(len_training) ...
                                            '_' num2str(len_test) ...
                                            '_' num2str(tmp_algor) ...
                                            '_' num2str(tmp_wW) ...
                                            '_' num2str(tmp_uU) ...
                                            '_' num2str(tmp_portion) ...
                                            '_' num2str(tmp_rR) ...
                                            '_' num2str(tmp_per_neu) ...
                                            '_' num2str(iter) ...
                                            '_' num2str(tmp_tRatio)];
                                        file_training_label=[svmlin_datap '/' mod '/' training_label_fname];
                                        
                                        test_data_fname=['test_data' ...
                                            '_' mod ...
                                            '_' num2str(len_training) ...
                                            '_' num2str(len_test) ...
                                            '_' num2str(tmp_algor) ...
                                            '_' num2str(tmp_wW) ...
                                            '_' num2str(tmp_uU) ...
                                            '_' num2str(tmp_portion) ...
                                            '_' num2str(tmp_rR) ...
                                            '_' num2str(tmp_per_neu) ...
                                            '_' num2str(iter) ...
                                            '_' num2str(tmp_tRatio)];
                                        file_test_data=[svmlin_datap '/' mod '/' test_data_fname];
                                        
                                        test_label_fname=['test_label' ...
                                            '_' mod ...
                                            '_' num2str(len_training) ...
                                            '_' num2str(len_test) ...
                                            '_' num2str(tmp_algor) ...
                                            '_' num2str(tmp_wW) ...
                                            '_' num2str(tmp_uU) ...
                                            '_' num2str(tmp_portion) ...
                                            '_' num2str(tmp_rR) ...
                                            '_' num2str(tmp_per_neu) ...
                                            '_' num2str(iter) ...
                                            '_' num2str(tmp_tRatio)];
                                        file_test_label=[svmlin_datap '/' mod '/' test_label_fname];
                                        
                                        % open the data files for writing
                                        fid1=fopen(file_training_data,'w+');
                                        fid2=fopen(file_training_label,'w+');
                                        fid3=fopen(file_test_data,'w+');
                                        fid4=fopen(file_test_label,'w+');
                                        
                                        datastr=[];
                                        
                                        % writing training data files
                                        for i=1:length(training_label)
                                            
                                            tmpVol=find(training_data_corr(i,:)>0);
                                            for j=1:length(tmpVol)
                                                if j==1
                                                    datastr=[mat2str(tmpVol(j)) ':' num2str(training_data_corr(i,tmpVol(j)))];
                                                else
                                                    datastr=[datastr ' ' mat2str(tmpVol(j)) ':' num2str(training_data_corr(i,tmpVol(j)))];
                                                end
                                            end
                                            fprintf(fid1, '%s\n', datastr);
                                            fprintf(fid2, '%s\n', num2str(training_label(i)));
                                            
                                            tmpVol=[];
                                            datastr=[];
                                        end
                                        
                                        % writing test data files
                                        for i=1:length(test_label)
                                            tmpVol=find(test_data_corr(i,:));
                                            for j=1:length(tmpVol)
                                                if j==1
                                                    datastr=[mat2str(tmpVol(j)) ':' num2str(test_data_corr(i,tmpVol(j)))];
                                                else
                                                    datastr=[datastr ' ' mat2str(tmpVol(j)) ':' num2str(test_data_corr(i,tmpVol(j)))];
                                                end
                                            end
                                            fprintf(fid3, '%s\n', datastr);
                                            fprintf(fid4, '%s\n', num2str(test_label(i)));
                                            
                                            tmpVol=[];
                                            datastr=[];
                                        end
                                        
                                        fclose(fid1);
                                        fclose(fid2);
                                        fclose(fid3);
                                        fclose(fid4);
                                        
                                        % algorithm: 3 -- Deterministic Annealing Semi-supervised SVM (using L2-SVM-MFN)
                                        % [w,o]=system(['./svmlin -A 3 -W 0.001 -U 1.3 -R 0.56 -S 1500 ' sparse(training_data_corr) training_label]);
                                        % [w,o]=svmlin([],test_data_corr,[],w);
                                        % [w,o]=svmlin([],test_data_corr,test_label,w);
                                        % ./svmlin -A 3 -W 0.001 -U 1.3 -R 0.53 -S 1300 example/walk_train_1 example/walk_train_label_1
                                        % ./svmlin -f walk_train_1.weights example/walk_test_1 example/walk_test_label_1
                                        % system(['./svmlin -A ' algo ' -W 0.001 -U ' num2str(uU(u)) ' -R ' num2str(rR(r)) ' -S 1500 data/walk_train_' num2str(iter) ' data/walk_train_label_' num2str(iter)]);
                                        % system(['./svmlin -f walk_train_' num2str(iter) '.weights data/walk_test_' num2str(iter) ' data/walk_test_label_' num2str(iter)]);
                                        
                                        system(['./svmlin -A ' num2str(tmp_algor) ...
                                            ' -W ' num2str(tmp_wW) ...
                                            ' -U ' num2str(tmp_uU) ...
                                            ' -R ' num2str(tmp_rR) ...
                                            ' -S ' num2str(sS) ...
                                            ' ' data_folder '/' mod '/' training_data_fname ...
                                            ' ' data_folder '/' mod '/' training_label_fname]);
                                        
                                        system(['./svmlin -f ' training_data_fname '.weights' ...
                                            ' ' data_folder '/' mod '/' test_data_fname ...
                                            ' ' data_folder '/' mod '/' test_label_fname]);
                                        
                                        % load the outputs and check the performance
                                        valid=sign(dlmread([test_data_fname '.outputs']));
                                        accur=sum(test_label==valid)./length(test_label);
                                        
                                        results(iter,fr,lu,w)=accur;
                                        sensitivity(iter,fr,lu,w)=sum(test_label==valid & test_label==1)./sum(test_label==1);
                                        specificity(iter,fr,lu,w)=sum(test_label==valid & test_label==-1)./sum(test_label==-1);
                                        
                                        % display the results
                                        accur
%                                         disp(results);
%                                         disp(sensitivity);
%                                         disp(specificity);
                                        
                                        clear label_tmp
                                        clear training_*
                                        clear test_*
                                        
                                        clear file* fid* tmpVol datastr
                                        clear valid accur
                                    end
                                    
                                    %% create the weighting mask, including the information as follows:
                                    % - the name of currently running mode
                                    % - the index of hit voxels
                                    % - len_training
                                    % - len_test
                                    % - algor
                                    % - wW
                                    % - uU
                                    % - portion
                                    % - rR
                                    % - per_neu
                                    % - times
                                    % - tRatio
%                                     ion20150427195810_create_weights_map(svmlin_p, ...
%                                         svmlin_mask, ...
%                                         mod, ...
%                                         hit_ind, ...
%                                         num2str(len_training), ...
%                                         num2str(len_test), ...
%                                         num2str(tmp_algor), ...
%                                         num2str(tmp_wW), ...
%                                         num2str(tmp_uU), ...
%                                         num2str(tmp_portion), ...
%                                         num2str(tmp_rR), ...
%                                         num2str(tmp_per_neu), ...
%                                         num2str(times), ...
%                                         num2str(tmp_tRatio))
                                    
                                    %% clear the outputs
                                    % system(['rm -r ' svmlin_p '/' data_folder '/' mod]);
                                    rmfiles1=dir(['*_' mod '_*.weights']);
                                    rmfiles2=dir(['*_' mod '_*.outputs']);
                                    
                                    for rm1=1:length(rmfiles1)
                                        system(['rm ' rmfiles1(rm1).name]);
                                    end
                                    for rm2=1:length(rmfiles2)
                                        system(['rm ' rmfiles2(rm2).name]);
                                    end
                                    
                                    %% proceeding the results and save to specified location
                                    mean2sen=mean(sensitivity(:));
                                    mean2spc=mean(specificity(:));
                                    
                                    t_mean2sen=trimmean(sensitivity(:),percent);
                                    t_mean2spc=trimmean(specificity(:),percent);
                                    
                                    med2sen=median(sensitivity(:));
                                    med2spc=median(specificity(:));
                                    
                                    std2sen_err=std(sensitivity(:))./sqrt(times);
                                    std2spc_err=std(specificity(:))./sqrt(times);
                                    
                                    keys(:,1)=mean2sen(:) + mean2spc(:);
                                    
                                    keys(:,2)=mean2sen(:);
                                    keys(:,3)=mean2spc(:);
                                    
                                    keys(:,4)=t_mean2sen(:);
                                    keys(:,5)=t_mean2spc(:);
                                    
                                    keys(:,6)=med2sen(:);
                                    keys(:,7)=med2spc(:);
                                    
                                    keys(:,8)=std2sen_err(:);
                                    keys(:,9)=std2spc_err(:);
                                    
                                    save([svmlin_resp '/' mod '/keys' ...
                                        '_' num2str(len_training) ...
                                        '_' num2str(len_test) ...
                                        '_' num2str(tmp_algor) ...
                                        '_' num2str(tmp_wW) ...
                                        '_' num2str(tmp_uU) ...
                                        '_' num2str(tmp_portion) ...
                                        '_' num2str(tmp_rR) ...
                                        '_' num2str(tmp_per_neu) ...
                                        '_' num2str(times) ...
                                        '_' num2str(tmp_tRatio) '.mat'], 'keys');
                                    
%                                     save([svmlin_resp '/' mod '/results' ...
%                                         '_' num2str(len_training) ...
%                                         '_' num2str(len_test) ...
%                                         '_' num2str(tmp_algor) ...
%                                         '_' num2str(tmp_wW) ...
%                                         '_' num2str(tmp_uU) ...
%                                         '_' num2str(tmp_portion) ...
%                                         '_' num2str(tmp_rR) ...
%                                         '_' num2str(tmp_per_neu) ...
%                                         '_' num2str(times) ...
%                                         '_' num2str(tmp_tRatio) '.mat'], 'results', 'sensitivity', 'specificity');
                                    
                                    clear keys mean2sen mean2spc t_mean2sen t_mean2spc med2sen med2spc std2sen_err std2spc_err;
                                    % clear results sensitivity specificity;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

save([svmlin_resp '/' mod '/whole_results' ...
                                        '_' num2str(len_training) ...
                                        '_' num2str(len_test) ...
                                        '_' num2str(tmp_algor) ...
                                        '_' num2str(tmp_wW) ...
                                        '_' num2str(tmp_uU) ...
                                        '_' num2str(tmp_portion) ...
                                        '_' num2str(tmp_rR) ...
                                        '_' num2str(tmp_per_neu) ...
                                        '_' num2str(times) ...
                                        '_' num2str(tmp_tRatio) '.mat'], 'results', 'sensitivity', 'specificity');

                                    
%% end of this function
end

