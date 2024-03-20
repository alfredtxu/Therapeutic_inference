% summary: classification mode for SVMLin
%
% Tianbo XU
% init.date: 07.05.2015
% revision.date: 11.05.2015
function ion20150511182306_s8_G1_LvR_CT()

%% comments
% ** about the test cases, refer to file: Stroke_Prediction_Test_Cases.xls
%
% gaze: CT resc_clmp_calc
% data: image data + age + volume
% deviation: [-12 12; -11 11; -10 10; -9 9; -8 8; -7 7; -6 6; -5 5; -4 4; -3 3]
% hit rate: 5; 6; 8
% training ratio:0.8
% tiems: 10
% svmlin algroithm: 1
% W: 10^3; 10^4
% U: 1 : 0.1 : 2
% R: positive labels / labels
% S: 0

clc
clear
close all

%% load the prepared data matrices
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
% percentage of trim
percent=10;

% vector for labels
label=[];

% time stamp
stamp=datestr(now,'yyyymmddHHMMSS');

% key mode feature
mod=['s8_G1_LvR_CT_' stamp];

% svmlin realted paths
svmlin_p='/media/2TB_1/pro_Predict_stroke/workspace/functions_matlab/ml/svmlin2015';
data_folder='ana_data';
res_folder='ana_results';
% mask_folder='ana_masks';

svmlin_datap=[svmlin_p '/' data_folder];
svmlin_resp=[svmlin_p '/' res_folder];
% svmlin_mask=[svmlin_p '/' mask_folder];

system(['mkdir ' svmlin_datap '/' mod]);
system(['mkdir ' svmlin_resp '/' mod]);
% system(['mkdir ' svmlin_mask '/' mod]);

%% set the range of parameters realated to the performance
% threshold of hit rate
hit=[5; 6; 8];

% threshold of labelled / unlabelled dataset
% * the threshold is estimated by the probability distribution function: circ_ksdensity
% * refer to the figures of 'circ_ksd_bin50.fig' & 'circ_ksd_bin100.fig'
% threshold of the severe deviated
devia=[-12 12; -11 11; -10 10; -9 9; -8 8; -7 7; -6 6; -5 5; -4 4; -3 3];

% -A algorithm : set algorithm (default 1)
%    0 -- Regularized Least Squares (RLS) Classification
%    1 -- SVM (L2-SVM-MFN) (default choice)
%    2 -- Multi-switch Transductive SVM (using L2-SVM-MFN)
%    3 -- Deterministic Annealing Semi-supervised SVM (using L2-SVM-MFN)
algor=2;

% -W  regularization parameter lambda   (default 1)
wW=[10^3; 10^4];

% -U  regularization parameter lambda_u (default 1)
uU=1 : 0.1 : 2;

% -S  maximum number of switches in TSVM (default 0.5*number of unlabeled examples)
portion=0;

% the number of iterations
times=10;

% the ratio of training
tRatio=0.80;

%% loop all the cases
for h=1:length(hit)
    
    tmp_hit=hit(h);
    
    % call function: ion20150430110732_reduce_dimensionality(data,hits)
    [data_red, age_red, info_red]=ion20150430110732_reduce_dimensionality(data,age,info,tmp_hit);
    
    % the affected volume of images
    datavol_red=sum(data_red,2);
    datavol_red=datavol_red./max(datavol_red);
    
    % label the data and get the refined data
    for ga=1:size(devia,1)
        
        tmp_dev=devia(ga,:);
        
        for i=1:length(info_red)
            
            ctgazeF=info_red(i).ctgazeF;
            
            if ctgazeF==1
                
                tgaze=mean(info_red(i).ctresc_clmp_calc);
                
                if le(tgaze,tmp_dev(1))
                    label(i,:)=1;
                elseif ge(tgaze,tmp_dev(2))
                    label(i,:)=-1;
                else
                    label(i,:)=9;
                end
            else
                label(i,:)=0;
            end
            
            clear ctgazeF tgaze
        end
        
        labind=find(label~=9);
        sublabel=label(labind);
        subdata=data_red(labind,:);
        subage=age_red(labind);
        subvol=datavol_red(labind);
        
        clear label
        
        % index of the hit voxels (the voxels involved into the lesions at least once)
        %data_sum=sum(subdata,1);
        %hit_ind=find(data_sum>0);
        
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
        
        % -R  positive class fraction of unlabeled data  (0.5)
        % set -R parameter to be the same as the ratio of labelled set; but involve the coverted ones (positive / negitive)
        if ge(posnum,negnum)
            rR=(posnum+(posnum-negnum))./(posnum+negnum+abs(posnum-negnum));
        else
            rR=posnum./(posnum+negnum+abs(posnum-negnum));
        end
        
        for ar=1:length(algor)
            
            tmp_algor=algor(ar);
            
            for w=1:length(wW)
                
                tmp_wW=wW(w);
                
                for lu=1:length(uU)
                    
                    tmp_uU=uU(lu);
                    
                    for po=1:length(portion)
                        
                        tmp_portion=portion(po);
                        
                        for ra=1:length(tRatio)
                            
                            tmp_tRatio=tRatio(ra);
                            
                            for iter=1:times
                                
                                sS=floor(tmp_portion.*(neunum+(abs(posnum-negnum))));
                                
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
                                    
                                    % combined training indeices
                                    training_index=[training_posind;training_negind;posind_cov;neuind];
                                    
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
                                    
                                    % combined training indeices
                                    training_index=[training_posind;training_negind;neuind];
                                    
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
                                    
                                    % combined training indeices
                                    training_index=[training_posind;training_negind;negind_cov;neuind];
                                end
                                
                                % training data with the extra column of normalised age as a weighting factor
                                training_label=label_tmp(training_index,:);
                                training_data=subdata(training_index,:);
                                
                                training_age=subage(training_index,:);
                                training_vol=subvol(training_index,:);
                                
                                %training_data_corr=training_data;
                                %training_data_corr=[training_data training_age];
                                %training_data_corr=[training_data training_vol];
                                training_data_corr=[training_data training_age training_vol];
                                
                                
                                % test data with the extra column of normalised age as a weighting factor
                                test_index=[test_posind;test_negind];
                                test_label=label_tmp(test_index,:);
                                test_data=subdata(test_index,:);
                                
                                test_age=subage(test_index,:);
                                test_vol=subvol(test_index,:);
                                
                                %test_data_corr=test_data;
                                %test_data_corr=[test_data test_age];
                                %test_data_corr=[test_data test_vol];
                                test_data_corr=[test_data test_age test_vol];
                                
                                % the nunber of training and test data
                                len_training=length(training_posind) + length(training_negind);
                                len_test=length(test_label);
                                
                                % data files
                                training_data_fname=['training_data' ...
                                    '_' mod ...
                                    '_' num2str(tmp_hit) ...
                                    '_' num2str(tmp_dev(1)) ...
                                    '_' num2str(tmp_dev(2)) ...
                                    '_' num2str(len_training) ...
                                    '_' num2str(len_test) ...
                                    '_' num2str(tmp_algor) ...
                                    '_' num2str(tmp_wW) ...
                                    '_' num2str(tmp_uU) ...
                                    '_' num2str(tmp_portion) ...
                                    '_' num2str(rR) ...
                                    '_' num2str(iter) ...
                                    '_' num2str(tmp_tRatio)];
                                file_training_data=[svmlin_datap '/' mod '/' training_data_fname];
                                
                                training_label_fname=['training_label' ...
                                    '_' mod ...
                                    '_' num2str(tmp_hit) ...
                                    '_' num2str(tmp_dev(1)) ...
                                    '_' num2str(tmp_dev(2)) ...
                                    '_' num2str(len_training) ...
                                    '_' num2str(len_test) ...
                                    '_' num2str(tmp_algor) ...
                                    '_' num2str(tmp_wW) ...
                                    '_' num2str(tmp_uU) ...
                                    '_' num2str(tmp_portion) ...
                                    '_' num2str(rR) ...
                                    '_' num2str(iter) ...
                                    '_' num2str(tmp_tRatio)];
                                file_training_label=[svmlin_datap '/' mod '/' training_label_fname];
                                
                                test_data_fname=['test_data' ...
                                    '_' mod ...
                                    '_' num2str(tmp_hit) ...
                                    '_' num2str(tmp_dev(1)) ...
                                    '_' num2str(tmp_dev(2)) ...
                                    '_' num2str(len_training) ...
                                    '_' num2str(len_test) ...
                                    '_' num2str(tmp_algor) ...
                                    '_' num2str(tmp_wW) ...
                                    '_' num2str(tmp_uU) ...
                                    '_' num2str(tmp_portion) ...
                                    '_' num2str(rR) ...
                                    '_' num2str(iter) ...
                                    '_' num2str(tmp_tRatio)];
                                file_test_data=[svmlin_datap '/' mod '/' test_data_fname];
                                
                                test_label_fname=['test_label' ...
                                    '_' mod ...
                                    '_' num2str(tmp_hit) ...
                                    '_' num2str(tmp_dev(1)) ...
                                    '_' num2str(tmp_dev(2)) ...
                                    '_' num2str(len_training) ...
                                    '_' num2str(len_test) ...
                                    '_' num2str(tmp_algor) ...
                                    '_' num2str(tmp_wW) ...
                                    '_' num2str(tmp_uU) ...
                                    '_' num2str(tmp_portion) ...
                                    '_' num2str(rR) ...
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
                                    ' -S ' num2str(sS) ...
                                    ' -R ' num2str(rR) ...
                                    ' ' data_folder '/' mod '/' training_data_fname ...
                                    ' ' data_folder '/' mod '/' training_label_fname]);
                                
                                system(['./svmlin -f ' training_data_fname '.weights' ...
                                    ' ' data_folder '/' mod '/' test_data_fname ...
                                    ' ' data_folder '/' mod '/' test_label_fname]);
                                
                                % load the outputs and check the performance
                                valid=sign(dlmread([test_data_fname '.outputs']));
                                accur=sum(test_label==valid)./length(test_label);
                                
                                results(iter,ra,po,lu,w,ar,ga,h)=accur;
                                sensitivity(iter,ra,po,lu,w,ar,ga,h)=sum(test_label==valid & test_label==1)./sum(test_label==1);
                                specificity(iter,ra,po,lu,w,ar,ga,h)=sum(test_label==valid & test_label==-1)./sum(test_label==-1);
                                
                                tmp_sn(iter)=sum(test_label==valid & test_label==1)./sum(test_label==1);
                                tmp_sp(iter)=sum(test_label==valid & test_label==-1)./sum(test_label==-1);
                                
                                % display the results
                                display(accur)
                                
                                clear label_tmp
                                clear training_*
                                clear test_*
                                
                                clear file* fid* tmpVol datastr
                                clear valid accur
                                
                                % clear the outputs
                                rmfiles1=dir(['*_' mod '_*.weights']);
                                rmfiles2=dir(['*_' mod '_*.outputs']);
                                
                                for rm1=1:length(rmfiles1)
                                    system(['rm ' rmfiles1(rm1).name]);
                                end
                                for rm2=1:length(rmfiles2)
                                    system(['rm ' rmfiles2(rm2).name]);
                                end
                            end
                            
                            %% proceeding the results and save to specified location
                            mean2sen=mean(tmp_sn);
                            mean2spc=mean(tmp_sp);
                            
                            t_mean2sen=trimmean(tmp_sn,percent);
                            t_mean2spc=trimmean(tmp_sp,percent);
                            
                            med2sen=median(tmp_sn);
                            med2spc=median(tmp_sp);
                            
                            std2sen_err=std(tmp_sn)./sqrt(times);
                            std2spc_err=std(tmp_sp)./sqrt(times);
                            
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
                                '_' num2str(tmp_hit) ...
                                '_' num2str(tmp_dev(1)) ...
                                '_' num2str(tmp_dev(2)) ...
                                '_' num2str(len_training) ...
                                '_' num2str(len_test) ...
                                '_' num2str(tmp_algor) ...
                                '_' num2str(tmp_wW) ...
                                '_' num2str(tmp_uU) ...
                                '_' num2str(tmp_portion) ...
                                '_' num2str(rR) ...
                                '_' num2str(times) ...
                                '_' num2str(tmp_tRatio) '.mat'], 'keys');
                            
                            clear tmp_sn tmp_sp keys mean2sen mean2spc t_mean2sen t_mean2spc med2sen med2spc std2sen_err std2spc_err;
                        end
                    end
                end
            end
        end
    end
end

%% save the entire results
save([svmlin_resp '/' mod '/whole_results.mat'], 'results', 'sensitivity', 'specificity');


%% end of this function
end

