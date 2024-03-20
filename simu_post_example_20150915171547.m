% Description. simulating a simulation of the treatment
%
% Author. Tianbo XU
%
% Organisation. Institute of Neurology, UCL
%
% init. 15.09.2015
%
% comm.
%

clc
clear
% close all

% the number of cases
cnum = 150;

% labels
lab = [ones(1, cnum/2), ones(1, cnum/2)*-1];

% treatment portions
tp = .1 : .1 : .9;

% prediction accuracy
accu = .8;

% the number of iterations
iter = 300;

% significat p-value
sp = 0.05;

% glmfit distribution
distr = 'poisson';


for i = 1 : length(tp)
    
    tmp_tp = tp(i);
    
    for j = 1 : iter
        
        % create treated label vector
        tmplab = randperm(length(lab));
        tmplab = tmplab(1:ceil(length(lab) / 2));
        
        lab_tr = lab;
        
        if ~eq(tmp_tp, 0)
            tmp_treated = tmplab(1 : ceil(length(tmplab) * tmp_tp));
            lab_tr(tmp_treated) = 1;
        end
        
        % vector of treating marker
        treating = zeros(1, length(lab));
        treating(tmplab) = 1;
        
        % create valid vector
        tmpval = randperm(length(lab));
        tmpval_err = tmpval(1 : ceil(length(lab) * (1 - accu)));
        
        valid = lab;
        
        for e = 1 : length(tmpval_err)
            valid(tmpval_err(e)) = valid(tmpval_err(e)) * -1;
        end
        
        % a glm model that helps to isolate the treatment effect
        [b, dev, stats] = glmfit([lab_tr(:) valid(:)], treating(:), distr);
        pv1(j, i)=stats.p(2);
        pv2(j, i)=stats.p(3);
        clear b dev stats
        
        % compare without the prediction
        [b, dev, stats] = glmfit(lab_tr(:), treating(:), distr);
        p(j, i)=stats.p(2);
        clear b dev stats
        
        clear tmplab tmp_treated lab_tr treating tmpval* val*
    end
    
    clear tmp_tp
end

% check through the existing results
for i = 1 : length(tp)
    pv1_s(i) = sum(pv1(:, i) <= sp) / iter;
    pv2_s(i) = sum(pv2(:, i) <= sp) / iter;
    p_s(i) = sum(p(:, i) <= sp) / iter;
end

% plotting
figure
hold on

plot(pv1_s, 'b');
% plot(pv2_s, 'y');
plot(p_s, 'g');

