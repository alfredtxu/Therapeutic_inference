% Description. simulating a simulation of the treatment with meta-analysis
% (effect size)
%
% Author. Tianbo XU
%
% Organisation. Institute of Neurology, UCL
%
% init. 11.12.2015
%
% comm.
%

clc
clear
close all

% the number of cases
cnum = 100;

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
       
        
        
        clear tmplab tmp_treated lab_tr treating tmpval* val*
    end
    
    clear tmp_tp
end

% plotting
figure
hold on

