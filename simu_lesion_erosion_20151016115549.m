% summary: Sumulate drug treatment resulting in rt (ratio) shrinkage of lesion volume in invol.
% 
% Author. Parashkev Nachev
% Organisation. Institute of Neurology, UCL
% 
% init. 2013
% upd. 15.10.2015 by Tianbo XU
% 
function output_dat = simu_lesion_erosion_20151016115549(input_dat, ratio)
% comm.
% input_dat -> inout imaging data, dimensions are [31 37 31] by defaults;
% ratio -> the ratio of lesion involved into the analysis

%% variables
% swithch-on trigger by defaults
trigger = 1;

% input data for further processing
tmp_dat = input_dat;

%% simulation of treatments: lesion shrinking
while trigger == 1
    
    % * this is for interative perimeter of input data
    % volume of input data
    tmp_dat_vol = sum(tmp_dat(:));

    % target volume for analysis
    % target = ceil(tmp_dat_vol * ratio);
    target = floor(tmp_dat_vol * ratio);

    % perimeter of input data (connectivities is 6)
    % * A pixel is part of the perimeter if it is nonzero 
    %   and it is connected to at least one zero-valued pixel
    tmp_dat_perim = bwperim(tmp_dat, 6);

    % volume of inout data perimeter
    tmp_dat_perim_vol = sum(tmp_dat_perim(:));

    % there is a core in the lesion (the voxels withno zero-valued neighbors)
    if gt(tmp_dat_vol, tmp_dat_perim_vol)
        
        % more voxels in the periphery than the volume of target
        if gt(tmp_dat_perim_vol, target) 
            
            % indices of postive voxels in perimeter image
            pos_idx = find(tmp_dat_perim == 1);
            
            % treat the number of target voxels recovered (appearing as no
            % lesions any longer) randomly
            idx = randperm(tmp_dat_perim_vol);
            tmp_dat(pos_idx(idx(1 : target))) = 0;
            
            % swithch-off the trigger
            trigger = 0;
            
        % same number in periphery as target
        elseif eq(tmp_dat_perim_vol, target)
            
            % treat the number of target voxels recovered (appearing as no
            % lesions any longer)
            tmp_dat(tmp_dat_perim == 1) = 0;
            
            % swithch-off the trigger
            trigger = 0;
        
        % a greater number in target than in peripphery; then need to re-iterate
        else 
            tmp_dat(tmp_dat_perim) = 0;
        end
    
    % no core
    else
        
        % more voxels in the original input image than the volume of target
        if gt(tmp_dat_vol, target) 
            
            % indices of postive voxels
            pos_idx = find(tmp_dat);
            
            % treat the number of target voxels recovered (appearing as no
            % lesions any longer) randomly
            idx = randperm(tmp_dat_vol);
            tmp_dat(pos_idx(idx(1 : target))) = 0;
            
            % switch-off the trigger
            trigger = 0;
        
        % the whole lesion will be erosed    
        else
            
            % zero-valued the positive voxels
            tmp_dat(tmp_dat == 1) = 0;
            
            % switch-off the trigger
            trigger = 0;
        end
    end
end
   
%% output image data after simulated lesion treatments
output_dat=tmp_dat;


%% end of this function
end

