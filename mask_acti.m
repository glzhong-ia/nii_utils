clear;clc
path = 'D:\data\finger\202011_K19_ljx\acti_results_S2';
%% BL SS
BL_SS = load_nii( fullfile(path, 'mask.nii') );
FU2_SS = load_nii( fullfile(path, 'spmT_0001.nii') );

%% BL_FU2_SS_union_mask
BL_FU2_SS_mask = nan( size(BL_SS.img) );
ind_BL_SS_roi = find( BL_SS.img(:) > 0 );  % mask roi 1, no roi 0
ind_FU_SS_pos = find( FU2_SS.img(:) > 0 );
ind_FU_SS_neg = find( FU2_SS.img(:) < 0 );
ind_BL_FU_SS = unique( [ind_BL_SS_roi; ind_FU_SS_pos; ind_FU_SS_neg] );

if ~isempty(ind_BL_FU_SS)
    BL_FU2_SS_mask(ind_BL_FU_SS) = 1;
    nii = make_nii(BL_FU2_SS_mask, BL_SS.hdr.dime.pixdim(2:4), BL_SS.hdr.hist.originator(1:3), [], 'BL_FU2_SS_mask');
    save_nii(nii, fullfile( path, 'BL_FU2_SF_mask.nii') );
end
