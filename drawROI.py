# -*- coding: utf-8 -*-\

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


mask_path = './commonMask.hdr'
spm_path = './spmT_0001.nii'
mask_nii = nib.load(mask_path)
spm_nii = nib.load(spm_path)

spm_dot_mask = spm_nii.get_data() * mask_nii.get_data()
max_pos = np.unravel_index(np.argmax(spm_dot_mask), spm_dot_mask.shape)
# max_pos_axis = np.array(max_pos) + np.array(common_region['t'])
print('aaa')