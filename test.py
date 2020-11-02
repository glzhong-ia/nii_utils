# -*- coding: utf-8 -*-\

import nibabel as nib
import numpy.matlib
import numpy as np

def get_common_region(nii_img_1, nii_img_2):
    t_1 = nii_img_1.affine[:3, 3]
    t_2 = nii_img_2.affine[:3, 3]
    t_diff  = t_2 - t_1
    shape_diff = np.array(nii_img_2.shape) - np.array(nii_img_1.shape) + t_diff
    new_t = [0,0,0]
    common_x_s_1 = t_diff[0] if t_diff[0] > 0 else 0
    new_t[0] = t_2[0] if t_diff[0] > 0 else t_1[0]
    common_y_s_1 = t_diff[1] if t_diff[1] > 0 else 0
    new_t[1] = t_2[1] if t_diff[1] > 0 else t_1[1]
    common_z_s_1 = t_diff[2] if t_diff[2] > 0 else 0
    new_t[2] = t_2[2] if t_diff[2] > 0 else t_1[2]
    common_x_s_2 = -t_diff[0] if t_diff[0] < 0 else 0
    common_y_s_2 = -t_diff[1] if t_diff[1] < 0 else 0
    common_z_s_2 = -t_diff[2] if t_diff[2] < 0 else 0
    t_1 = t_1 + t_diff
    t_2 = t_2 - t_diff

    common_x_e_1 = nii_img_1.shape[0] if shape_diff[0] > 0 else nii_img_2.shape[0] + t_diff[0]
    common_y_e_1 = nii_img_1.shape[1] if shape_diff[1] > 0 else nii_img_2.shape[1] + t_diff[1]
    common_z_e_1 = nii_img_1.shape[2] if shape_diff[2] > 0 else nii_img_2.shape[2] + t_diff[2]
    common_x_e_2 = nii_img_2.shape[0] if shape_diff[0] < 0 else nii_img_1.shape[0] - t_diff[0]
    common_y_e_2 = nii_img_2.shape[1] if shape_diff[1] < 0 else nii_img_1.shape[1] - t_diff[1]
    common_z_e_2 = nii_img_2.shape[2] if shape_diff[2] < 0 else nii_img_1.shape[2] - t_diff[2]

    try:
        common_nii_img_1 = nii_img_1.get_data()[
            int(common_x_s_1):int(common_x_e_1),
            int(common_y_s_1):int(common_y_e_1),
            int(common_z_s_1):int(common_z_e_1)
        ]
        common_nii_img_2 = nii_img_2.get_data()[
            int(common_x_s_2):int(common_x_e_2),
            int(common_y_s_2):int(common_y_e_2),
            int(common_z_s_2):int(common_z_e_2)
        ]
    except:
        common_nii_img_1 = np.array([[[]]])
        common_nii_img_2 = np.array([[[]]])
    common_nii = {}
    common_nii['img_1'] = common_nii_img_1
    common_nii['img_2'] = common_nii_img_2
    common_nii['t'] = new_t
    return common_nii
    
obj = nib.load('C:/Users/zhong/Desktop/暂存/fingerfmri/acti_results_S2_HeadMotion/spmT_0001.nii')
mask = nib.load('C:/Users/zhong/Desktop/暂存/fingerfmri/acti_results_S2_HeadMotion/ch2.hdr')

common_region = get_common_region(mask, obj)
# common_mask 是mask中重叠部分
# common_obj 是obj中重叠部分
common_mask = common_region['img_1']
common_obj = common_region['img_2']

obj_dot_mask = common_obj * common_mask

# max_pos 是最大值所在obj_dot_mask矩阵中的位置
# max_pos_axis 是最大值所在位置的MNI坐标系下的坐标值
max_pos = np.unravel_index(np.argmax(obj_dot_mask), obj_dot_mask.shape)
max_pos_axis = np.array(max_pos) + np.array(common_region['t'])
