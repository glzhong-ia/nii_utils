# -*- coding: utf-8 -*-\

import nibabel as nib
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
# import skimage.io as io

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
        com56tmon_nii_img_2 = np.array([[[]]])
    common_nii = {}
    common_nii['img_1'] = common_nii_img_1
    common_nii['img_2'] = common_nii_img_2
    common_nii['t'] = new_t
    return common_nii

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 显示一个图
def show_img(ori_img):
    plt.imshow(ori_img[90,:,:], cmap = 'gray')
    plt.show()

# 显示一系列图 
# def show_img(ori_data):
#     for i in range(ori_data.shape[0]):
#         io.imshow(ori_data[i,:,:], cmap = 'gray')
#         print(i)
#         io.show()

objpath = 'D:/DATA/BNARobot/fMRI/20201014_K19_S001/20201014_K19_S001_nii_used/acti_results_S2_HeadMotion/spmT_0001.nii'
maskpath = 'D:/DATA/BNARobot/fMRI/20201014_K19_S001/20201014_K19_S001_nii_used/acti_results_S2_HeadMotion/brant_extract_BN_Atlas_274_with_cerebellum_without_255.nii'

obj_data = nib.load(objpath)
obj_img = obj_data.get_fdata()
# show_img(obj_img)

mask_data = nib.load(maskpath)
mask_img = mask_data.get_fdata()
# show_img(mask_img)

common_region = get_common_region(mask_data, obj_data)
# show_img(common_region)
path_save = 'D:/DATA/BNARobot/fMRI/20201014_K19_S001/20201014_K19_S001_nii_used/acti_results_S2_HeadMotion/fgpgcmn_spmT_0001.nii'
# nib.save(common_region,path_save)

# common_mask 是mask中重叠部分
# common_obj 是obj中重叠部分
common_mask = common_region['img_1']
show_img(common_mask)
common_obj = common_region['img_2']
show_img(common_obj)

obj_dot_mask = common_obj * common_mask
show_img(obj_dot_mask)

# max_pos 是最大值所在obj_dot_mask矩阵中的位置
# max_pos_axis 是最大值所在位置的MNI坐标系下的坐标值
max_pos = np.unravel_index(np.argmax(obj_dot_mask), obj_dot_mask.shape)
max_pos_axis = np.array(max_pos) + np.array(common_region['t'])
