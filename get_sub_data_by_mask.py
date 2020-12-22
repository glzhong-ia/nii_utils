# -*- coding: utf-8 -*-\

import nibabel as nib
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
# import skimage.io as io

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 显示一个图
def show_img(ori_img):
    plt.imshow(ori_img[90,:,:], cmap = 'gray')
    plt.show()

# 可以设置显示范围下限
def drawPoints(points_, lower_limmit=None):
    fig = plt.figure()

    ax1 = plt.axes(projection='3d')
    points = np.array(points_)
    if lower_limmit is None:
        lower_limmit = points[:,3].min()
    points_limit = points[
        np.where(points[:,3] >= lower_limmit)[0]
        ,:]
    zd = points_limit[:, 2]
    xd = points_limit[:, 0]
    yd = points_limit[:, 1]
    ax3d = ax1.scatter(xd, yd, zd, c=points_limit[:,3], cmap='coolwarm')  # 绘制散点图
    plt.colorbar(ax3d, ax=ax1)
    plt.show()

def getLabelofPoints(input_data, mask_data):
    mask_x_unique = np.sort(np.unique(mask_data[:, 0]))
    mask_y_unique = np.unique(mask_data[:, 1])
    mask_z_unique = np.unique(mask_data[:, 2])
    delta = mask_x_unique[1] - mask_x_unique[0]
    minx = np.min(mask_data[:,:3], axis=0)
    maxx = np.max(mask_data[:,:3], axis=0)
    mask_data_bak = mask_data
    mask_data[:, 0] = np.round((mask_data[:, 0] - minx[0])/delta).astype(np.int32)
    mask_data[:, 1] = np.round((mask_data[:, 1] - minx[1])/delta).astype(np.int32)
    mask_data[:, 2] = np.round((mask_data[:, 2] - minx[2])/delta).astype(np.int32)
    mask_x_unique = np.unique(mask_data[:, 0])
    mask_y_unique = np.unique(mask_data[:, 1])
    mask_z_unique = np.unique(mask_data[:, 2])

    # 补全mask_data为立方体
    all_mask_data = np.zeros((len(mask_x_unique), len(mask_y_unique), len(mask_z_unique)), dtype=np.int32)
    mask_data_index = (mask_data[:, 0] * len(mask_y_unique) * len(mask_z_unique) + \
                        mask_data[:, 1] * len(mask_z_unique) + \
                        mask_data[:, 2]).astype(np.int32)
    all_mask_data_flatten = all_mask_data.flatten()
    all_mask_data_flatten[mask_data_index] = mask_data[:, 3]
    all_mask_data = all_mask_data_flatten.reshape(all_mask_data.shape)
    
    # 补全input为立方体
    input_x_int = np.round((input_data[:, 0] - minx[0])/delta).astype(np.int32)
    input_y_int = np.round((input_data[:, 1] - minx[1])/delta).astype(np.int32)
    input_z_int = np.round((input_data[:, 2] - minx[2])/delta).astype(np.int32)
    sub_x_index = np.unique(input_x_int)
    sub_y_index = np.unique(input_y_int)
    sub_z_index = np.unique(input_z_int)
    sub_min = input_data.min(axis=0)
    # input 在立方体all_sub_data中的index
    sub_index = (np.round((input_data[:, 0] - sub_min[0])/delta).astype(np.int32) * len(sub_y_index) * len(sub_z_index) + \
                 np.round((input_data[:, 1] - sub_min[1])/delta).astype(np.int32) * len(sub_z_index) + \
                 np.round((input_data[:, 2] - sub_min[2])/delta).astype(np.int32)).astype(np.int32)
    all_sub_data = np.zeros((len(sub_x_index), len(sub_y_index), len(sub_z_index)), dtype=np.int32)
    
    # 计算子区间
    sub_sub_x_start = int(0 if sub_x_index.min() > mask_x_unique.min() else mask_x_unique.min() - sub_x_index.min())
    sub_sub_y_start = int(0 if sub_y_index.min() > mask_y_unique.min() else mask_y_unique.min() - sub_y_index.min())
    sub_sub_z_start = int(0 if sub_z_index.min() > mask_z_unique.min() else mask_z_unique.min() - sub_z_index.min())
    sub_sub_x_end = int(len(sub_x_index) if sub_x_index.max() < mask_x_unique.max() else len(sub_x_index) + mask_x_unique.max() - sub_x_index.max())
    sub_sub_y_end = int(len(sub_y_index) if sub_y_index.max() < mask_y_unique.max() else len(sub_y_index) + mask_y_unique.max() - sub_y_index.max())
    sub_sub_z_end = int(len(sub_z_index) if sub_z_index.max() < mask_z_unique.max() else len(sub_z_index) + mask_z_unique.max() - sub_z_index.max())

    sub_mask_x_start = int(0 if sub_x_index.min() < mask_x_unique.min() else sub_x_index.min() - mask_x_unique.min())
    sub_mask_y_start = int(0 if sub_y_index.min() < mask_y_unique.min() else sub_y_index.min() - mask_y_unique.min())
    sub_mask_z_start = int(0 if sub_z_index.min() < mask_z_unique.min() else sub_z_index.min() - mask_z_unique.min())
    sub_mask_x_end = int(len(mask_x_unique) if sub_x_index.max() > mask_x_unique.max() else len(mask_x_unique) - mask_x_unique.max() + sub_x_index.max())
    sub_mask_y_end = int(len(mask_y_unique) if sub_y_index.max() > mask_y_unique.max() else len(mask_y_unique) - mask_y_unique.max() + sub_y_index.max())
    sub_mask_z_end = int(len(mask_z_unique) if sub_z_index.max() > mask_z_unique.max() else len(mask_z_unique) - mask_z_unique.max() + sub_z_index.max())

    all_sub_data[sub_sub_x_start:sub_sub_x_end, sub_sub_y_start:sub_sub_y_end, sub_sub_z_start:sub_sub_z_end] = \
    all_mask_data[sub_mask_x_start:sub_mask_x_end, sub_mask_y_start:sub_mask_y_end, sub_mask_z_start:sub_mask_z_end]

    # label_list = all_sub_data.flatten().astype(np.int32)# np.zeros((len(input), 1), dtype=np.int32)+
    label_list = all_sub_data.flatten().astype(np.int32)[sub_index]
    
    return label_list

def get_sub_data_by_mask(input_nii, mask_nii):
    input_data = input_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    input_data_flatten = np.zeros((input_data.shape[0] * input_data.shape[1] * input_data.shape[2], 4))
    mask_data_flatten = np.zeros((mask_data.shape[0] * mask_data.shape[1] * mask_data.shape[2], 4))
    input_data_flatten[:,3] = input_data.flatten()
    mask_data_flatten[:,3] = mask_data.flatten()
    input_data_flatten[:,:3] = np.array(np.unravel_index(range(len(input_data_flatten)), input_data.shape)).T
    mask_data_flatten[:,:3] = np.array(np.unravel_index(range(len(mask_data_flatten)), mask_data.shape)).T
    # 仿射计算
    input_data_flatten_affine = input_data_flatten
    input_data_flatten_affine[:,0] = input_nii.affine[0,0] * input_data_flatten[:,0] + \
                                     input_nii.affine[0,1] * input_data_flatten[:,1] + \
                                     input_nii.affine[0,2] * input_data_flatten[:,2] + \
                                     input_nii.affine[0,3]
    input_data_flatten_affine[:,1] = input_nii.affine[1,0] * input_data_flatten[:,0] + \
                                     input_nii.affine[1,1] * input_data_flatten[:,1] + \
                                     input_nii.affine[1,2] * input_data_flatten[:,2] + \
                                     input_nii.affine[1,3]
    input_data_flatten_affine[:,2] = input_nii.affine[2,0] * input_data_flatten[:,0] + \
                                     input_nii.affine[2,1] * input_data_flatten[:,1] + \
                                     input_nii.affine[2,2] * input_data_flatten[:,2] + \
                                     input_nii.affine[2,3]
    
    mask_data_flatten_affine = mask_data_flatten
    mask_data_flatten_affine[:,0] = mask_nii.affine[0,0] * mask_data_flatten[:,0] + \
                                     mask_nii.affine[0,1] * mask_data_flatten[:,1] + \
                                     mask_nii.affine[0,2] * mask_data_flatten[:,2] + \
                                     mask_nii.affine[0,3]
    mask_data_flatten_affine[:,1] = mask_nii.affine[1,0] * mask_data_flatten[:,0] + \
                                     mask_nii.affine[1,1] * mask_data_flatten[:,1] + \
                                     mask_nii.affine[1,2] * mask_data_flatten[:,2] + \
                                     mask_nii.affine[1,3]
    mask_data_flatten_affine[:,2] = mask_nii.affine[2,0] * mask_data_flatten[:,0] + \
                                     mask_nii.affine[2,1] * mask_data_flatten[:,1] + \
                                     mask_nii.affine[2,2] * mask_data_flatten[:,2] + \
                                     mask_nii.affine[2,3]
    
    labels = getLabelofPoints(input_data_flatten_affine, mask_data_flatten_affine)
    output_data = input_data_flatten_affine[
        np.where(labels > 0)[0]
        ,:]
    return output_data


# 显示一系列图 
# def show_img(ori_data):
#     for i in range(ori_data.shape[0]):
#         io.imshow(ori_data[i,:,:], cmap = 'gray')
#         print(i)
#         io.show()

objpath = 'D:/DATA/BNARobot/fMRI/202011_K19_ljx/acti_results_S2_HeadMotion/spmT_0001.nii'
maskpath = 'D:/DATA/BNARobot/fMRI/brant_extract_BN_Atlas_274_with_cerebellum_without_255.nii'

obj_data = nib.load(objpath)
obj_img = obj_data.get_fdata()
# show_img(obj_img)

mask_data = nib.load(maskpath)
mask_img = mask_data.get_fdata()
# show_img(mask_img)

sub_obj_data_list = get_sub_data_by_mask(obj_data, mask_data)
drawPoints(sub_obj_data_list, lower_limmit=5.0)
# show_img(common_region)



