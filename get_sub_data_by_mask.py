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
    
    # 补全input_data为立方体
    input_x_index = np.unique(np.round((input_data[:, 0] - minx[0])/delta).astype(np.int32))
    input_y_index = np.unique(np.round((input_data[:, 1] - minx[1])/delta).astype(np.int32))
    input_z_index = np.unique(np.round((input_data[:, 2] - minx[2])/delta).astype(np.int32))
    all_input_data = np.zeros((len(input_x_index), len(input_y_index), len(input_z_index)), dtype=np.int32)
    
    # 计算子区间
    input_input_x_start = int(0 if input_x_index.min() > mask_x_unique.min() else mask_x_unique.min() - input_x_index.min())
    input_input_y_start = int(0 if input_y_index.min() > mask_y_unique.min() else mask_y_unique.min() - input_y_index.min())
    input_input_z_start = int(0 if input_z_index.min() > mask_z_unique.min() else mask_z_unique.min() - input_z_index.min())
    input_input_x_end = int(len(input_x_index) if input_x_index.max() < mask_x_unique.max() else len(input_x_index) + mask_x_unique.max() - input_x_index.max())
    input_input_y_end = int(len(input_y_index) if input_y_index.max() < mask_y_unique.max() else len(input_y_index) + mask_y_unique.max() - input_y_index.max())
    input_input_z_end = int(len(input_z_index) if input_z_index.max() < mask_z_unique.max() else len(input_z_index) + mask_z_unique.max() - input_z_index.max())

    input_mask_x_start = int(0 if input_x_index.min() < mask_x_unique.min() else input_x_index.min() - mask_x_unique.min())
    input_mask_y_start = int(0 if input_y_index.min() < mask_y_unique.min() else input_y_index.min() - mask_y_unique.min())
    input_mask_z_start = int(0 if input_z_index.min() < mask_z_unique.min() else input_z_index.min() - mask_z_unique.min())
    input_mask_x_end = int(len(mask_x_unique) if input_x_index.max() > mask_x_unique.max() else len(mask_x_unique) - mask_x_unique.max() + input_x_index.max())
    input_mask_y_end = int(len(mask_y_unique) if input_y_index.max() > mask_y_unique.max() else len(mask_y_unique) - mask_y_unique.max() + input_y_index.max())
    input_mask_z_end = int(len(mask_z_unique) if input_z_index.max() > mask_z_unique.max() else len(mask_z_unique) - mask_z_unique.max() + input_z_index.max())

    all_input_data[input_input_x_start:input_input_x_end, input_input_y_start:input_input_y_end, input_input_z_start:input_input_z_end] = \
    all_mask_data[input_mask_x_start:input_mask_x_end, input_mask_y_start:input_mask_y_end, input_mask_z_start:input_mask_z_end]

    label_list = all_input_data.flatten().astype(np.int32)# np.zeros((len(input_data), 1), dtype=np.int32)+
    
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
        np.where(labels == 1)[0]
        ,:]
    return output_data


# 显示一系列图 
# def show_img(ori_data):
#     for i in range(ori_data.shape[0]):
#         io.imshow(ori_data[i,:,:], cmap = 'gray')
#         print(i)
#         io.show()

objpath = 'D:/data/finger/202009_K19_jsd/acti_results_S2_HeadMotion/spmT_0001.nii'
maskpath = 'D:/data/finger/202009_K19_jsd/acti_results_S2_HeadMotion/mask.nii'

obj_data = nib.load(objpath)
obj_img = obj_data.get_fdata()
# show_img(obj_img)

mask_data = nib.load(maskpath)
mask_img = mask_data.get_fdata()
# show_img(mask_img)


sub_obj_data_list = get_sub_data_by_mask(obj_data, mask_data)
drawPoints(sub_obj_data_list, lower_limmit=5.0)
# show_img(common_region)



