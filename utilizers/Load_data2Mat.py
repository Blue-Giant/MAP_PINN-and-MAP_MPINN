# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io
import torch


# load the data from matlab of .mat
def load_Matlab_data(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_meshData2Bayes(dim=2, data_path=None, mesh_number=5, to_torch=False, to_float=True, to_cuda=False,
                       gpu_no=0, use_grad2x=False):
    file2mesh_XY = data_path + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(file2mesh_XY)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


def get_3D_meshData_FixedAxis(dim=2, data_path=None, to_torch=False, to_float=True, to_cuda=False,
                       gpu_no=0, use_grad2x=False):
    assert dim == 3
    file_name2data = str(data_path) + '/' + str('testXYZ') + str('.mat')
    data2matlab = load_Matlab_data(file_name2data)
    data2points = data2matlab['XYZ']
    shape2XY = np.shape(data2points)
    if shape2XY[0] == 3:
        xyz_data = np.transpose(data2points, (1, 0))
    else:
        xyz_data = data2points
    if to_float:
        xyz_data = xyz_data.astype(np.float32)

    if to_torch:
        xyz_data = torch.from_numpy(xyz_data)

        if to_cuda:
            xyz_data = xyz_data.cuda(device='cuda:' + str(gpu_no))

        xyz_data.requires_grad = use_grad2x
    return xyz_data


def get_randomData2mat(dim=2, data_path=None, to_torch=False, to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False):
    if dim == 2:
        file_name2data = str(data_path) + '/' + str('testXY') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XY']
    elif dim == 3:
        file_name2data = str(data_path) + '/' + str('testXYZ') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZ']
    elif dim == 4:
        file_name2data = str(data_path) + '/' + str('testXYZS') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZS']
    elif dim == 5:
        file_name2data = str(data_path) + '/' + str('testXYZST') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZST']
    if to_float:
        data2points = data2points.astype(np.float32)

    if to_torch:
        data2points = torch.from_numpy(data2points)

        if to_cuda:
            data2points = data2points.cuda(device='cuda:' + str(gpu_no))

        data2points.requires_grad = use_grad2x
    return data2points


def get_random_points_1D_noise(data_path=None, noise_level=0.01, to_torch=False, to_float=True, to_cuda=False,
                               gpu_no=0, use_grad2x=False):
    if noise_level == 0.01:
        file_name2x_solu = str(data_path) + '/' + str('0p01') + '/' + str('x_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['x2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p01') + '/' + str('x_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['x2fside']

        file_name2x_coef = str(data_path) + '/' + str('0p01') + '/' + str('x_points2coef') + str('.mat')
        x_mat2coef = load_Matlab_data(file_name2x_coef)
        x_data2coef = x_mat2coef['x2coef']

        file_name2solu_noise = str(data_path) + '/' + str('0p01') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p01') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

        file_name2coef_noise = str(data_path) + '/' + str('0p01') + '/' + str('coef_noise') + str('.mat')
        noise_mat2coef = load_Matlab_data(file_name2coef_noise)
        noise_data2coef = noise_mat2coef['noise2coef']
    elif noise_level == 0.05:
        file_name2x_solu = str(data_path) + '/' + str('0p05') + '/' + str('x_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['x2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p05') + '/' + str('x_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['x2fside']

        file_name2x_coef = str(data_path) + '/' + str('0p05') + '/' + str('x_points2coef') + str('.mat')
        x_mat2coef = load_Matlab_data(file_name2x_coef)
        x_data2coef = x_mat2coef['x2coef']

        file_name2solu_noise = str(data_path) + '/' + str('0p05') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p05') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

        file_name2coef_noise = str(data_path) + '/' + str('0p05') + '/' + str('coef_noise') + str('.mat')
        noise_mat2coef = load_Matlab_data(file_name2coef_noise)
        noise_data2coef = noise_mat2coef['noise2coef']
    elif noise_level == 0.1:
        file_name2x_solu = str(data_path) + '/' + str('0p1') + '/' + str('x_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['x2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p1') + '/' + str('x_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['x2fside']

        file_name2x_coef = str(data_path) + '/' + str('0p1') + '/' + str('x_points2coef') + str('.mat')
        x_mat2coef = load_Matlab_data(file_name2x_coef)
        x_data2coef = x_mat2coef['x2coef']

        file_name2solu_noise = str(data_path) + '/' + str('0p1') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p1') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

        file_name2coef_noise = str(data_path) + '/' + str('0p1') + '/' + str('coef_noise') + str('.mat')
        noise_mat2coef = load_Matlab_data(file_name2coef_noise)
        noise_data2coef = noise_mat2coef['noise2coef']
    elif noise_level == 0.2:
        file_name2x_solu = str(data_path) + '/' + str('0p2') + '/' + str('x_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['x2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p2') + '/' + str('x_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['x2fside']

        file_name2x_coef = str(data_path) + '/' + str('0p2') + '/' + str('x_points2coef') + str('.mat')
        x_mat2coef = load_Matlab_data(file_name2x_coef)
        x_data2coef = x_mat2coef['x2coef']

        file_name2solu_noise = str(data_path) + '/' + str('0p2') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p2') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

        file_name2coef_noise = str(data_path) + '/' + str('0p2') + '/' + str('coef_noise') + str('.mat')
        noise_mat2coef = load_Matlab_data(file_name2coef_noise)
        noise_data2coef = noise_mat2coef['noise2coef']
    elif noise_level == 0.5:
        file_name2x_solu = str(data_path) + '/' + str('0p5') + '/' + str('x_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['x2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p5') + '/' + str('x_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['x2fside']

        file_name2x_coef = str(data_path) + '/' + str('0p5') + '/' + str('x_points2coef') + str('.mat')
        x_mat2coef = load_Matlab_data(file_name2x_coef)
        x_data2coef = x_mat2coef['x2coef']

        file_name2solu_noise = str(data_path) + '/' + str('0p5') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p5') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

        file_name2coef_noise = str(data_path) + '/' + str('0p5') + '/' + str('coef_noise') + str('.mat')
        noise_mat2coef = load_Matlab_data(file_name2coef_noise)
        noise_data2coef = noise_mat2coef['noise2coef']

    if to_float:
        x_data2solu = x_data2solu.astype(np.float32)
        x_data2fside = x_data2fside.astype(np.float32)
        x_data2coef = x_data2coef.astype(np.float32)

        noise_data2solu = noise_data2solu.astype(np.float32)
        noise_data2fside = noise_data2fside.astype(np.float32)
        noise_data2coef = noise_data2coef.astype(np.float32)

    if to_torch:
        x_data2solu = torch.from_numpy(x_data2solu)
        x_data2fside = torch.from_numpy(x_data2fside)
        x_data2coef = torch.from_numpy(x_data2coef)

        noise_data2solu = torch.from_numpy(noise_data2solu)
        noise_data2fside = torch.from_numpy(noise_data2fside)
        noise_data2coef = torch.from_numpy(noise_data2coef)

        if to_cuda:
            x_data2solu = x_data2solu.cuda(device='cuda:' + str(gpu_no))
            x_data2fside = x_data2fside.cuda(device='cuda:' + str(gpu_no))
            x_data2coef = x_data2coef.cuda(device='cuda:' + str(gpu_no))

            noise_data2solu = noise_data2solu.cuda(device='cuda:' + str(gpu_no))
            noise_data2fside = noise_data2fside.cuda(device='cuda:' + str(gpu_no))
            noise_data2coef = noise_data2coef.cuda(device='cuda:' + str(gpu_no))

        x_data2solu.requires_grad = use_grad2x
        x_data2fside.requires_grad = use_grad2x
        x_data2coef.requires_grad = use_grad2x

        noise_data2solu.requires_grad = use_grad2x
        noise_data2fside.requires_grad = use_grad2x
        noise_data2coef.requires_grad = use_grad2x
    return x_data2solu, x_data2fside, x_data2coef, noise_data2solu, noise_data2fside, noise_data2coef


def get_random_points_2D_noise(data_path=None, noise_level=0.01, to_torch=False, to_float=True, to_cuda=False,
                               gpu_no=0, use_grad2x=False):
    if noise_level == 0.01:
        file_name2x_solu = str(data_path) + '/' + str('0p01') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p01') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p01') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p01') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']
    elif noise_level == 0.02:
        file_name2x_solu = str(data_path) + '/' + str('0p02') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p02') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p02') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p02') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']
    elif noise_level == 0.05:
        file_name2x_solu = str(data_path) + '/' + str('0p05') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p05') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p05') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p05') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']
    elif noise_level == 0.1:
        file_name2x_solu = str(data_path) + '/' + str('0p1') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p1') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p1') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p1') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']
    elif noise_level == 0.2:
        file_name2x_solu = str(data_path) + '/' + str('0p2') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p2') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p2') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p2') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']
    elif noise_level == 0.5:
        file_name2x_solu = str(data_path) + '/' + str('0p5') + '/' + str('xy_points2solu') + str('.mat')
        x_mat2solu = load_Matlab_data(file_name2x_solu)
        x_data2solu = x_mat2solu['xy2solu']

        file_name2x_fside = str(data_path) + '/' + str('0p5') + '/' + str('xy_points2fside') + str('.mat')
        x_mat2fside = load_Matlab_data(file_name2x_fside)
        x_data2fside = x_mat2fside['xy2fside']

        file_name2solu_noise = str(data_path) + '/' + str('0p5') + '/' + str('solu_noise') + str('.mat')
        noise_mat2solu = load_Matlab_data(file_name2solu_noise)
        noise_data2solu = noise_mat2solu['noise2solu']

        file_name2fside_noise = str(data_path) + '/' + str('0p5') + '/' + str('fside_noise') + str('.mat')
        noise_mat2fside = load_Matlab_data(file_name2fside_noise)
        noise_data2fside = noise_mat2fside['noise2fside']

    if to_float:
        x_data2solu = x_data2solu.astype(np.float32)
        x_data2fside = x_data2fside.astype(np.float32)

        noise_data2solu = noise_data2solu.astype(np.float32)
        noise_data2fside = noise_data2fside.astype(np.float32)

    if to_torch:
        x_data2solu = torch.from_numpy(x_data2solu)
        x_data2fside = torch.from_numpy(x_data2fside)

        noise_data2solu = torch.from_numpy(noise_data2solu)
        noise_data2fside = torch.from_numpy(noise_data2fside)

        if to_cuda:
            x_data2solu = x_data2solu.cuda(device='cuda:' + str(gpu_no))
            x_data2fside = x_data2fside.cuda(device='cuda:' + str(gpu_no))

            noise_data2solu = noise_data2solu.cuda(device='cuda:' + str(gpu_no))
            noise_data2fside = noise_data2fside.cuda(device='cuda:' + str(gpu_no))

        x_data2solu.requires_grad = use_grad2x
        x_data2fside.requires_grad = use_grad2x

        noise_data2solu.requires_grad = use_grad2x
        noise_data2fside.requires_grad = use_grad2x
    return x_data2solu, x_data2fside, noise_data2solu, noise_data2fside


def get_meshData2Laplace(equation_name=None, mesh_number=2, to_torch=False, to_float=True, to_cuda=False, gpu_no=0,
                         use_grad2x=False):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = '../dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = '../dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = '../dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = '../dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = '../dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        assert (mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(test_meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert(len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


def get_meshData2Boltzmann(equation_name=None, domain_lr='01', mesh_number=2, to_torch=False, to_float=True,
                           to_cuda=False, gpu_no=0, use_grad2x=False):
    if domain_lr == '01':
        meshXY_file = '../dataMat2Boltz/meshData_01/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif domain_lr == '11':
        meshXY_file = '../dataMat2Boltz/meshData_11/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    else:
        xy_data = XY_points

    if to_float:
        xy_data = xy_data.astype(np.float32)

    if to_torch:
        xy_data = torch.from_numpy(xy_data)

        if to_cuda:
            xy_data = xy_data.cuda(device='cuda:' + str(gpu_no))

        xy_data.requires_grad = use_grad2x
    return xy_data


if __name__ == '__main__':
    mat_data_path = '../dataMat_highDim'
    mat_data = get_randomData2mat(dim=2, data_path=mat_data_path)
    print('end!!!!')