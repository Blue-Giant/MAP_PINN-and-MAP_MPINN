import os
import sys
import time
import datetime
import platform
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Network import hamiltorch
from Network import BNN_EstimatePara_Utils
from Network import BNN_General_Approximator
from Problems import Eqs2BayesNN3d
from utilizers import plot2Bayes
from Log_Print import BPINN_Config_Log_Print
from Log_Print import BPINN_Train_Log_Print
from Log_Print import BPINN_Test_Log_Print
from Log_Print import Aux_Log_Print
from utilizers import saveData
from utilizers import save_load_NetModule
from utilizers import Load_data2Mat


def calculate_snr(signal=None, noise=None):
    power2signal = torch.mean(torch.square(signal), dim=0)
    power2noise = torch.mean(torch.square(noise), dim=0)
    snr = 10.0*torch.log10(power2signal/power2noise)
    return snr


def print_log_test_mse_rel(mse2solu=None, rel2solu=None, mse2force=None, rel2force=None, log_out=None):
    BPINN_Test_Log_Print.print_log_solu2testing(mse2solu, rel2solu, log_file=log_out)
    BPINN_Test_Log_Print.print_log_fside2testing(mse2force, rel2force, log_file=log_out)


def model_loss2multiscale(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxyz = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy, Du_Dz = Du2Dxyz[:, 0:1], Du2Dxyz[:, 1:2], Du2Dxyz[:, 2:3]
    DDu_Dxxyz = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                    create_graph=True, retain_graph=True)[0]
    DDu_Dyxyz = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                    create_graph=True, retain_graph=True)[0]
    DDu_Dzxyz = torch.autograd.grad(Du_Dz, x_f, grad_outputs=torch.ones_like(Du_Dz),
                                    create_graph=True, retain_graph=True)[0]
    u_xx = DDu_Dxxyz[:, 0:1]
    u_yy = DDu_Dyxyz[:, 1:2]
    u_zz = DDu_Dzxyz[:, 2:3]

    aesp = torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 0], shape=[-1, 1])) * \
           torch.sin(5.0 * torch.pi * torch.reshape(x_f[:, 1], shape=[-1, 1])) * \
           torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 2], shape=[-1, 1]))

    da_dx = -3.0 * torch.pi * torch.sin(3.0 * torch.pi * torch.reshape(x_f[:, 0], shape=[-1, 1])) * \
            torch.sin(5.0 * torch.pi*torch.reshape(x_f[:, 1], shape=[-1, 1])) * \
            torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 2], shape=[-1, 1]))

    da_dy = 5.0 * torch.pi * torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 0], shape=[-1, 1])) * \
            torch.cos(5.0 * torch.pi * torch.reshape(x_f[:, 1], shape=[-1, 1])) * \
            torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 2], shape=[-1, 1]))

    da_dz = -3.0 * torch.pi * torch.cos(3.0 * torch.pi * torch.reshape(x_f[:, 0], shape=[-1, 1])) * \
            torch.cos(5.0 * torch.pi * torch.reshape(x_f[:, 1], shape=[-1, 1])) * \
            torch.sin(3.0 * torch.pi * torch.reshape(x_f[:, 2], shape=[-1, 1]))
    # -div(a·grad u)=f -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy + (da/dz)·(du/dz) + a·ddu/dzz] = f

    pred_f = -1.0*(torch.multiply(da_dx, Du_Dx) + torch.multiply(aesp, u_xx) +
                   torch.multiply(da_dy, Du_Dy) + torch.multiply(aesp, u_yy) +
                   torch.multiply(da_dz, Du_Dz) + torch.multiply(aesp, u_zz))
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, Du_Dz, u_xx, u_yy, u_zz, pred_u, pred_f, aesp, da_dx, da_dy, da_dz
        torch.cuda.empty_cache()

    return ll, output


def solve_bayes_pinn(Rdic=None):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['act_name2Hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    BPINN_Config_Log_Print.dictionary_out2file(R, log_fileout)

    if Rdic['with_gpu'] is True:
        assert (torch.cuda.is_available() is True)
        print(f"Is CUDA available?: {torch.cuda.is_available()}")
        device = 'cuda:0'
    else:
        device = "cpu"

    # hyperparameters
    hamiltorch.set_random_seed(42)

    prior_std = 1
    like_std = Rdic['noise_level']

    lr = Rdic['learning_rate']  # the learning rate for optimizer in PINN model or Hamilton  model
    step2update_lr = Rdic['step2update_lr']
    gamma2update_lr = Rdic['gamma2update_lr']
    open_update_lr = Rdic['update_lr']

    pde = True
    max_epoch = R['max_epoch']  # the total iteration epoch of training for PINN
    burn = 100
    num_samples = R['sample_num2hamilton']  # the number of samplings for hamilton sampler,Hamilton 抽样次数
    L = Rdic['step_num2per_sample']

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    left_b = 0.0        # the left boundary of interested domain
    right_b = 1.0        # the right boundary of interested domain
    behind_b = 0.0      # the behind boundary of interested domain
    front_b = 1.0       # the front boundary of interested domain
    bottom_b = 0.0      # the bottom boundary of interested domain
    top_b = 1.0          # the top boundary of interested domain

    N_tr_u = 30     # the number of sampled points for dealing with solution
    N_tr_f = 1600   # the number of sampled points for dealing with force-side
    N_val = 4900    # the number of sampled points for obtaining real solution, parameter and force-side

    u, f = Eqs2BayesNN3d.get_infos_3d(equa_name=Rdic['equa_name'])   # get the infos for PDE problem

    data = {}
    if 'mesh_grid' == str.lower(Rdic['opt2sampling']):
        # Generating the boundary points for interested domain
        # Constructing the points for left and right boundaries, fixing the x_left and x_right, the y and z are meshed
        y_coord2lr_bd = np.reshape(
            np.linspace(behind_b, front_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        z_coord2lr_bd = np.reshape(
            np.linspace(bottom_b, top_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        meshY2lr, meshZ2lr = np.meshgrid(y_coord2lr_bd, z_coord2lr_bd)  # Construct the mesh points
        mesh_XYpoints2lr = np.concatenate((np.reshape(meshY2lr, newshape=[-1, 1]),
                                           np.reshape(meshZ2lr, newshape=[-1, 1])), axis=-1)

        x_points2left_bd = np.ones(shape=[N_tr_u*N_tr_u, 1]) * left_b
        x_points2right_bd = np.ones(shape=[N_tr_u*N_tr_u, 1]) * right_b
        xyz_left_b = np.concatenate([x_points2left_bd, mesh_XYpoints2lr], axis=-1)
        xyz_right_b = np.concatenate([x_points2right_bd, mesh_XYpoints2lr], axis=-1)

        # Constructing the points for behind and front boundaries, fix the y_behind and y_front, the z and z are meshed
        x_coord2bf_bd = np.reshape(
            np.linspace(left_b, right_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        z_coord2bf_bd = np.reshape(
            np.linspace(bottom_b, top_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        meshX2bf, meshZ2bf = np.meshgrid(x_coord2bf_bd, z_coord2bf_bd)  # Construct the mesh points
        mesh_Xpoints2bf = np.reshape(meshX2bf, newshape=[-1, 1])
        mesh_Zpoints2bf = np.reshape(meshZ2bf, newshape=[-1, 1])
        y_point2behind_bd = np.ones(shape=[N_tr_u*N_tr_u, 1]) * behind_b
        y_point2front_bd = np.ones(shape=[N_tr_u*N_tr_u, 1]) * front_b

        xyz_behind_b = np.concatenate([mesh_Xpoints2bf, y_point2behind_bd, mesh_Zpoints2bf], axis=-1)
        xyz_front_b = np.concatenate([mesh_Xpoints2bf, y_point2front_bd, mesh_Zpoints2bf], axis=-1)

        # Constructing the points for bottom and top boundaries, fixing the z_bottom and z_top, the y and z are meshed
        x_coord2bt_bd = np.reshape(
            np.linspace(left_b, right_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        y_coord2bt_bd = np.reshape(
            np.linspace(behind_b, front_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        meshX2bt, meshZ2bt = np.meshgrid(x_coord2bt_bd, y_coord2bt_bd)  # Construct the mesh points
        mesh_XYpoints2bt = np.concatenate((np.reshape(meshX2bt, newshape=[-1, 1]),
                                           np.reshape(meshZ2bt, newshape=[-1, 1])), axis=-1)
        z_point2bottom_bd = np.ones(shape=[N_tr_u * N_tr_u, 1]) * bottom_b
        z_point2top_bd = np.ones(shape=[N_tr_u * N_tr_u, 1]) * top_b

        xyz_bottom_b = np.concatenate([mesh_XYpoints2bt, z_point2bottom_bd], axis=-1)
        xyz_top_b = np.concatenate([mesh_XYpoints2bt, z_point2top_bd], axis=-1)

        xyz_bd = np.concatenate([xyz_left_b, xyz_right_b, xyz_behind_b, xyz_front_b, xyz_bottom_b, xyz_top_b],
                                axis=0, dtype=np.float32)

        np.random.shuffle(xyz_bd)
        data["x_u"] = torch.from_numpy(xyz_bd)
        data["y_u"] = u(data["x_u"]) + torch.randn_like(u(data["x_u"])) * like_std  # adding bias

        # Generating the interior mesh points for interested domain
        N_tr_f_mseh = int(np.sqrt(N_tr_f)) + 1  # the number of mesh for every direction

        # the coord points for x direction
        x_coord2in = np.reshape(np.linspace(left_b+0.001, right_b, N_tr_f_mseh, endpoint=False, dtype=np.float32),
                                newshape=(-1, 1))

        # the coord points for y direction
        y_coord2in = np.reshape(np.linspace(behind_b+0.001, front_b, N_tr_f_mseh, endpoint=False, dtype=np.float32),
                                newshape=(-1, 1))

        # the coord points for z direction
        z_coord2in = np.reshape(np.linspace(bottom_b + 0.001, top_b, N_tr_f_mseh, endpoint=False, dtype=np.float32),
                                newshape=(-1, 1))
        mesh_x, mesh_y, mesh_z = np.meshgrid(x_coord2in, y_coord2in, z_coord2in)
        xyz_in = np.concatenate([np.reshape(mesh_x, newshape=[-1, 1]), np.reshape(mesh_y, newshape=[-1, 1]),
                                np.reshape(mesh_z, newshape=[-1, 1])], axis=-1)
        np.random.shuffle(xyz_in)
        data["x_f"] = torch.from_numpy(xyz_in)                                       # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(f(data["x_f"])) * like_std     # adding bias
    else:
        N_tr_u = 500
        y_point2left_right_bd = (front_b - behind_b) * np.random.random(size=[N_tr_u, 1]) + behind_b
        z_point2left_right_bd = (top_b - bottom_b) * np.random.random(size=[N_tr_u, 1]) + bottom_b
        x_points2left_bd = np.ones(shape=[N_tr_u, 1]) * left_b
        x_points2right_bd = np.ones(shape=[N_tr_u, 1]) * right_b
        xyz_left_b = np.concatenate([x_points2left_bd, y_point2left_right_bd, z_point2left_right_bd], axis=-1)
        xyz_right_b = np.concatenate([x_points2right_bd, y_point2left_right_bd, z_point2left_right_bd], axis=-1)

        x_point2behind_front_bd = (right_b - left_b) * np.random.random(size=[N_tr_u, 1]) + left_b
        z_point2behind_front_bd = (top_b - bottom_b) * np.random.random(size=[N_tr_u, 1]) + bottom_b
        y_points2behind_bd = np.ones(shape=[N_tr_u, 1]) * behind_b
        y_points2front_bd = np.ones(shape=[N_tr_u, 1]) * front_b
        xyz_behind_b = np.concatenate([x_point2behind_front_bd, y_points2behind_bd, z_point2behind_front_bd], axis=-1)
        xyz_front_b = np.concatenate([x_point2behind_front_bd, y_points2front_bd, z_point2behind_front_bd], axis=-1)

        x_point2bottom_top_bd = (right_b - left_b) * np.random.random(size=[N_tr_u, 1]) + left_b
        y_point2left_right_bd = (front_b - behind_b) * np.random.random(size=[N_tr_u, 1]) + behind_b
        z_points2bottom_bd = np.ones(shape=[N_tr_u, 1]) * bottom_b
        z_points2top_bd = np.ones(shape=[N_tr_u, 1]) * top_b
        xyz_bottom_b = np.concatenate([x_point2bottom_top_bd, y_point2left_right_bd, z_points2bottom_bd], axis=-1)
        xyz_top_b = np.concatenate([x_point2bottom_top_bd, y_point2left_right_bd, z_points2top_bd], axis=-1)

        xyz_bd = np.concatenate([xyz_left_b, xyz_right_b, xyz_behind_b, xyz_front_b, xyz_bottom_b, xyz_top_b],
                                axis=0, dtype=np.float32)
        data["x_u"] = torch.from_numpy(xyz_bd)
        data["y_u"] = u(data["x_u"]) + torch.randn_like(u(data["x_u"])) * like_std  # adding bias

        x_rand2in = (right_b - left_b) * np.random.random(size=[N_tr_f, 1]) + left_b
        y_rand2in = (front_b - behind_b) * np.random.random(size=[N_tr_f, 1]) + behind_b
        z_rand2in = (top_b - bottom_b) * np.random.random(size=[N_tr_f, 1]) + bottom_b
        xy_in = np.concatenate([x_rand2in, y_rand2in, z_rand2in], axis=-1)
        data["x_f"] = torch.from_numpy(xy_in)  # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(f(data["x_f"])) * like_std  # adding bias

    # exact value of solution, parameter and force-side
    data_val = {}
    if 'gene_mesh_grid' == str.lower(Rdic['opt2gene_test_data']):
        N_val2mesh = int(np.sqrt(N_val))+1
        val_x_coord2in = np.reshape(np.linspace(left_b, right_b, N_val2mesh, endpoint=True, dtype=np.float32),
                                    newshape=(-1, 1))
        val_y_coord2in = np.reshape(np.linspace(behind_b, front_b, N_val2mesh, endpoint=True, dtype=np.float32),
                                    newshape=(-1, 1))
        val_z_coord2in = np.reshape(np.linspace(bottom_b, top_b, N_val2mesh, endpoint=True, dtype=np.float32),
                                    newshape=(-1, 1))
        mesh_x2val, mesh_y2val, mesh_z2val = np.meshgrid(val_x_coord2in, val_y_coord2in, val_z_coord2in)
        val_xyz_points = np.concatenate([np.reshape(mesh_x2val, newshape=[-1, 1]),
                                         np.reshape(mesh_y2val, newshape=[-1, 1]),
                                         np.reshape(mesh_z2val, newshape=[-1, 1])], axis=-1)
        np.random.shuffle(val_xyz_points)
        torch_val_xy = torch.from_numpy(val_xyz_points)
        data_val["x_u"] = torch_val_xy

        data_val["y_u"] = u(data_val["x_u"])
        data_val["x_f"] = torch_val_xy
        data_val["y_f"] = f(data_val["x_f"])
    elif 'load_matlab_data' == str.lower(Rdic['opt2gene_test_data']):
        path2test_data = '../Matlab_data2Bayes3D/'
        torch_val_xyz = Load_data2Mat.get_meshData2Bayes(dim=3, data_path=path2test_data, mesh_number=7, to_torch=True,
                                                         to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False)
        data_val["x_u"] = torch_val_xyz

        data_val["y_u"] = u(data_val["x_u"])
        data_val["x_f"] = torch_val_xyz
        data_val["y_f"] = f(data_val["x_f"])
    else:
        val_x_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
        val_y_coord2in = (front_b - behind_b) * np.random.random(size=[N_val, 1]) + behind_b
        val_z_coord2in = (top_b - bottom_b) * np.random.random(size=[N_val, 1]) + bottom_b
        val_xyz_points = np.concatenate([val_x_coord2in, val_y_coord2in, val_z_coord2in], axis=-1, dtype=np.float32)
        torch_val_xyz = torch.from_numpy(val_xyz_points)
        data_val["x_u"] = torch_val_xyz

        data_val["y_u"] = u(data_val["x_u"])
        data_val["x_f"] = torch_val_xyz
        data_val["y_f"] = f(data_val["x_f"])

    for d in data:
        data[d] = data[d].to(device)
    for d in data_val:
        data_val[d] = data_val[d].to(device)

    net_u = BNN_General_Approximator.General_NN_Approximator(
        NN_Model=Rdic['model'], dim_in=Rdic['indim'], hidden_units=Rdic['hidden_units'], dim_out=Rdic['outdim'],
        Input_actName=Rdic['act_name2Input'], Hidden_actName=Rdic['act_name2Hidden'],
        Output_actName=Rdic['act_name2Output'], mode2init_Weight=Rdic['initW'], mode2init_Bias=Rdic['initB'],
        float_type='float32', with_gpu=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'])

    nets = [net_u]

    # sampling!! The training data is fixed for all training process? why? Can it be varied? No, it is fixed
    time_begin = time.time()
    if 'PINN' == Rdic['mode2update_para']:
        params_hmc, losses = BNN_EstimatePara_Utils.update_paras_by_pinn(
            nets, data, model_loss=model_loss2multiscale, learning_rate=lr, update_lr=open_update_lr,
            step_cum2update_lr=step2update_lr,
            step_gamma2update_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
            total_epochs=max_epoch)
    else:
        params_hmc = BNN_EstimatePara_Utils.update_paras_by_hamilton(
            nets, data, model_loss=model_loss2multiscale, num_samples=num_samples, step_num2per_sample=L,
            learning_rate=lr, update_lr=open_update_lr, step_cum2update_lr=step2update_lr,
            step_gamma2update_lr=gamma2update_lr,
            burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

    pred_list, log_prob_list = BNN_EstimatePara_Utils.predict_model_bpinns(
        nets, params_hmc, data_val, model_loss=model_loss2multiscale, tau_priors=tau_priors,
        tau_likes=tau_likes, pde=pde)

    time_end = time.time()
    run_time = time_end - time_begin

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    BPINN_Test_Log_Print.print_log_validation(Expected, log_out=log_fileout)

    save_load_NetModule.save_bayes_net2file_with_keys(
        outPath=R['FolderName'], model2net=net_u, paras2net=params_hmc, name2model='Solu', learning_rate=lr,
        expected_log_prob=Expected, epoch=R['max_epoch'], opt2update_para=Rdic['mode2update_para'])

    pred_list_u = pred_list[0].cpu().numpy()
    pred_list_f = pred_list[1].cpu().numpy()
    mean2pred_solu = np.mean(pred_list_u, axis=0)
    mean2pred_f = np.mean(pred_list_f, axis=0)
    if Rdic['with_gpu'] is True:
        u_val = data_val['y_u'].cpu().detach().numpy()
        x_val = data_val['x_u'].cpu().detach().numpy()

        f_val = data_val['y_f'].cpu().detach().numpy()
        xf_val = data_val['x_f'].cpu().detach().numpy()
    else:
        u_val = data_val['y_u'].detach().numpy()
        x_val = data_val['x_u'].detach().numpy()

        f_val = data_val['y_f'].detach().numpy()
        xf_val = data_val['x_f'].detach().numpy()

    solu_test = np.reshape(mean2pred_solu, newshape=[-1, 1])
    point_abs_errs2solu_test = np.abs(solu_test - u_val)
    mse2solu_test = np.mean(np.square(solu_test - u_val), axis=0)
    rel2solu_test = np.sqrt(np.mean(np.square(solu_test - u_val), axis=0) / np.mean(np.square(u_val), axis=0))

    force_test = np.reshape(mean2pred_f, newshape=[-1, 1])
    point_abs_err2force_test = np.abs(force_test - f_val)
    mse2force_test = np.mean(np.square(force_test - f_val), axis=0)
    rel2force_test = np.sqrt(np.mean(np.square(force_test - f_val), axis=0) / np.mean(np.square(f_val), axis=0))

    print_log_test_mse_rel(mse2solu=mse2solu_test, rel2solu=rel2solu_test,
                           mse2force=mse2force_test, rel2force=rel2force_test, log_out=log_fileout)

    saveData.saveTestPoints_Solus2mat(x_val, u_val, mean2pred_solu, name2point_data='x',
                                      name2solu_exact='Uexact_' + str(R['act_name2Hidden']),
                                      name2solu_predict='Umean_' + str(R['act_name2Hidden']),
                                      file_name='solu2test', outPath=R['FolderName'])

    saveData.saveMultiTestPoints_Solus2mat(x_val, u_val, pred_list_u, name2point_data='x',
                                           name2solu_exact='Uexact_' + str(R['act_name2Hidden']),
                                           name2solu_predict='MultiU_' + str(R['act_name2Hidden']),
                                           file_name='Multi_Solu2test', outPath=R['FolderName'])

    saveData.saveMultiTestPoints_Solus2mat(x_val, u_val, pred_list_u, name2point_data='x',
                                           name2solu_exact='Fexact_' + str(R['act_name2Hidden']),
                                           name2solu_predict='MultiF_' + str(R['act_name2Hidden']),
                                           file_name='Multi_Force2test', outPath=R['FolderName'])

    saveData.saveTestPoints_Solus2mat(x_val, u_val, mean2pred_solu, name2point_data='x',
                                      name2solu_exact='Uexact_' + str(R['act_name2Hidden']),
                                      name2solu_predict='Umean_' + str(R['act_name2Hidden']),
                                      file_name='solu2test', outPath=R['FolderName'])

    saveData.saveTestPoints_Solus2mat(xf_val, f_val, mean2pred_f, name2point_data='x',
                                      name2solu_exact='Fexact_' + str(R['act_name2Hidden']),
                                      name2solu_predict='Fmean_' + str(R['act_name2Hidden']),
                                      file_name='force2test', outPath=R['FolderName'])

    # plot2Bayes.plot_scatter_solution2test(solu_test, test_xy=x_val, name2solu='BPINN', outPath=R['FolderName'])
    # plot2Bayes.plot_scatter_solution2test(u_val, test_xy=x_val, name2solu='Exact', outPath=R['FolderName'])
    # plot2Bayes.plot_scatter_solution2test(point_abs_errs2solu_test, test_xy=x_val, name2solu='Absolute Error',
    #                                       outPath=R['FolderName'])
    #
    # plot2Bayes.plot_scatter_force2test(force_test, test_xy=xf_val, name2solu='BPINN', outPath=R['FolderName'])
    # plot2Bayes.plot_scatter_force2test(f_val, test_xy=xf_val, name2solu='Exact', outPath=R['FolderName'])
    # plot2Bayes.plot_scatter_force2test(point_abs_err2force_test, test_xy=xf_val, name2solu='Absolute Error',
    #                                    outPath=R['FolderName'])

    if Rdic['with_gpu'] is True:
        x_u = data["x_u"].cpu().detach().numpy()
        y_u = data["y_u"].cpu().detach().numpy()
        x_f = data["x_f"].cpu().detach().numpy()
        y_f = data["y_f"].cpu().detach().numpy()
    else:
        x_u = data["x_u"].detach().numpy()
        y_u = data["y_u"].detach().numpy()
        x_f = data["x_f"].detach().numpy()
        y_f = data["y_f"].detach().numpy()

    saveData.saveTrainData2mat(x_u, y_u, name2point_data='xu_train',
                               name2solu_exact='Utrain_' + str(R['act_name2Hidden']),
                               file_name='solu2train', outPath=R['FolderName'])
    saveData.saveTrainData2mat(x_f, y_f, name2point_data='xf_train',
                               name2solu_exact='Ftrain_' + str(R['act_name2Hidden']),
                               file_name='force2train', outPath=R['FolderName'])

    Aux_Log_Print.log_print_run_time(run_time=run_time, log_file=log_fileout)


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    file2results = 'Results'
    store_file = 'BPINN_MultiScaleElliptic3D'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    current_day_time = datetime.datetime.now()                 # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')

    # The setups of Problems
    # R['PDE_type'] = 'Linear_Poisson'
    # R['equa_name'] = 'PDE1'
    # # R['equa_name'] = 'PDE2'
    # # R['equa_name'] = 'PDE3'

    # R['PDE_type'] = 'Nonlinear_Poisson'
    # R['equa_name'] = 'PDE1'
    # # R['equa_name'] = 'PDE2'
    # # R['equa_name'] = 'PDE3'

    R['PDE_type'] = 'Multiscale'
    R['equa_name'] = 'Multiscale1'
    # # R['equa_name'] = 'Multiscale2'

    # The setups of DNN for approximating solution, parameter and force-side
    R['model'] = 'Net_2Hidden'
    # R['model'] = 'Net_3Hidden'
    # R['model'] = 'Net_4Hidden'
    # R['model'] = 'Net_5Hidden'
    # R['model'] = 'Net_6Hidden'

    R['mode2update_para'] = 'PINN'
    # R['mode2update_para'] = 'Hamilton'
    # R['mode2update_para'] = 'PINN+Hamilton'

    R['noise_level'] = 0.01
    # R['noise_level'] = 0.02
    # R['noise_level'] = 0.05
    # R['noise_level'] = 0.2
    # R['noise_level'] = 0.3
    # R['noise_level'] = 0.5

    OUT_DIR_PDE = os.path.join(OUT_DIR, str(R['equa_name']))  # 路径连
    sys.path.append(OUT_DIR_PDE)
    if not os.path.exists(OUT_DIR_PDE):
        print('---------------------- OUT_DIR_PDE ---------------------:', OUT_DIR_PDE)
        os.mkdir(OUT_DIR_PDE)

    Module_Time = str(R['model']) + '_' + str(R['mode2update_para']) + '_Noise' + str(R['noise_level']) + '_' + str(
        date_time_dir)
    FolderName = os.path.join(OUT_DIR_PDE, Module_Time)  # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)
    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    R['indim'] = 3
    R['outdim'] = 1

    R['opt2sampling'] = 'mesh_grid'
    # R['opt2sampling'] = 'random'

    # R['opt2gene_test_data'] = 'load_matlab_data'
    R['opt2gene_test_data'] = 'gene_mesh_grid'

    if R['model'] == 'Net_2Hidden':
        R['hidden_units'] = [30, 30]
    elif R['model'] == 'Net_3Hidden':
        R['hidden_units'] = [30, 30, 30]
    elif R['model'] == 'Net_4Hidden':
        R['hidden_units'] = [30, 30, 30, 30]
    elif R['model'] == 'Net_5Hidden':
        R['hidden_units'] = [30, 30, 30, 30, 30]
    else:
        R['hidden_units'] = [30, 30, 30, 30, 30, 30]

    # R['act_name2Input'] = 'tanh'
    # R['act_name2Input'] = 'enh_tanh'
    R['act_name2Input'] = 'sin'
    # R['act_name2Input'] = 'gelu'
    # R['act_name2Input'] = 'sinAddcos'

    # R['act_name2Hidden'] = 'tanh'
    R['act_name2Hidden'] = 'sin'
    # R['act_name2Hidden'] = 'silu'
    # R['act_name2Hidden'] = 'enh_tanh'
    # R['act_name2Hidden'] = 'gelu'
    # R['act_name2Hidden'] = 'sinAddcos'

    R['act_name2Output'] = 'linear'

    R['with_gpu'] = True

    # R['initW'] = 'standard_gauss'
    R['initW'] = 'xavier'
    # R['initW'] = None

    R['initB'] = 'uniform'
    # R['initB'] = 'zero'

    if R['mode2update_para'] == 'PINN':
        R['update_lr'] = True
        # R['update_lr'] = False
        R['learning_rate'] = 0.01     # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0025   # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00001  # this is the learning rate for optimizer in PINN  model
        R['step2update_lr'] = 100
        R['gamma2update_lr'] = 0.95

        R['show_lr'] = True

        R['sample_num2hamilton'] = 200
        R['step_num2per_sample'] = 500
        R['burn'] = 50
        # stop_flag2pinn = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
        R['activate_stop2pinn'] = int(0)
        # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
        R['max_epoch'] = 20000
        if 0 != R['activate_stop2pinn']:
            max_epoch2training = input('please input a stop epoch:')
            R['max_epoch'] = int(max_epoch2training)
    elif R['mode2update_para'] == 'Hamilton':
        # R['update_lr'] = True
        R['update_lr'] = False
        # R['learning_rate'] = 0.01  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.005  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.0025  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.001  # this is the learning rate for optimizer in Hamilton  model
        R['learning_rate'] = 0.0005  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.00025  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.0001  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.00005  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.00001  # this is the learning rate for optimizer in Hamilton  model
        R['step2update_lr'] = 20
        R['gamma2update_lr'] = 0.95

        R['show_lr'] = True

        R['sample_num2hamilton'] = 200
        R['step_num2per_sample'] = 500
        R['burn'] = 50
        R['max_epoch'] = 20000

    solve_bayes_pinn(Rdic=R)
    # 隐藏层激活函数：sin 比 tanh 好
    # 采样方法：均匀采样，然后打乱，比随机采样好



