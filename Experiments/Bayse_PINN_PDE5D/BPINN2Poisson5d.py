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
from Problems import Eqs2BayesNN5d
from utilizers import plot2Bayes
from Log_Print import BPINN_Config_Log_Print
from Log_Print import BPINN_Train_Log_Print
from Log_Print import BPINN_Test_Log_Print
from Log_Print import Aux_Log_Print
from utilizers import saveData
from utilizers import save_load_NetModule
from utilizers import Load_data2Mat


def generate_point_set_rand_without_obs(Nin_u=10, Nbd_u=10, Nin_f=10, region_a=0.0, region_b=1.0):
    """
    Args:
        Nin_u:
        Nbd_u:
        Nin_f:
        region_a:
        region_b:

    Returns:

    """
    x_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    y_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    z_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    s_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    t_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a

    x_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    x_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    x_a_bd = np.concatenate([x_region_a_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    x_b_bd = np.concatenate([x_region_b_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    y_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    y_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    y_a_bd = np.concatenate([x_rand_point2x_bd, y_region_a_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    y_b_bd = np.concatenate([x_rand_point2x_bd, y_region_b_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    z_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    z_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    z_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_region_a_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    z_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_region_b_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    s_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    s_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    s_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_region_a_bd, t_rand_point2x_bd],
                            axis=-1)
    s_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_region_b_bd, t_rand_point2x_bd],
                            axis=-1)

    t_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    t_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    t_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_region_a_bd],
                            axis=-1)
    t_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_region_b_bd],
                            axis=-1)

    xyzst_bd_numpy = np.concatenate([x_a_bd, x_b_bd, y_a_bd, y_b_bd, z_a_bd, z_b_bd, s_a_bd, s_b_bd, t_a_bd, t_b_bd],
                                     axis=0, dtype=np.float32)

    xin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    yin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    zin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    sin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    tin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    xyzst_in2solu_numpy = np.concatenate([xin_rand2solu, yin_rand2solu, zin_rand2solu, sin_rand2solu, tin_rand2solu], axis=-1, dtype=np.float32)

    x_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    y_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    z_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    s_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    t_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    xyzst_in2f_numpy = np.concatenate([x_rand2in, y_rand2in, z_rand2in, s_rand2in, t_rand2in], axis=-1, dtype=np.float32)

    return xyzst_bd_numpy, xyzst_in2solu_numpy, xyzst_in2f_numpy


def generate_point_set_rand(Nin_u=10, Nbd_u=10, Nin_f=10, region_a=0.0, region_b=1.0):
    """
    Args:
        Nin_u:
        Nbd_u:
        Nin_f:
        region_a:
        region_b:

    Returns:

    """
    x_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    y_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    z_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    s_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a
    t_rand_point2x_bd = (region_b - region_a) * np.random.random(size=[Nbd_u, 1]) + region_a

    x_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    x_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    x_a_bd = np.concatenate([x_region_a_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    x_b_bd = np.concatenate([x_region_b_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    y_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    y_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    y_a_bd = np.concatenate([x_rand_point2x_bd, y_region_a_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    y_b_bd = np.concatenate([x_rand_point2x_bd, y_region_b_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    z_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    z_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    z_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_region_a_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)
    z_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_region_b_bd, s_rand_point2x_bd, t_rand_point2x_bd],
                            axis=-1)

    s_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    s_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    s_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_region_a_bd, t_rand_point2x_bd],
                            axis=-1)
    s_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_region_b_bd, t_rand_point2x_bd],
                            axis=-1)

    t_region_a_bd = np.ones(shape=[Nbd_u, 1]) * region_a
    t_region_b_bd = np.ones(shape=[Nbd_u, 1]) * region_b
    t_a_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_region_a_bd],
                            axis=-1)
    t_b_bd = np.concatenate([x_rand_point2x_bd, y_rand_point2x_bd, z_rand_point2x_bd, s_rand_point2x_bd, t_region_b_bd],
                            axis=-1)

    xyzst_bd_numpy = np.concatenate([x_a_bd, x_b_bd, y_a_bd, y_b_bd, z_a_bd, z_b_bd, s_a_bd, s_b_bd, t_a_bd, t_b_bd],
                                     axis=0, dtype=np.float32)

    xin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    yin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    zin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    sin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    tin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    xyzst_in2solu_numpy = np.concatenate([xin_rand2solu, yin_rand2solu, zin_rand2solu, sin_rand2solu, tin_rand2solu], axis=-1, dtype=np.float32)

    x_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    y_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    z_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    s_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    t_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    xyzst_in2f_numpy = np.concatenate([x_rand2in, y_rand2in, z_rand2in, s_rand2in, t_rand2in], axis=-1, dtype=np.float32)

    return xyzst_bd_numpy, xyzst_in2solu_numpy, xyzst_in2f_numpy


def generate_point_set_without_bd_rand(Nin_u=10, Nbd_u=10, Nin_f=10, region_a=0.0, region_b=1.0):
    xin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    yin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    zin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    sin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    tin_rand2solu = (region_b - region_a) * np.random.random(size=[Nin_u, 1]) + region_a
    xyzst_in2solu_numpy = np.concatenate([xin_rand2solu, yin_rand2solu, zin_rand2solu, sin_rand2solu, tin_rand2solu],
                                         axis=-1)

    x_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    y_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    z_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    s_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    t_rand2in = (region_b - region_a) * np.random.random(size=[Nin_f, 1]) + region_a
    xyzst_in2f_numpy = np.concatenate([x_rand2in, y_rand2in, z_rand2in, s_rand2in, t_rand2in], axis=-1)

    return xyzst_in2solu_numpy, xyzst_in2f_numpy


def calculate_snr(signal=None, noise=None):
    power2signal = torch.mean(torch.square(signal), dim=0)
    power2noise = torch.mean(torch.square(noise), dim=0)
    snr = 10.0 * torch.log10(power2signal / power2noise)
    return snr


def print_log_test_mse_rel(mse2solu=None, rel2solu=None, mse2force=None, rel2force=None, log_out=None):
    BPINN_Test_Log_Print.print_log_solu2testing(mse2solu, rel2solu, log_file=log_out)
    BPINN_Test_Log_Print.print_log_fside2testing(mse2force, rel2force, log_file=log_out)


def model_loss2Poisson(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)

    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxyzst = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy, Du_Dz, Du_Ds, Du_Dt = Du2Dxyzst[:, 0:1], Du2Dxyzst[:, 1:2], Du2Dxyzst[:, 2:3], Du2Dxyzst[:, 3:4], Du2Dxyzst[:, 4:5]
    DDu_Dxxyzst = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                      create_graph=True, retain_graph=True)[0]
    DDu_Dyxyzst = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                      create_graph=True, retain_graph=True)[0]
    DDu_Dzxyzst = torch.autograd.grad(Du_Dz, x_f, grad_outputs=torch.ones_like(Du_Dz),
                                      create_graph=True, retain_graph=True)[0]
    DDu_Dsxyzst = torch.autograd.grad(Du_Ds, x_f, grad_outputs=torch.ones_like(Du_Ds),
                                      create_graph=True, retain_graph=True)[0]
    DDu_Dtxyzst = torch.autograd.grad(Du_Dt, x_f, grad_outputs=torch.ones_like(Du_Dt),
                                      create_graph=True, retain_graph=True)[0]
    u_xx = DDu_Dxxyzst[:, 0:1]
    u_yy = DDu_Dyxyzst[:, 1:2]
    u_zz = DDu_Dzxyzst[:, 2:3]
    u_ss = DDu_Dsxyzst[:, 3:4]
    u_tt = DDu_Dtxyzst[:, 4:5]

    pred_f = u_xx + u_yy + u_zz + u_ss + u_tt
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, Du_Dz, u_xx, u_yy, u_zz, pred_u, pred_f
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

    # hyperparameter for random seed, 随机种子设定, noise 就固定了(每次都一样), 网络参数W和B每次初始化都一样。
    hamiltorch.set_random_seed(42)  # 这个命令包括下面三个
    # np.random.seed(123)
    # torch.manual_seed(123)
    # torch.cuda.manual_seed(123)

    prior_std = 1
    like_std = Rdic['noise_level']

    lr = Rdic['learning_rate']  # the learning rate for optimizer in PINN model or Hamilton  model
    step2update_lr = Rdic['step2update_lr']
    gamma2update_lr = Rdic['gamma2update_lr']
    open_update_lr = Rdic['update_lr']

    pde = True
    max_epoch = R['max_epoch']  # the total iteration epoch of training for PINN
    burn = Rdic['burn']  # Hamilton认为前burn=100次估计的不准。取后面的B=600-100=500次作为结果
    num_samples = R['sample_num2hamilton']  # the number of samplings for hamilton sampler,Hamilton 抽样次数
    L = Rdic['step_num2per_sample']     # 每个轨迹走的步数

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    left_b = 0.0         # the left boundary of interested domain
    right_b = 1.0        # the right boundary of interested domain

    Nin2Solu = 5000    # the number of sampled points for dealing with solution
    Nbd2Solu = 200    # the number of sampled points for dealing with solution
    Nin2Force = 5000  # the number of sampled points for dealing with force-side
    N_val = 10000        # the number of sampled points for obtaining real solution, parameter and force-side
    # with_bd2train_set = False
    with_bd2train_set = True    # 有边界点，效果会好

    u_exact, f = Eqs2BayesNN5d.get_infos_5d(equa_name=Rdic['equa_name'])   # get the infos for PDE problem

    data = {}

    if with_bd2train_set is True:
        xyzst2bd, xyzst2solu, xyzst2f = generate_point_set_rand(
            Nin_u=Nin2Solu, Nbd_u=Nbd2Solu, Nin_f=Nin2Force, region_a=left_b, region_b=right_b)
        # xyzst_u = xyzst2bd
        xyzst_u = np.concatenate([xyzst2bd, xyzst2solu], axis=0)
    else:
        xyzst_u, xyzst2f = generate_point_set_without_bd_rand(
            Nin_u=Nin2Solu, Nbd_u=Nbd2Solu, Nin_f=Nin2Force, region_a=left_b, region_b=right_b)

    data["x_u"] = torch.from_numpy(xyzst_u)

    clean_u = u_exact(data["x_u"])  # the exact solution without noise on interior points
    noise2u = torch.randn_like(u_exact(data["x_u"])) * like_std  # the noisy data
    data["y_u"] = clean_u + noise2u  # adding bias

    data["x_f"] = torch.from_numpy(xyzst2f)  # interior points
    clean_f = f(data["x_f"])  # the exact force-side without noise on interior points
    noise2f = torch.randn_like(f(data["x_f"])) * like_std  # the noisy data
    data["y_f"] = clean_f + noise2f  # adding bias

    snr2solu = calculate_snr(signal=clean_u, noise=noise2u)
    snr2force_side = calculate_snr(signal=clean_f, noise=noise2f)
    Aux_Log_Print.print_log_SNR_Solu_Fside(snr2solu=snr2solu, snr2fside=snr2force_side, log_file=log_fileout)
    # exact value of solution, parameter and force-side

    data_val = {}
    val_x_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
    val_y_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
    val_z_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
    val_s_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
    val_t_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b

    val_xyzst_points = np.concatenate([val_x_coord2in, val_y_coord2in, val_z_coord2in, val_s_coord2in, val_t_coord2in],
                                      axis=-1, dtype=np.float32)
    torch_val_xyzst = torch.from_numpy(val_xyzst_points)
    data_val["x_u"] = torch_val_xyzst

    data_val["y_u"] = u_exact(data_val["x_u"])
    data_val["x_f"] = torch_val_xyzst
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
            nets, data, model_loss=model_loss2Poisson, learning_rate=lr, show_learning_rate=Rdic['show_lr'],
            update_lr=open_update_lr,
            step_cum2update_lr=step2update_lr, step_gamma2update_lr=gamma2update_lr, tau_priors=tau_priors,
            tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
    else:
        params_hmc = BNN_EstimatePara_Utils.update_paras_by_hamilton(
            nets, data, model_loss=model_loss2Poisson, num_samples=num_samples, step_num2per_sample=L,
            show_learning_rate=Rdic['show_lr'],
            learning_rate=lr, update_lr=open_update_lr, step_cum2update_lr=step2update_lr,
            step_gamma2update_lr=gamma2update_lr,
            burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

    pred_list, log_prob_list = BNN_EstimatePara_Utils.predict_model_bpinns(
        nets, params_hmc, data_val, model_loss=model_loss2Poisson, tau_priors=tau_priors,
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
    store_file = 'BPINN_Poisson5D'
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
    R['PDE_type'] = 'Poisson'
    # R['equa_name'] = 'Linear_Poisson0'
    R['equa_name'] = 'Linear_Poisson1'   # 1+sin(pi*x)sin(pi*y)sin(pi*z)sin(pi*s)sin(pi*t)
    # R['equa_name'] = 'Linear_Poisson2'  # multiscale
    # R['equa_name'] = 'Linear_Poisson3'
    # R['equa_name'] = 'Linear_Poisson4'

    # The setups of DNN for approximating solution, parameter and force-side
    R['model'] = 'Net_2Hidden'
    # R['model'] = 'Net_3Hidden'
    # R['model'] = 'Net_4Hidden'
    # R['model'] = 'Net_5Hidden'
    # R['model'] = 'Net_6Hidden'

    R['mode2update_para'] = 'PINN'
    # R['mode2update_para'] = 'Hamilton'

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

    R['indim'] = 5
    R['outdim'] = 1

    # R['opt2sampling'] = 'mesh_grid'
    R['opt2sampling'] = 'random'

    # R['opt2gene_test_data'] = 'load_matlab_data'
    # R['opt2gene_test_data'] = 'gene_mesh_grid'
    R['opt2gene_test_data'] = 'gene_rand'

    if R['model'] == 'Net_2Hidden':
        # R['hidden_units'] = [30, 30]
        # R['hidden_units'] = [40, 40]
        # R['hidden_units'] = [50, 50]
        R['hidden_units'] = [60, 60]
    elif R['model'] == 'Net_3Hidden':
        # R['hidden_units'] = [30, 30, 30]
        # R['hidden_units'] = [40, 40, 40]
        # R['hidden_units'] = [50, 50, 50]
        R['hidden_units'] = [60, 60, 60]
    elif R['model'] == 'Net_4Hidden':
        # R['hidden_units'] = [30, 30, 30, 30]
        # R['hidden_units'] = [40, 40, 40, 40]
        # R['hidden_units'] = [50, 50, 50, 50]
        R['hidden_units'] = [60, 60, 60, 60]
    elif R['model'] == 'Net_5Hidden':
        # R['hidden_units'] = [30, 30, 30, 30, 30]
        # R['hidden_units'] = [40, 40, 40, 40, 40]
        # R['hidden_units'] = [50, 50, 50, 50, 50]
        R['hidden_units'] = [60, 60, 60, 60, 60]
    else:
        # R['hidden_units'] = [30, 30, 30, 30, 30, 30]
        # R['hidden_units'] = [40, 40, 40, 40, 40, 40]
        # R['hidden_units'] = [50, 50, 50, 50, 50, 50]
        R['hidden_units'] = [60, 60, 60, 60, 60, 60]

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
    # R['initB'] = 'None'

    if R['mode2update_para'] == 'PINN':
        R['update_lr'] = True
        # R['update_lr'] = False
        # R['learning_rate'] = 0.01  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0025   # this is the learning rate for optimizer in PINN  model
        R['learning_rate'] = 0.005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00001  # this is the learning rate for optimizer in PINN  model
        R['step2update_lr'] = 100
        R['gamma2update_lr'] = 0.97

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



