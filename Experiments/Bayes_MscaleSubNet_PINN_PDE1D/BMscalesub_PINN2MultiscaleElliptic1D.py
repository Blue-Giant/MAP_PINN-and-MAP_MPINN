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
from Network import BNN_Mscale_Approximator
from Problems import Eqs2BayesNN
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


def model_loss_0p5(data, fmodel, params_unflattened, tau_likes, gradients, params_single):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)

    x_f = data["x_f"]
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]

    Aeps = 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5))

    temp1 = torch.sin(2.0 * torch.pi * x_f / 0.5) * (2.0 * torch.pi / 0.5)
    temp2 = (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5)) * (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5))
    dAeps_dx = temp1/temp2

    pred_f = -1.0*(Aeps*u_xx + dAeps_dx*u_x)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, Aeps, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


def model_loss_0p1(data, fmodel, params_unflattened, tau_likes, gradients, params_single):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)

    x_f = data["x_f"]
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]

    Aeps = 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1))

    temp1 = torch.sin(2.0 * torch.pi * x_f / 0.1) * (2.0 * torch.pi / 0.1)
    temp2 = (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1)) * (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1))
    dAeps_dx = temp1/temp2

    pred_f = -1.0*(Aeps*u_xx + dAeps_dx*u_x)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, Aeps, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


def solve_bayes_mscale_sub_fourier_pinn(Rdic=None):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['act_name2Hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    BPINN_Config_Log_Print.dictionary_out2file(R, log_fileout)

    # using PINN mode to update the parameters of DNN or not
    if Rdic['with_gpu'] is True:
        assert (torch.cuda.is_available() is True)
        print(f"Is CUDA available?: {torch.cuda.is_available()}")
        device = 'cuda:0'
    else:
        device = "cpu"

    # hyperparameter for random seed, 随机种子设定, noise 就固定了(每次都一样), 网络参数W和B每次初始化都一样。
    hamiltorch.set_random_seed(42)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(0)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(100)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(101)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(121)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(122)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(124)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(125)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(111)  # 这个命令包括下面三个
    # hamiltorch.set_random_seed(1024)  # 这个命令包括下面三个
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
    # B= num_samples - burn 为总的抽样次数，即BxNxD, 其中B为总的抽样次数，N为测试点个数，D为解的维度
    L = Rdic['step_num2per_sample']

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    lb = 0.0        # the left boundary of interested interval
    ub = 1.0        # the right boundary of interested interval
    N_tr_u = 50   # the number of sampled points for dealing with solution
    N_tr_f = 50   # the number of sampled points for dealing with force-side
    N_val = 1000  # the number of sampled points for obtaining real solution, parameter and force-side

    # eps = Rdic['epsilon']
    eps = Rdic['epsilon']
    u = lambda x: x-torch.mul(x, x)+(0.25*eps/torch.pi)*torch.sin(torch.pi*2.0*x/eps) - \
                  (0.5*eps/torch.pi)*x*torch.sin(torch.pi*2.0*x/eps)-eps*eps/(4.0*torch.pi**2)*torch.cos(torch.pi*2.0*x/eps)+\
                  0.25*eps/(torch.pi ** 2)
    f = lambda x: torch.ones_like(x)
    k = lambda x: 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x / eps))

    data = {}
    if 'equidistance' == str.lower(Rdic['opt2sampling']):
        xu = np.reshape(np.linspace(lb, ub, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xu)
        data["x_u"] = torch.from_numpy(xu)              # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                        # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std  # the noisy data
        data["y_u"] = clean_u + noise2u                 # adding bias

        xf = np.reshape(np.linspace(lb, ub, N_tr_f, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xf)
        data["x_f"] = torch.from_numpy(xf)              # interior points for governed equation
        clean_f = f(data["x_f"])                        # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std  # the noisy data
        data["y_f"] = clean_f + noise2f                 # adding bias
    elif 'lhs' == str.lower(Rdic['opt2sampling']):
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)  # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                        # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std  # the noisy data
        data["y_u"] = clean_u + noise2u                 # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb  # interior points
        clean_f = f(data["x_f"])                              # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std        # the noisy data
        data["y_f"] = clean_f + noise2f                       # adding bias
    elif 'load_random_collection' == str.lower(Rdic['opt2sampling']):
        if Rdic['load_random_obs_data_num'] == 50:
            data_path = '../matdata50Points2Bayes1D_Multiscale'
        elif Rdic['load_random_obs_data_num'] == 75:
            data_path = '../matdata75Points2Bayes1D_Multiscale'
        elif Rdic['load_random_obs_data_num'] == 100:
            data_path = '../matdata100Points2Bayes1D_Multiscale'
        elif Rdic['load_random_obs_data_num'] == 125:
            data_path = '../matdata125Points2Bayes1D_Multiscale'
        elif Rdic['load_random_obs_data_num'] == 150:
            data_path = '../matdata150Points2Bayes1D_Multiscale'
        elif Rdic['load_random_obs_data_num'] == 175:
            data_path = '../matdata175Points2Bayes1D_Multiscale'
        else:
            data_path = '../matdata200Points2Bayes1D_Multiscale'
        x_data2solu, x_data2fside, x_data2coef, noise2u, noise2f, noise2k = \
            Load_data2Mat.get_random_points_1D_noise(data_path=data_path, noise_level=Rdic['noise_level'],
                                                     to_torch=True, to_float=True, to_cuda=Rdic['with_gpu'],
                                                     gpu_no=0, use_grad2x=False)
        data["x_u"] = x_data2solu
        clean_u = u(data["x_u"])                        # the exact solution without noise on interior points
        data["y_u"] = clean_u + noise2u         # adding bias

        data["x_f"] = x_data2fside                      # interior points
        clean_f = f(data["x_f"])
        data["y_f"] = clean_f + noise2f        # adding bias
    else:
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)    # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                          # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std    # the noisy data
        data["y_u"] = clean_u + noise2u                   # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb    # interior points
        clean_f = f(data["x_f"])                                # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std          # the noisy data
        data["y_f"] = clean_f + noise2f                         # adding bias

    snr2solu = calculate_snr(signal=clean_u, noise=noise2u)
    snr2force_side = calculate_snr(signal=clean_f, noise=noise2f)
    Aux_Log_Print.print_log_SNR_Solu_Fside(snr2solu=snr2solu, snr2fside=snr2force_side, log_file=log_fileout)

    # exact value of solution, parameter and force-side
    data_val = {}
    data_val["x_u"] = torch.linspace(lb, ub, N_val).view(-1, 1)
    data_val["y_u"] = u(data_val["x_u"])
    data_val["x_f"] = torch.linspace(lb, ub, N_val).view(-1, 1)
    data_val["y_f"] = f(data_val["x_f"])

    for d in data:
        data[d] = data[d].to(device)
    for d in data_val:
        data_val[d] = data_val[d].to(device)

    scale2u = Rdic['scales']
    net_u = BNN_Mscale_Approximator.Multiscale_FourierSub_NN_Approximator(
        NN_Model=Rdic['model'], dim_in=Rdic['indim'], hidden_units=Rdic['hidden_units'], dim_out=Rdic['outdim'],
        Input_actName=Rdic['act_name2Input'], Hidden_actName=Rdic['act_name2Hidden'],
        Output_actName=Rdic['act_name2Output'], mode2init_Weight=Rdic['initW'], mode2init_Bias=Rdic['initB'],
        float_type='float32', with_gpu=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'], scales=scale2u, subnet_num=len(scale2u))

    nets = [net_u]
    time_begin = time.time()
    if 'PINN' == Rdic['mode2update_para']:
        if eps == 0.5:
            params_hmc, losses = BNN_EstimatePara_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p5, learning_rate=lr, show_learning_rate=Rdic['show_lr'],
                update_lr=open_update_lr,
                step_cum2update_lr=step2update_lr, step_gamma2update_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
        elif eps == 0.1:
            params_hmc, losses = BNN_EstimatePara_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p1, learning_rate=lr, show_learning_rate=Rdic['show_lr'],
                update_lr=open_update_lr,
                step_cum2update_lr=step2update_lr, step_gamma2update_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
        else:
            params_hmc, losses = BNN_EstimatePara_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p1, learning_rate=lr, show_learning_rate=Rdic['show_lr'],
                update_lr=open_update_lr,
                step_cum2update_lr=step2update_lr, step_gamma2update_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
    else:
        assert 'hamilton' == str.lower(Rdic['mode2update_para'])
        if eps == 0.5:
            params_hmc = BNN_EstimatePara_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p5, num_samples=num_samples, step_num2per_sample=L,
                show_learning_rate=Rdic['show_lr'],
                learning_rate=lr, update_lr=open_update_lr, step_cum2update_lr=step2update_lr,
                step_gamma2update_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)
        elif eps == 0.1:
            params_hmc = BNN_EstimatePara_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p1, num_samples=num_samples, step_num2per_sample=L,
                show_learning_rate=Rdic['show_lr'],
                learning_rate=lr, update_lr=open_update_lr, step_gamma2update_lr=gamma2update_lr,
                step_cum2update_lr=step2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)
        else:
            params_hmc = BNN_EstimatePara_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p1, num_samples=num_samples, step_num2per_sample=L,
                show_learning_rate=Rdic['show_lr'],
                learning_rate=lr, update_lr=open_update_lr, step_cum2update_lr=step2update_lr,
                step_gamma2update_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

    if eps == 0.5:
        pred_list, log_prob_list = BNN_EstimatePara_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p5, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)
    elif eps == 0.1:
        pred_list, log_prob_list = BNN_EstimatePara_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p1, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)
    else:
        pred_list, log_prob_list = BNN_EstimatePara_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p1, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

    time_end = time.time()
    run_time = time_end - time_begin

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    BPINN_Test_Log_Print.print_log_validation(Expected, log_out=log_fileout)

    pred_list_u = pred_list[0].cpu().numpy()
    pred_list_f = pred_list[1].cpu().numpy()

    mean2pred_u = np.reshape(pred_list_u.mean(0).squeeze().T, newshape=[-1, 1])
    mean2pred_f = np.reshape(pred_list_f.mean(0).squeeze().T, newshape=[-1, 1])

    if Rdic['with_gpu'] is True:
        x_val = data_val["x_u"].cpu().detach().numpy()
        u_val = data_val["y_u"].cpu().detach().numpy()
        f_val = data_val["y_f"].cpu().detach().numpy()

        x_u = data["x_u"].cpu().detach().numpy()
        y_u = data["y_u"].cpu().detach().numpy()
        x_f = data["x_f"].cpu().detach().numpy()
        y_f = data["y_f"].cpu().detach().numpy()
    else:
        # plot
        x_val = data_val["x_u"].detach().numpy()
        u_val = data_val["y_u"].detach().numpy()
        f_val = data_val["y_f"].detach().numpy()

        x_u = data["x_u"].detach().numpy()
        y_u = data["y_u"].detach().numpy()
        x_f = data["x_f"].detach().numpy()
        y_f = data["y_f"].detach().numpy()

    diff2mean_pred_U = mean2pred_u - u_val
    diff2mean_pred_F = mean2pred_f - f_val

    mse2U = np.mean(np.square(diff2mean_pred_U), axis=0)
    rel2U = np.sqrt(mse2U/np.mean(np.square(u_val)))

    mse2F = np.mean(np.square(diff2mean_pred_F))
    rel2F = np.sqrt(mse2F / np.mean(np.square(f_val)))

    print_log_test_mse_rel(mse2solu=mse2U, rel2solu=rel2U, mse2force=mse2F, rel2force=rel2F, log_out=log_fileout)

    saveData.saveTestPoints_Solus2mat(x_val, u_val, mean2pred_u, name2point_data='x',
                                      name2solu_exact='Uexact_'+str(R['act_name2Hidden']),
                                      name2solu_predict='Umean_'+str(R['act_name2Hidden']),
                                      file_name='solu2test', outPath=R['FolderName'])

    saveData.saveTestPoints_Solus2mat(x_val, f_val, mean2pred_f, name2point_data='x',
                                      name2solu_exact='Fexact_' + str(R['act_name2Hidden']),
                                      name2solu_predict='Fmean_' + str(R['act_name2Hidden']),
                                      file_name='force2test', outPath=R['FolderName'])

    saveData.saveTrainData2mat(x_u, y_u, name2point_data='xu_train',
                               name2solu_exact='Utrain_' + str(R['act_name2Hidden']),
                               file_name='solu2train', outPath=R['FolderName'])

    saveData.saveTrainData2mat(x_f, y_f, name2point_data='xf_train',
                               name2solu_exact='Ftrain_' + str(R['act_name2Hidden']),
                               file_name='force2train', outPath=R['FolderName'])

    plot2Bayes.plot2u(x_val=x_val, u_val=u_val, pred_list_u=pred_list_u, x_u=x_u, y_u=y_u, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='solu2u')

    plot2Bayes.plot2f(x_val=x_val, f_val=f_val, pred_list_f=pred_list_f, x_f=x_f, y_f=y_f, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='force')

    save_load_NetModule.save_bayes_net2file_with_keys(
        outPath=R['FolderName'], model2net=net_u, paras2net=params_hmc, name2model='Solu', learning_rate=lr,
        expected_log_prob=Expected, epoch=R['max_epoch'], opt2update_para=Rdic['mode2update_para'])

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
    store_file = 'BMscaleSubFourierPINN2MultiScaleElliptic1D'
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
    R['PDE_type'] = 'MultiScale'
    R['equa_name'] = 'MultiScale_Elliptic'

    # The setups of DNN for approximating solution, parameter and force-side
    R['model'] = 'Net_2Hidden_Fourier_sub'
    # R['model'] = 'Net_3Hidden_Fourier_sub'
    # R['model'] = 'Net_4Hidden_Fourier_sub'
    # R['model'] = 'Net_5Hidden_Fourier_sub'
    # R['model'] = 'Net_6Hidden_Fourier_sub'

    R['mode2update_para'] = 'PINN'
    # R['mode2update_para'] = 'Hamilton'

    # R['noise_level'] = 0.01
    # R['noise_level'] = 0.02
    R['noise_level'] = 0.05
    # R['noise_level'] = 0.1

    OUT_DIR_PDE = os.path.join(OUT_DIR, str(R['equa_name']))  # 路径连
    sys.path.append(OUT_DIR_PDE)
    if not os.path.exists(OUT_DIR_PDE):
        print('---------------------- OUT_DIR_PDE ---------------------:', OUT_DIR_PDE)
        os.mkdir(OUT_DIR_PDE)

    Module_Time = str(R['model']) + '_' + str(R['mode2update_para']) + '_Noise' + str(R['noise_level']) + '_' + str(date_time_dir)
    FolderName = os.path.join(OUT_DIR_PDE, Module_Time)          # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)
    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    R['indim'] = 1
    R['outdim'] = 1

    # R['opt2sampling'] = 'equidistance'
    # R['opt2sampling'] = 'lhs'
    # R['opt2sampling'] = 'random'
    R['opt2sampling'] = 'load_random_collection'

    if R['opt2sampling'] == 'load_random_collection':
        # R['load_random_obs_data_num'] = 50
        # R['load_random_obs_data_num'] = 75
        R['load_random_obs_data_num'] = 100
        # R['load_random_obs_data_num'] = 125
        # R['load_random_obs_data_num'] = 150
        # R['load_random_obs_data_num'] = 175
        # R['load_random_obs_data_num'] = 200

    if R['model'] == 'Net_2Hidden_Fourier_sub':
        # R['hidden_units'] = [6, 6]
        R['hidden_units'] = [8, 8]
        # R['hidden_units'] = [10, 10]
    elif R['model'] == 'Net_3Hidden_Fourier_sub':
        # R['hidden_units'] = [6, 6, 6]
        # R['hidden_units'] = [8, 8, 8]
        R['hidden_units'] = [10, 10, 10]
    elif R['model'] == 'Net_4Hidden_Fourier_sub':
        R['hidden_units'] = [8, 8, 8, 8]
        # R['hidden_units'] = [10, 10, 10, 10]
    elif R['model'] == 'Net_5Hidden_Fourier_sub':
        R['hidden_units'] = [8, 8, 8, 8, 8]
        # R['hidden_units'] = [10, 10, 10, 10, 10]
    elif R['model'] == 'Net_6Hidden_Fourier_sub':
        R['hidden_units'] = [8, 8, 8, 8, 8, 8]
        # R['hidden_units'] = [10, 10, 10, 10, 10, 10]

    R['act_name2Input'] = 'fourier'

    # R['act_name2Hidden'] = 'tanh'
    # R['act_name2Hidden'] = 'enh_tanh'
    R['act_name2Hidden'] = 'sin'
    # R['act_name2Hidden'] = 'silu'
    # R['act_name2Hidden'] = 'gelu'
    # R['act_name2Hidden'] = 'sinAddcos'

    R['act_name2Output'] = 'linear'

    R['with_gpu'] = True

    R['initWB'] = True

    # R['initW'] = 'standard_gauss'
    R['initW'] = 'xavier'
    # R['initW'] = None

    R['initB'] = 'uniform'
    # R['initB'] = 'zero'

    if R['mode2update_para'] == 'PINN':
        R['update_lr'] = True
        # R['update_lr'] = False
        # R['learning_rate'] = 0.01     # this is the learning rate for optimizer in PINN  model
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

    R['epsilon'] = 0.1
    # R['epsilon'] = 0.5

    # R['scales'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    # R['scales'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    # R['scales'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    R['scales'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # R['scales'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # R['scales'] = np.array([1, 2, 3, 4, 5])
    solve_bayes_mscale_sub_fourier_pinn(Rdic=R)


