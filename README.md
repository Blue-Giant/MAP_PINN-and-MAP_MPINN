# 目的 
Bayes PINN 进行偏微分方程的求解，包括Poisson， multi-scale Elliptic

# Remark
1. Add the network of  2Hidden_Fourier_Sub and 3Hidden_Fourier_Sub for Bayes_1d_multiscale, correct the wrong of 
   get_get_infos_1d for PDE5 in Eqs2BayesNN

2. BayesNN_Utils.sample_model_bpinns() 拆分为 BayesNN_Utils.update_paras_by_pinn()和
   BayesNN_Utils.update_paras_by_hamilton() 两个功能函数，其中BayesNN_Utils.update_paras_by_pinn()
   不仅返回网络参数，还返回训练过程中的losses。

3. hamilton 代码中，增加了学习率调整功能，stepLR模式；PINN更新参数模块增加了学习率调整功能，stepLR模式

4. BPINN2Poisson1D.py 为求解1维Poisson方程的代码，BPINN_Pre2Poisson1D.py 为装载已经训练好的模型，进行测试

5. 隐藏层激活函数：sin 比 tanh 好

6. 采样方法：均匀采样，然后打乱，比随机采样好
   
7. 对于PINN模式，边界点 N_tr_u = 100, 内部点 N_tr_f = 1600 时， 效果不如 边界点 N_tr_u = 200, 内部点 N_tr_f = 2500 时
   
8. 对于多尺度问题, 学习率要设置的比较小，不然hamilton会崩溃。如 multiscale1，lr=0.00005 才合适；multiscale2，lr=0.000025才合适

## 2025.04.19 
      使用 Hamilton 训练时, 网络的权重参数使用标准正态分布初始化，偏置参数使用均匀分布，效果较好。
      使用 PINN 训练时, 网络的权重参数使用Xavier初始化，偏置参数使用均匀分布，效果较好，

## 2025.10.25
      隐藏层激活函数：sin 比 tanh 好, 也比sin+cos 好, 也比 gelu 好, 也比 enh_tanh 好
      采样方法：均匀采样，然后打乱，比随机采样好

      Under the case of learning rate: 0.0005, the performance of net_2hidden and net_3hidden are good for smooth

      Net2Hidden_FourierBaisi, for PDE3, sin 作为激活函数，效果最好。比 tanh 好, 也比sin+cos 好, 也比 gelu 好, 也比 enh_tanh 好
      Net2Hidden_FourierBaisi, for PDE1, sin 作为激活函数，效果比 全sin激活函数的Net2Hidden好。全tanh激活函数的Net2Hidden效果很差
      对于光滑的函数，使用FourierFeature网络时，如2FF,sigma要小，反之sigma要大些

#     HMC的效果受学习率影响还蛮大的！！！！！

