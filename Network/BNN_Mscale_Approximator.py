from Network.BNN_MscaleBase import *


def Multiscale_NN_Approximator(NN_Model='DNN', dim_in=2, hidden_units=None, dim_out=1, Input_actName='sin',
                               Hidden_actName='sin', Output_actName='sin', mode2init_Weight='xavier',
                               mode2init_Bias='xavier', float_type='float32', with_gpu=False, gpu_no=0,
                               scales=None, repeatHighFreq=False):
    if 'NET_2HIDDEN_MULTISCALE' == str.upper(NN_Model):
        solu_net = Net_2Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          repeat_Highfreq=repeatHighFreq, freq=scales,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_3HIDDEN_MULTISCALE' == str.upper(NN_Model):
        solu_net = Net_3Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          repeat_Highfreq=repeatHighFreq, freq=scales,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_4HIDDEN_MULTISCALE' == str.upper(NN_Model):
        solu_net = Net_4Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          repeat_Highfreq=repeatHighFreq, freq=scales,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_5HIDDEN_MULTISCALE' == str.upper(NN_Model):
        solu_net = Net_5Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          repeat_Highfreq=repeatHighFreq, freq=scales,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    else:
        solu_net = Net_6Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          repeat_Highfreq=repeatHighFreq, freq=scales,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    return solu_net


def Multiscale_FourierBase_NN_Approximator(NN_Model='DNN', dim_in=2, hidden_units=None, dim_out=1, Input_actName='sin',
                                           Hidden_actName='sin', Output_actName='sin', mode2init_Weight='xavier',
                                           mode2init_Bias='xavier', float_type='float32', with_gpu=False, gpu_no=0,
                                           scales=None, repeatHighFreq=False):
    if 'NET_2HIDDEN_MULTISCALE_FOURIER_BASE' == str.upper(NN_Model):
        solu_net = Net_2Hidden_MultiScale_FourierBase(
            indim=dim_in, outdim=dim_out, hidden_layer=hidden_units, actName2in=Input_actName, actName=Hidden_actName,
            actName2out=Output_actName, type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, repeat_Highfreq=repeatHighFreq,
            freq=scales, init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_3HIDDEN_MULTISCALE_FOURIER_BASE' == str.upper(NN_Model):
        solu_net = Net_3Hidden_MultiScale_FourierBase(
            indim=dim_in, outdim=dim_out, hidden_layer=hidden_units, actName2in=Input_actName, actName=Hidden_actName,
            actName2out=Output_actName, type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, repeat_Highfreq=repeatHighFreq,
            freq=scales, init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_4HIDDEN_MULTISCALE_FOURIER_BASE' == str.upper(NN_Model):
        solu_net = Net_4Hidden_MultiScale_FourierBase(
            indim=dim_in, outdim=dim_out, hidden_layer=hidden_units, actName2in=Input_actName, actName=Hidden_actName,
            actName2out=Output_actName, type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, repeat_Highfreq=repeatHighFreq,
            freq=scales, init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_5HIDDEN_MULTISCALE_FOURIER_BASE' == str.upper(NN_Model):
        solu_net = Net_5Hidden_MultiScale_FourierBase(
            indim=dim_in, outdim=dim_out, hidden_layer=hidden_units, actName2in=Input_actName, actName=Hidden_actName,
            actName2out=Output_actName, type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, repeat_Highfreq=repeatHighFreq,
            freq=scales, init_W=mode2init_Weight, init_B=mode2init_Bias)
    else:
        solu_net = Net_6Hidden_MultiScale_FourierBase(
            indim=dim_in, outdim=dim_out, hidden_layer=hidden_units, actName2in=Input_actName, actName=Hidden_actName,
            actName2out=Output_actName, type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
            repeat_Highfreq=repeatHighFreq,
            freq=scales, init_W=mode2init_Weight, init_B=mode2init_Bias)
    return solu_net


def Multiscale_FourierSub_NN_Approximator(NN_Model='DNN', dim_in=2, hidden_units=None, dim_out=1, Input_actName='sin',
                                          Hidden_actName='sin', Output_actName='sin', mode2init_Weight='xavier',
                                          mode2init_Bias='xavier', float_type='float32', with_gpu=False, gpu_no=0,
                                          scales=None, subnet_num=5):
    if 'NET_2HIDDEN_FOURIER_SUB' == str.upper(NN_Model):
        Solu_Net = Net_2Hidden_FourierSub(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          freq=scales, num2subnets=subnet_num,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_3HIDDEN_FOURIER_SUB' == str.upper(NN_Model):
        Solu_Net = Net_3Hidden_FourierSub(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          freq=scales, num2subnets=subnet_num,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)

    elif 'NET_4HIDDEN_FOURIER_SUB' == str.upper(NN_Model):
        Solu_Net = Net_4Hidden_FourierSub(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          freq=scales, num2subnets=subnet_num,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'NET_5HIDDEN_FOURIER_SUB' == str.upper(NN_Model):
        Solu_Net = Net_5Hidden_FourierSub(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          freq=scales, num2subnets=subnet_num,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    else:
        Solu_Net = Net_6Hidden_FourierSub(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                          actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                          type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                          freq=scales, num2subnets=subnet_num,
                                          init_W=mode2init_Weight, init_B=mode2init_Bias)
    return Solu_Net
