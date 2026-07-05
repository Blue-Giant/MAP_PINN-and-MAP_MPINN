from Network.BNN_GeneralBase import *


def General_NN_Approximator(NN_Model='DNN', dim_in=2, hidden_units=None, dim_out=1, Input_actName='sin',
                            Hidden_actName='sin', Output_actName='sin', mode2init_Weight='xavier',
                            mode2init_Bias='xavier', float_type='float32', with_gpu=False, gpu_no=0):
    assert hidden_units is not None
    if 'NET_2HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_2Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                               actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName, 
                               type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, init_W=mode2init_Weight,
                               init_B=mode2init_Bias)
    elif 'NET_3HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_3Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                               actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                               type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, init_W=mode2init_Weight,
                               init_B=mode2init_Bias)
    elif 'NET_4HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_4Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                               actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                               type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, init_W=mode2init_Weight,
                               init_B=mode2init_Bias)
    elif 'NET_5HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_5Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                               actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                               type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, init_W=mode2init_Weight,
                               init_B=mode2init_Bias)
    else:
        Solu_Net = Net_6Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                               actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                               type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no, init_W=mode2init_Weight,
                               init_B=mode2init_Bias)
    return Solu_Net


def Fourier_Basis_NN_Approximator(NN_Model='DNN', dim_in=2, hidden_units=None, dim_out=1, Input_actName='sin',
                                  Hidden_actName='sin', Output_actName='sin', mode2init_Weight='xavier',
                                  mode2init_Bias='xavier', float_type='float32', with_gpu=False, gpu_no=0):
    if 'FOURIER_BASIS_NET_2HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_2Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                            actName2in=Input_actName, actName=Hidden_actName, actName2out=Output_actName,
                                            type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                            init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'FOURIER_BASIS_NET_3HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_3Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                            actName2in=Input_actName, actName=Hidden_actName,
                                            actName2out=Output_actName,
                                            type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                            init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'FOURIER_BASIS_NET_4HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_4Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                            actName2in=Input_actName, actName=Hidden_actName,
                                            actName2out=Output_actName,
                                            type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                            init_W=mode2init_Weight, init_B=mode2init_Bias)
    elif 'FOURIER_BASIS_NET_5HIDDEN' == str.upper(NN_Model):
        Solu_Net = Net_5Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                            actName2in=Input_actName, actName=Hidden_actName,
                                            actName2out=Output_actName,
                                            type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                            init_W=mode2init_Weight, init_B=mode2init_Bias)
    else:
        Solu_Net = Net_6Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden_units,
                                            actName2in=Input_actName, actName=Hidden_actName,
                                            actName2out=Output_actName,
                                            type2float=float_type, to_gpu=with_gpu, gpu_no=gpu_no,
                                            init_W=mode2init_Weight, init_B=mode2init_Bias)
    return Solu_Net
