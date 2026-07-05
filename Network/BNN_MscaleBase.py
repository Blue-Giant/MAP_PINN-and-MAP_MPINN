import torch
import torch.nn as tn
import torch.nn as nn
from Network import ActFUnc_Module
import numpy as np


# This model is the original model from the codes of # B-PINNs (Bayesian Physics-Informed Neural Networks)
# This is the pytorch implementation of B-PINNs with Hamiltonian monte carlo algorithm.
class Net_1Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_1Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[0], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_2Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_3Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[2], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_4Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_4Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[3], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_5Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_5Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[4], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_6Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_6Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.hidden5 = nn.Linear(hidden_layer[4], hidden_layer[5], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[5], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.hidden5.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.hidden5.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden5.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.hidden5.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden5(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_1Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_1Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[0], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_2Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_3Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[2], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_4Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_4Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[3], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_5Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_5Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[4], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_6Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                type2float='float32', to_gpu=False, gpu_no=0, init_W='Xavier', init_B='uniform'):
        super(Net_6Hidden_FourierBasis, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.hidden5 = nn.Linear(hidden_layer[4], hidden_layer[5], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[5], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.hidden5.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.hidden5.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden5.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.hidden5.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

    def forward(self, x):
        H_FF = self.input_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden5(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_1Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_1Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[0], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_2Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_3Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[2], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_4Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_4Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[3], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_5Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_5Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[4], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_6Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_6Hidden_MultiScale, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.hidden5 = nn.Linear(hidden_layer[4], hidden_layer[5], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[5], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.hidden5.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.hidden5.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden5.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.hidden5.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in * self.torch_mixcoe)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden5(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_1Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_1Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(2*hidden_layer[0], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_2Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_3Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[2], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_4Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_4Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[3], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_5Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_5Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[4], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_6Hidden_MultiScale_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None, init_W='Xavier',
                 init_B='uniform'):
        super(Net_6Hidden_MultiScale_FourierBase, self).__init__()
        self.hidden_units = hidden_layer
        self.repeat_Highfreq = repeat_Highfreq

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.hidden4 = nn.Linear(hidden_layer[3], hidden_layer[4], dtype=self.float_type, device=self.opt2device)
        self.hidden5 = nn.Linear(hidden_layer[4], hidden_layer[5], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[5], outdim, dtype=self.float_type, device=self.opt2device)

        try:
            if str.lower(init_W) == 'xavier':
                tn.init.xavier_normal_(self.input_layer.weight)
                tn.init.xavier_normal_(self.hidden1.weight)
                tn.init.xavier_normal_(self.hidden2.weight)
                tn.init.xavier_normal_(self.hidden3.weight)
                tn.init.xavier_normal_(self.hidden4.weight)
                tn.init.xavier_normal_(self.hidden5.weight)
                tn.init.xavier_normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'standard_gauss':
                tn.init.normal_(self.input_layer.weight)
                tn.init.normal_(self.hidden1.weight)
                tn.init.normal_(self.hidden2.weight)
                tn.init.normal_(self.hidden3.weight)
                tn.init.normal_(self.hidden4.weight)
                tn.init.normal_(self.hidden5.weight)
                tn.init.normal_(self.output_layer.weight)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as ew:
            print('Wrong Initialization:', ew)

        try:
            if str.lower(init_B) == 'uniform':
                tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden4.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.hidden5.bias, a=-1.0, b=1.0)
                tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)
            elif str.lower(init_B) == 'zero':
                tn.init.zeros_(self.input_layer.bias)
                tn.init.zeros_(self.hidden1.bias)
                tn.init.zeros_(self.hidden2.bias)
                tn.init.zeros_(self.hidden3.bias)
                tn.init.zeros_(self.hidden4.bias)
                tn.init.zeros_(self.hidden5.bias)
                tn.init.zeros_(self.output_layer.bias)
            elif str.lower(init_W) == 'none':
                print('Without Initialization!!!!!!')
        except ValueError as eb:
            print('Wrong Initialization:', eb)

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq is True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(gpu_no))

    def forward(self, x):
        H_in = self.input_layer(x)
        Hs = H_in * self.torch_mixcoe
        H = torch.cat([torch.sin(Hs), torch.cos(Hs)], dim=-1)

        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden4(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.hidden5(H)  # Activation function is sin or tanh
        H = self.actFunc(H)

        H = self.output_layer(H)
        H = self.actFunc_out(H)
        return H


class Net_1Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_1Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, 2*hidden_layer[0], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[0] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result


class Net_2Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_2Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        w_l1_temp = torch.rand((num2subnets, 2 * hidden_layer[0], hidden_layer[1]), dtype=self.float_type)
        self.W_L1 = tn.Parameter(w_l1_temp, requires_grad=True)
        b_l1_temp = torch.rand((num2subnets, 1, hidden_layer[1]), dtype=self.float_type)
        self.B_L1 = tn.Parameter(b_l1_temp, requires_grad=True)
        stddev_WB_L1 = (2.0 / (hidden_layer[0] + hidden_layer[1])) ** 0.5
        tn.init.normal_(self.W_L1, mean=0.0, std=stddev_WB_L1)
        tn.init.uniform_(self.B_L1, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, hidden_layer[1], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[1] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.W_L1) + self.B_L1
        H = self.actFunc(H)

        H = torch.matmul(H, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result


class Net_3Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_3Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        w_l1_temp = torch.rand((num2subnets, 2 * hidden_layer[0], hidden_layer[1]), dtype=self.float_type)
        self.W_L1 = tn.Parameter(w_l1_temp, requires_grad=True)
        b_l1_temp = torch.rand((num2subnets, 1, hidden_layer[1]), dtype=self.float_type)
        self.B_L1 = tn.Parameter(b_l1_temp, requires_grad=True)
        stddev_WB_L1 = (2.0 / (hidden_layer[0] + hidden_layer[1])) ** 0.5
        tn.init.normal_(self.W_L1, mean=0.0, std=stddev_WB_L1)
        tn.init.uniform_(self.B_L1, -1.0, 1.0)

        w_l2_temp = torch.rand((num2subnets, hidden_layer[1], hidden_layer[2]), dtype=self.float_type)
        self.W_L2 = tn.Parameter(w_l2_temp, requires_grad=True)
        b_l2_temp = torch.rand((num2subnets, 1, hidden_layer[2]), dtype=self.float_type)
        self.B_L2 = tn.Parameter(b_l2_temp, requires_grad=True)
        stddev_WB_L2 = (2.0 / (hidden_layer[1] + hidden_layer[2])) ** 0.5
        tn.init.normal_(self.W_L2, mean=0.0, std=stddev_WB_L2)
        tn.init.uniform_(self.B_L2, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, hidden_layer[2], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[2] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.W_L1) + self.B_L1
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L2) + self.B_L2
        H = self.actFunc(H)

        H = torch.matmul(H, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result


class Net_4Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_4Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        w_l1_temp = torch.rand((num2subnets, 2 * hidden_layer[0], hidden_layer[1]), dtype=self.float_type)
        self.W_L1 = tn.Parameter(w_l1_temp, requires_grad=True)
        b_l1_temp = torch.rand((num2subnets, 1, hidden_layer[1]), dtype=self.float_type)
        self.B_L1 = tn.Parameter(b_l1_temp, requires_grad=True)
        stddev_WB_L1 = (2.0 / (hidden_layer[0] + hidden_layer[1])) ** 0.5
        tn.init.normal_(self.W_L1, mean=0.0, std=stddev_WB_L1)
        tn.init.uniform_(self.B_L1, -1.0, 1.0)

        w_l2_temp = torch.rand((num2subnets, hidden_layer[1], hidden_layer[2]), dtype=self.float_type)
        self.W_L2 = tn.Parameter(w_l2_temp, requires_grad=True)
        b_l2_temp = torch.rand((num2subnets, 1, hidden_layer[2]), dtype=self.float_type)
        self.B_L2 = tn.Parameter(b_l2_temp, requires_grad=True)
        stddev_WB_L2 = (2.0 / (hidden_layer[1] + hidden_layer[2])) ** 0.5
        tn.init.normal_(self.W_L2, mean=0.0, std=stddev_WB_L2)
        tn.init.uniform_(self.B_L2, -1.0, 1.0)

        w_l3_temp = torch.rand((num2subnets, hidden_layer[2], hidden_layer[3]), dtype=self.float_type)
        self.W_L3 = tn.Parameter(w_l3_temp, requires_grad=True)
        b_l3_temp = torch.rand((num2subnets, 1, hidden_layer[3]), dtype=self.float_type)
        self.B_L3 = tn.Parameter(b_l3_temp, requires_grad=True)
        stddev_WB_L3 = (2.0 / (hidden_layer[2] + hidden_layer[3])) ** 0.5
        tn.init.normal_(self.W_L3, mean=0.0, std=stddev_WB_L3)
        tn.init.uniform_(self.B_L3, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, hidden_layer[3], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[3] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.W_L1) + self.B_L1
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L2) + self.B_L2
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L3) + self.B_L3
        H = self.actFunc(H)

        H = torch.matmul(H, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result


class Net_5Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_5Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        w_l1_temp = torch.rand((num2subnets, 2 * hidden_layer[0], hidden_layer[1]), dtype=self.float_type)
        self.W_L1 = tn.Parameter(w_l1_temp, requires_grad=True)
        b_l1_temp = torch.rand((num2subnets, 1, hidden_layer[1]), dtype=self.float_type)
        self.B_L1 = tn.Parameter(b_l1_temp, requires_grad=True)
        stddev_WB_L1 = (2.0 / (hidden_layer[0] + hidden_layer[1])) ** 0.5
        tn.init.normal_(self.W_L1, mean=0.0, std=stddev_WB_L1)
        tn.init.uniform_(self.B_L1, -1.0, 1.0)

        w_l2_temp = torch.rand((num2subnets, hidden_layer[1], hidden_layer[2]), dtype=self.float_type)
        self.W_L2 = tn.Parameter(w_l2_temp, requires_grad=True)
        b_l2_temp = torch.rand((num2subnets, 1, hidden_layer[2]), dtype=self.float_type)
        self.B_L2 = tn.Parameter(b_l2_temp, requires_grad=True)
        stddev_WB_L2 = (2.0 / (hidden_layer[1] + hidden_layer[2])) ** 0.5
        tn.init.normal_(self.W_L2, mean=0.0, std=stddev_WB_L2)
        tn.init.uniform_(self.B_L2, -1.0, 1.0)

        w_l3_temp = torch.rand((num2subnets, hidden_layer[2], hidden_layer[3]), dtype=self.float_type)
        self.W_L3 = tn.Parameter(w_l3_temp, requires_grad=True)
        b_l3_temp = torch.rand((num2subnets, 1, hidden_layer[3]), dtype=self.float_type)
        self.B_L3 = tn.Parameter(b_l3_temp, requires_grad=True)
        stddev_WB_L3 = (2.0 / (hidden_layer[2] + hidden_layer[3])) ** 0.5
        tn.init.normal_(self.W_L3, mean=0.0, std=stddev_WB_L3)
        tn.init.uniform_(self.B_L3, -1.0, 1.0)

        w_l4_temp = torch.rand((num2subnets, hidden_layer[3], hidden_layer[4]), dtype=self.float_type)
        self.W_L4 = tn.Parameter(w_l4_temp, requires_grad=True)
        b_l4_temp = torch.rand((num2subnets, 1, hidden_layer[4]), dtype=self.float_type)
        self.B_L4 = tn.Parameter(b_l4_temp, requires_grad=True)
        stddev_WB_L4 = (2.0 / (hidden_layer[3] + hidden_layer[4])) ** 0.5
        tn.init.normal_(self.W_L4, mean=0.0, std=stddev_WB_L4)
        tn.init.uniform_(self.B_L4, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, hidden_layer[4], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[4] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.W_L1) + self.B_L1
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L2) + self.B_L2
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L3) + self.B_L3
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L4) + self.B_L4
        H = self.actFunc(H)

        H = torch.matmul(H, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result


class Net_6Hidden_FourierSub(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, freq=None, num2subnets=5, init_W='Xavier',
                 init_B='uniform'):
        super(Net_6Hidden_FourierSub, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        # freq是numpy类型还是torch.tensor
        if isinstance(freq, np.ndarray):
            freq_shape = np.shape(freq)
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个列表或者元组(1,2,3,4,5)或者[1,2,3,4,5]或者np.array([1,2,3,4,5]), 形状均为(5,)
                mixcoe = np.expand_dims(freq, axis=-1)   # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = np.expand_dims(mixcoe, axis=-1) # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    np.transpose(freq, 1, 0)
                mixcoe = np.expand_dims(freq, axis=-1)
            else:
                assert(length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = torch.from_numpy(mixcoe)
        elif isinstance(freq, torch.Tensor):
            freq_shape = freq.shape
            length_freq_shape = len(freq_shape)
            if length_freq_shape == 1:
                # 代表是一个torch.tensor([1,2,3,4,5]), 形状(5)
                mixcoe = torch.unsqueeze(freq, dim=-1)  # 扩维变成[[1,2,3,4,5]], 二维矩阵形式，5行1列
                mixcoe = torch.unsqueeze(mixcoe, dim=-1)  # 扩维变成[[[1,2,3,4,5]]], 三维形式， 5页1行1列
            elif length_freq_shape == 2:
                if freq_shape[-1] != 1:
                    torch.transpose(freq, dim0=1, dim1=0)
                mixcoe = torch.unsqueeze(freq, dim=-1)
            else:
                assert (length_freq_shape == 3)
                mixcoe = freq
            self.torch_mixcoe = mixcoe
        else:
            raise TypeError('Unsupported type!!')

        self.torch_mixcoe = self.torch_mixcoe.to(dtype=self.float_type)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        win_temp2tensor = torch.rand((num2subnets, indim, hidden_layer[0]), dtype=self.float_type)
        self.Win = tn.Parameter(win_temp2tensor, requires_grad=True)
        bin_temp2tensor = torch.zeros((num2subnets, 1, hidden_layer[0]), dtype=self.float_type)
        self.Bin = tn.Parameter(bin_temp2tensor, requires_grad=True)
        stddev_WB_In = (2.0 / (indim + hidden_layer[0])) ** 0.5
        tn.init.normal_(self.Win, mean=0.0, std=stddev_WB_In)
        tn.init.uniform_(self.Bin, -1.0, 1.0)

        w_l1_temp = torch.rand((num2subnets, 2*hidden_layer[0], hidden_layer[1]), dtype=self.float_type)
        self.W_L1 = tn.Parameter(w_l1_temp, requires_grad=True)
        b_l1_temp = torch.rand((num2subnets, 1, hidden_layer[1]), dtype=self.float_type)
        self.B_L1 = tn.Parameter(b_l1_temp, requires_grad=True)
        stddev_WB_L1 = (2.0 / (hidden_layer[0] + hidden_layer[1])) ** 0.5
        tn.init.normal_(self.W_L1, mean=0.0, std=stddev_WB_L1)
        tn.init.uniform_(self.B_L1, -1.0, 1.0)

        w_l2_temp = torch.rand((num2subnets, hidden_layer[1], hidden_layer[2]), dtype=self.float_type)
        self.W_L2 = tn.Parameter(w_l2_temp, requires_grad=True)
        b_l2_temp = torch.rand((num2subnets, 1, hidden_layer[2]), dtype=self.float_type)
        self.B_L2 = tn.Parameter(b_l2_temp, requires_grad=True)
        stddev_WB_L2 = (2.0 / (hidden_layer[1] + hidden_layer[2])) ** 0.5
        tn.init.normal_(self.W_L2, mean=0.0, std=stddev_WB_L2)
        tn.init.uniform_(self.B_L2, -1.0, 1.0)

        w_l3_temp = torch.rand((num2subnets, hidden_layer[2], hidden_layer[3]), dtype=self.float_type)
        self.W_L3 = tn.Parameter(w_l3_temp, requires_grad=True)
        b_l3_temp = torch.rand((num2subnets, 1, hidden_layer[3]), dtype=self.float_type)
        self.B_L3 = tn.Parameter(b_l3_temp, requires_grad=True)
        stddev_WB_L3 = (2.0 / (hidden_layer[2] + hidden_layer[3])) ** 0.5
        tn.init.normal_(self.W_L3, mean=0.0, std=stddev_WB_L3)
        tn.init.uniform_(self.B_L3, -1.0, 1.0)

        w_l4_temp = torch.rand((num2subnets, hidden_layer[3], hidden_layer[4]), dtype=self.float_type)
        self.W_L4 = tn.Parameter(w_l4_temp, requires_grad=True)
        b_l4_temp = torch.rand((num2subnets, 1, hidden_layer[4]), dtype=self.float_type)
        self.B_L4 = tn.Parameter(b_l4_temp, requires_grad=True)
        stddev_WB_L4 = (2.0 / (hidden_layer[3] + hidden_layer[4])) ** 0.5
        tn.init.normal_(self.W_L4, mean=0.0, std=stddev_WB_L4)
        tn.init.uniform_(self.B_L4, -1.0, 1.0)

        w_l5_temp = torch.rand((num2subnets, hidden_layer[4], hidden_layer[5]), dtype=self.float_type)
        self.W_L5 = tn.Parameter(w_l5_temp, requires_grad=True)
        b_l5_temp = torch.rand((num2subnets, 1, hidden_layer[5]), dtype=self.float_type)
        self.B_L5 = tn.Parameter(b_l5_temp, requires_grad=True)
        stddev_WB_L5 = (2.0 / (hidden_layer[4] + hidden_layer[5])) ** 0.5
        tn.init.normal_(self.W_L5, mean=0.0, std=stddev_WB_L5)
        tn.init.uniform_(self.B_L5, -1.0, 1.0)

        wout_temp = torch.rand((num2subnets, hidden_layer[5], outdim), dtype=self.float_type)
        self.Wout = tn.Parameter(wout_temp, requires_grad=True)
        bout_temp = torch.rand((num2subnets, 1, outdim), dtype=self.float_type)
        self.Bout = tn.Parameter(bout_temp, requires_grad=True)
        stddev_WB_Out = (2.0 / (hidden_layer[5] + outdim)) ** 0.5
        tn.init.normal_(self.Wout, mean=0.0, std=stddev_WB_Out)
        tn.init.uniform_(self.Bout, -1.0, 1.0)

    def forward(self, x):
        H_in = torch.matmul(x, self.Win) + self.Bin
        H_FF = torch.cat([torch.cos(H_in * self.torch_mixcoe), torch.sin(H_in * self.torch_mixcoe)], dim=-1)

        H = torch.matmul(H_FF, self.W_L1) + self.B_L1
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L2) + self.B_L2
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L3) + self.B_L3
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L4) + self.B_L4
        H = self.actFunc(H)

        H = torch.matmul(H, self.W_L5) + self.B_L5
        H = self.actFunc(H)

        H = torch.matmul(H, self.Wout) + self.Bout
        Hout = self.actFunc_out(H)

        out_result = torch.multiply(Hout, 1.0 / self.torch_mixcoe)
        out_result = torch.mean(out_result, dim=0)
        return out_result