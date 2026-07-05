import torch
import numpy as np


def get_infos_5d(equa_name='PDE1'):
    if str.upper(equa_name) == 'LINEAR_POISSON0':
        # Laplace U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)sin(pi*z)sin(pi*s)sin(pi*t)
        utrue = lambda x: torch.sin(torch.pi*x[:, 0:1]) * torch.sin(torch.pi*x[:, 1:2]) * torch.sin(torch.pi*x[:, 2:3]) \
                          * torch.sin(torch.pi*x[:, 3:4]) * torch.sin(torch.pi*x[:, 4:5])
        fside = lambda x: -5.0*torch.pi*torch.pi*torch.sin(torch.pi*x[:, 0:1]) * torch.sin(torch.pi*x[:, 1:2]) \
                          * torch.sin(torch.pi*x[:, 2:3]) * torch.sin(torch.pi*x[:, 3:4]) * torch.sin(torch.pi*x[:, 4:5])
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON1':
        # Laplace U = f
        # u(x,y,z)=1+sin(pi*x)sin(pi*y)sin(pi*z)sin(pi*s)sin(pi*t)
        utrue = lambda x: torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * torch.sin(
            torch.pi * x[:, 2:3]) * torch.sin(torch.pi * x[:, 3:4]) * torch.sin(torch.pi * x[:, 4:5])+1.0
        fside = lambda x: -5.0*torch.pi*torch.pi*torch.sin(torch.pi*x[:, 0:1])*torch.sin(torch.pi*x[:, 1:2])*\
                          torch.sin(torch.pi*x[:, 2:3])*torch.sin(torch.pi*x[:, 3:4])*torch.sin(torch.pi * x[:, 4:5])
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON2':  # multiscale
        # Laplace U = f
        # u(x,y,z)=sin(pi*x)sin(3pi*y)sin(5pi*z)sin(3pi*s)sin(pi*t)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1])*torch.sin(3*np.pi*x[:, 1:2])*torch.sin(5*np.pi*x[:, 2:3])*\
                          torch.sin(3*torch.pi * x[:, 3:4]) * torch.sin(torch.pi * x[:, 4:5])
        fside = lambda x: -45*torch.pi*torch.pi*torch.sin(np.pi*x[:, 0:1])*torch.sin(3*np.pi*x[:, 1:2])*\
                          torch.sin(5*np.pi*x[:, 2:3])*torch.sin(3*torch.pi * x[:, 3:4]) * torch.sin(torch.pi * x[:, 4:5])
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON3':
        # Laplace U = f
        # u(x,y,z,s,t)=sin(pi*x)sin(pi*y)sin(pi*z)sin(pi*s)sin(pi*t)+0.1*sin(5pi*x)sin(5pi*y)sin(5pi*z)sin(5pi*s)sin(5pi*t)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1])*torch.sin(np.pi*x[:, 1:2])*torch.sin(np.pi*x[:, 2:3])*\
                          torch.sin(np.pi*x[:, 3:4])*torch.sin(np.pi*x[:, 4:5])+0.1*torch.sin(5*torch.pi*x[:, 0:1])*\
                          torch.sin(5*torch.pi*x[:, 1:2])*torch.sin(5*torch.pi*x[:, 2:3])*\
                          torch.sin(5*torch.pi*x[:, 3:4])*torch.sin(5*torch.pi*x[:, 4:5])

        fside = lambda x: -5.0*torch.pi*torch.pi*torch.sin(np.pi*x[:, 0:1])*torch.sin(np.pi*x[:, 1:2])*\
                          torch.sin(np.pi*x[:, 2:3])*torch.sin(np.pi*x[:, 3:4])*torch.sin(np.pi*x[:, 4:5])-\
                          0.1*125.0*torch.pi*torch.pi*torch.sin(5*torch.pi*x[:, 0:1])*\
                          torch.sin(5*torch.pi*x[:, 1:2])*torch.sin(5*torch.pi*x[:, 2:3])*\
                          torch.sin(5*torch.pi*x[:, 3:4])*torch.sin(5*torch.pi*x[:, 4:5])
        return utrue, fside