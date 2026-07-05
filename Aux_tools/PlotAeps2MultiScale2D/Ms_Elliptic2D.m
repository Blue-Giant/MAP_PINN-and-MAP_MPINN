%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%       div(a，grad U) = f(x,y) (x,y)in[0,1]X[0,1]
%%%             u = 0 on boundary.
%%%          a = 2+sin(pi*x/eps)*sin(pi*y/eps)
%%%          u_exact = sin(pi*x)*sin(pi*y)
%%%
%%%   dv(a，grad u) = a_x，u_x + a_y，u_y + a，delta u 
%%% u_x = 0.5*[u(i+1,j) - u(i-1,j)]; u_y = 0.5*[u(i,j+1) - u(i,j-1)]
clc;
clear all
close all
xstart = -1.0;
xend = 1.0;

ystart = -1.0;
yend = 1.0;

Nx = 128;
Ny = 128;


hx = (xend - xstart)/(Nx+1);
hy = (yend - ystart)/(Ny+1);
xline = xstart:hx: xend;
yline = ystart:hy:yend;
[Meshx, Meshy] = meshgrid(xline, yline);

eps = 0.2;
Aeps = @(XX,YY) 1.0./(4.0+cos(2.0*pi*(XX.*XX+YY.*YY)/eps));

MeshA = Aeps(Meshx, Meshy);

figure('name', 'Aeps')
surf(Meshx, Meshy,MeshA);
hold on
xlim([-1,1])
ylim([-1,1])
xlabel('$x_1$', 'Fontsize', 20, 'Interpreter', 'latex')
ylabel('$x_2$', 'Fontsize', 20, 'Interpreter', 'latex')
zlabel('$A^{\varepsilon}$', 'Fontsize', 20, 'Interpreter', 'latex')
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 16);
set(gcf, 'Renderer', 'zbuffer');
hold on