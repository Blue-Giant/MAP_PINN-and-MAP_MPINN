clc;
clear all
close all

q = 6;
nl = 2;
T = [];
J = [];

% geom: (only square geometry available now)
% generating 2d square mesh for the region [-1, 1] x [-1 1]
geom.q = q;
geom.nl = nl;
geom.L = 2; % side length 
geom.dim = 2; % dimension of the problem
geom.m = 2^geom.dim; % 
geom.N1 = 2^q; % dofs in one dimension
geom.N = (geom.m)^geom.q; % dofs in the domain
geom.h = geom.L/(geom.N1+1); % grid size
geom.xstart = 0;
geom.xend = 1;
geom.ystart = 0;
geom.yend = 1;

geom = assemble_fmesh(geom);

XYpoints = geom.pin;
Xs= XYpoints(1,:);
Ys = XYpoints(2,:);

eps = 0.2;
Uexact = 0.25*(Xs.*Xs+Ys.*Ys).^2+...
    eps/(16*pi)*(Xs.*Xs+Ys.*Ys).*sin(2*pi*(Xs.*Xs+Ys.*Ys)/eps)+ ...
    eps*eps/(32*pi*pi)*cos(2*pi*(Xs.*Xs+Ys.*Ys)/eps);

figure('name','Utrue')
mesh_Umean = plot_fun2in(geom,Uexact);
hold on




