clc;
clear all
close all
solu = @(x, y, z) sin(pi*x).*sin(pi*y).*sin(pi*z);

XYZ_data=load('testXYZ.mat');
xyz = XYZ_data.XYZ;
x=xyz(:,1);
y=xyz(:,2);
z=xyz(:,3);
u = solu(x,y,z);
figure('name','bunny')
scatter3(x,y,z, 10, u, '.');
hold on
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$z$', 'Fontsize', 18, 'Interpreter', 'latex')
hold on
colorbar;
caxis([-1 1])
