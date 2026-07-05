clc;
clear all
close all
solu = @(x, y, z) sin(pi*x).*sin(pi*y).*sin(pi*z);

%对于类似的txt文件，不含有字符，只有数字
data=load('bun_zipper.txt');
% data=load('bun_zipper_res2.txt');
% data=load('bun_zipper_res3.txt');
% data=load('bun_zipper_res4.txt');
x=data(:,1)+0.25;
y=data(:,3)-0.25;
z=data(:,2)-1.0;
XYZ = [x,y,z];
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
save('testXYZ11.mat','XYZ')
