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
x = x+1;
y = y+1;
z = z+1;

x_min = min(x);
y_min = min(y);
z_min = min(z);

x_max = max(x);
y_max = max(y);
z_max = max(z);

x_scale = x_max - x_min;
y_scale = y_max - y_min;
z_scale = z_max - z_min;

scale = max(z_scale,max(x_scale,y_scale));

NewX = (x-x_min)./scale;
NewY = (y-y_min)./scale;
NewZ = (z-z_min)./scale;

XYZ = [NewX,NewY,NewZ];
u = solu(NewX,NewY,NewZ);
figure('name','bunny')
scatter3(NewX,NewY,NewZ, 10, u, '.');
hold on
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$z$', 'Fontsize', 18, 'Interpreter', 'latex')
hold on
colorbar;
caxis([0 1])
save('testXYZ01.mat','XYZ')
