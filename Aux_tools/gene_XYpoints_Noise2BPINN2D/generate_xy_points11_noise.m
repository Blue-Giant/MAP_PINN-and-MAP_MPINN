clc; 
clear all
close all

point_num2force = 800;   % 影响右端项的点数
point_num2solu = 700;    % 内部点数目
bd_point_num = 25;       % 边界点数目

bd_left = -1.0;          % 左边界
bd_right = 1.0;          % 有边界
bd_bottom = -1.0;        % 下边界
bd_top = 1.0;            % 上边界

length2x = bd_right - bd_left;   % x方向的长度
length2y = bd_top - bd_bottom;   % y方向的长度

xpoints2left = bd_left*ones(bd_point_num, 1);
xpoints2right = bd_right*ones(bd_point_num, 1);
y_random2left_right_bd = rand(bd_point_num, 1)*length2y + bd_bottom;
xy_point2left_bd = [xpoints2left, y_random2left_right_bd];
xy_point2right_bd = [xpoints2right, y_random2left_right_bd];

x_random2bottom_top = rand(bd_point_num, 1)*length2x + bd_left;
ypoints2bottom = bd_bottom*ones(bd_point_num, 1);
ypoints2top = bd_top*ones(bd_point_num, 1);
xy_point2bottom_bd = [x_random2bottom_top, ypoints2bottom];
xy_point2top_bd = [x_random2bottom_top, ypoints2top];

xy_points2bd = [xy_point2left_bd;xy_point2right_bd;xy_point2bottom_bd;xy_point2top_bd];

x2solu= rand(point_num2solu, 1)*length2x + bd_left;
y2solu= rand(point_num2solu, 1)*length2y + bd_bottom;

xy2solu_in = [x2solu, y2solu];
xy2solu = [xy2solu_in;xy_points2bd];
figure('name','xy2solu')
plot(xy_points2bd(:, 1),xy_points2bd(:, 2), 'r*');
hold on
plot(x2solu,y2solu, 'b*');
hold on


x2fside= rand(point_num2force, 1)*length2x + bd_left;
y2fside= rand(point_num2force, 1)*length2y + bd_bottom;
xy2fside = [x2fside,y2fside];

plot(x2fside,y2fside, 'm*');
hold on

% noise_level = 0.01;
% noise_level = 0.02;
% noise_level = 0.05;
% noise_level = 0.1;
noise_level = 0.2;
% noise_level = 0.5;
noise2solu = randn(point_num2solu+bd_point_num*4,1)*noise_level;

noise2fside = randn(point_num2force,1)*noise_level;

if noise_level == 0.01
    SavePath = 'matdata2BPINN_BMPINN_2D_11/0p01';
elseif noise_level == 0.02
    SavePath = 'matdata2BPINN_BMPINN_2D_11/0p02';
elseif noise_level == 0.05
    SavePath = 'matdata2BPINN_BMPINN_2D_11/0p05';
elseif noise_level == 0.1
    SavePath = 'matdata2BPINN_BMPINN_2D_11/0p1';
elseif noise_level == 0.2
    SavePath = 'matdata2BPINN_BMPINN_2D_11/0p2';
else
    SavePath = 'matdata2Bayes2D_11/0p5';
end

if ~exist(SavePath,'dir')
    mkdir(SavePath)
end

if noise_level == 0.01
    save('matdata2Bayes2D_11/0p01/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p01/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p01/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p01/fside_noise.mat','noise2fside');
elseif noise_level == 0.02
    save('matdata2Bayes2D_11/0p02/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p02/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p02/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p02/fside_noise.mat','noise2fside');
elseif noise_level == 0.05
    save('matdata2Bayes2D_11/0p05/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p05/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p05/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p05/fside_noise.mat','noise2fside');
elseif noise_level == 0.1
    save('matdata2Bayes2D_11/0p1/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p1/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p1/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p1/fside_noise.mat','noise2fside');
elseif noise_level == 0.2
    save('matdata2Bayes2D_11/0p2/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p2/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p2/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p2/fside_noise.mat','noise2fside');
else
    save('matdata2Bayes2D_11/0p5/xy_points2solu.mat','xy2solu');
    save('matdata2Bayes2D_11/0p5/solu_noise.mat','noise2solu');

    save('matdata2Bayes2D_11/0p5/xy_points2fside.mat','xy2fside');
    save('matdata2Bayes2D_11/0p5/fside_noise.mat','noise2fside');
end
