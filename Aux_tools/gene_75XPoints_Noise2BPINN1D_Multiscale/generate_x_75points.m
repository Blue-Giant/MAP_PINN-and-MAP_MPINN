clc; 
clear all
close all

point_num2solu = 73;
bd_left = 0.0;
bd_right = 1.0;
length2x = bd_right - bd_left;
x_points= rand(point_num2solu, 1)*length2x + bd_left;
x2solu = [bd_left; x_points; bd_right];

point_num2force = 75;
x2fside= rand(point_num2force, 1)*length2x + bd_left;

point_num2coef = 75;
x2coef= rand(point_num2coef, 1)*length2x + bd_left;

% noise_level = 0.01;
% noise_level = 0.05;
% noise_level = 0.1;
% noise_level = 0.2;
noise_level = 0.5;
noise2solu = randn(point_num2solu+2,1)*noise_level;

noise2fside = randn(point_num2force,1)*noise_level;

noise2coef = randn(point_num2coef,1)*noise_level;

if noise_level == 0.01
    SavePath = 'matdata75Points2Bayes1D_Multiscale/0p01';
elseif noise_level == 0.05
    SavePath = 'matdata75Points2Bayes1D_Multiscale/0p05';
elseif noise_level == 0.1
    SavePath = 'matdata75Points2Bayes1D_Multiscale/0p1';
elseif noise_level == 0.2
    SavePath = 'matdata75Points2Bayes1D_Multiscale/0p2';
else
    SavePath = 'matdata75Points2Bayes1D_Multiscale/0p5';
end

if ~exist(SavePath,'dir')
    mkdir(SavePath)
end

if noise_level == 0.01
    save('matdata75Points2Bayes1D_Multiscale/0p01/x_points2solu.mat','x2solu');
    save('matdata75Points2Bayes1D_Multiscale/0p01/solu_noise.mat','noise2solu');

    save('matdata75Points2Bayes1D_Multiscale/0p01/x_points2fside.mat','x2fside');
    save('matdata75Points2Bayes1D_Multiscale/0p01/fside_noise.mat','noise2fside');

    save('matdata75Points2Bayes1D_Multiscale/0p01/x_points2coef.mat','x2coef');
    save('matdata75Points2Bayes1D_Multiscale/0p01/coef_noise.mat','noise2coef');
elseif noise_level == 0.05
    save('matdata75Points2Bayes1D_Multiscale/0p05/x_points2solu.mat','x2solu');
    save('matdata75Points2Bayes1D_Multiscale/0p05/solu_noise.mat','noise2solu');

    save('matdata75Points2Bayes1D_Multiscale/0p05/x_points2fside.mat','x2fside');
    save('matdata75Points2Bayes1D_Multiscale/0p05/fside_noise.mat','noise2fside');

    save('matdata75Points2Bayes1D_Multiscale/0p05/x_points2coef.mat','x2coef');
    save('matdata75Points2Bayes1D_Multiscale/0p05/coef_noise.mat','noise2coef');
elseif noise_level == 0.1
    save('matdata75Points2Bayes1D_Multiscale/0p1/x_points2solu.mat','x2solu');
    save('matdata75Points2Bayes1D_Multiscale/0p1/solu_noise.mat','noise2solu');

    save('matdata75Points2Bayes1D_Multiscale/0p1/x_points2fside.mat','x2fside');
    save('matdata75Points2Bayes1D_Multiscale/0p1/fside_noise.mat','noise2fside');

    save('matdata75Points2Bayes1D_Multiscale/0p1/x_points2coef.mat','x2coef');
    save('matdata75Points2Bayes1D_Multiscale/0p1/coef_noise.mat','noise2coef');
elseif noise_level == 0.2
    save('matdata75Points2Bayes1D_Multiscale/0p2/x_points2solu.mat','x2solu');
    save('matdata75Points2Bayes1D_Multiscale/0p2/solu_noise.mat','noise2solu');

    save('matdata75Points2Bayes1D_Multiscale/0p2/x_points2fside.mat','x2fside');
    save('matdata75Points2Bayes1D_Multiscale/0p2/fside_noise.mat','noise2fside');

    save('matdata75Points2Bayes1D_Multiscale/0p2/x_points2coef.mat','x2coef');
    save('matdata75Points2Bayes1D_Multiscale/0p2/coef_noise.mat','noise2coef');
else
    save('matdata75Points2Bayes1D_Multiscale/0p5/x_points2solu.mat','x2solu');
    save('matdata75Points2Bayes1D_Multiscale/0p5/solu_noise.mat','noise2solu');

    save('matdata75Points2Bayes1D_Multiscale/0p5/x_points2fside.mat','x2fside');
    save('matdata75Points2Bayes1D_Multiscale/0p5/fside_noise.mat','noise2fside');

    save('matdata75Points2Bayes1D_Multiscale/0p5/x_points2coef.mat','x2coef');
    save('matdata75Points2Bayes1D_Multiscale/0p5/coef_noise.mat','noise2coef');
end
