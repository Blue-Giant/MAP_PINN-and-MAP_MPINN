clc;
clear all
close all
ftsz = 15;

test_solus = load('solu2test');
Solu_UTrue_temp = double(test_solus.Uexact_sin);
Solu_UDNN_temp = double(test_solus.Umean_sin);

Solu_UTrue = reshape(Solu_UTrue_temp, 100,100);
Solu_UDNN =  reshape(Solu_UDNN_temp, 100, 100);

figure('name','Exact_Solu')
ct = Solu_UTrue;
scatter_solu2u = imagesc(Solu_UTrue);
shading interp
hold on
colorbar;
caxis([0 1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

figure('name','DNN_Solu')
cn = Solu_UDNN;
mesh_solu_unn =imagesc(Solu_UDNN);
shading interp
hold on
colorbar;
caxis([0 1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

err2solu = abs(Solu_UTrue-Solu_UDNN);
figure('name','Err2solu')
cp = err2solu;
mesh_solu_err2u = imagesc(err2solu);
shading interp
title('Absolute Error')
hold on
colorbar;
caxis([0 5e-1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on


