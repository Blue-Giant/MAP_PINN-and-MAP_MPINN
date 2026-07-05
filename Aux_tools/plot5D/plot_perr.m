clc;
clear all
close all

test_solus = load('solu2test');
Solu_UTrue_temp = double(test_solus.Uexact_sin);
Solu_UDNN_temp = double(test_solus.Umean_sin);

Solu_UTrue = reshape(Solu_UTrue_temp, 100,100);
Solu_UDNN =  reshape(Solu_UDNN_temp, 100, 100);

[X,Y]=meshgrid(1:1:100,1:1:100);

figure('name','TrueSolu')
surf(X,Y,Solu_UTrue)
set(gca,'FontSize',15.5) 
hold on
colorbar;
caxis([0 1.0])
hold on

pErr2scale = abs(Solu_UTrue-Solu_UDNN);
figure('name','pError2scale')

surf(X,Y,pErr2scale)
set(gca,'FontSize',15.5) 
hold on
colorbar;
caxis([0 0.2])
hold on