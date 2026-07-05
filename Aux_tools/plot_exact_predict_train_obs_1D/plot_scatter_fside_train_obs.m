clc;
clear all
close all
ftsz = 18;
mark_label_size = 15;

train_points = load('force2train.mat');
X_Obs_train = double(train_points.xf_train);
fside_Obs_train = double(train_points.Ftrain_sin);

test_results = load('force2test.mat');
X = double(test_results.x);
fside_UTrue = double(test_results.Fexact_sin);
fside_MAP_PINN = double(test_results.Fmean_sin);


figure('name','Exact_Predict_Solu_Train_Points')
fig2true = plot(X, fside_UTrue, 'b-', 'linewidth', 2);
hold on
figure2MAP_PINN = plot(X, fside_MAP_PINN, 'r-', 'linewidth', 2);
hold on
fig2train_obs = plot(X_Obs_train, fside_Obs_train, 'kX', 'linewidth', 1.5);
hold on
ylim([-1,1.5])
xlabel('$x$', 'Fontsize', ftsz, 'Interpreter', 'latex')
% ylabel('$u_\varepsilon(x)$', 'Fontsize', ftsz, 'Interpreter', 'latex')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

lgd0=legend(fig2true,'Exact','orientation','horizontal','location','North');
set(lgd0,'FontSize',mark_label_size);
lgd0.Position = [0.02  0.15  0.45  0.2];
legend boxoff;

ah1=axes('position',get(gca,'position'),'visible','off');
lgd1=legend(ah1,figure2MAP_PINN,'MAP-PINN','orientation','horizontal','location','North');
set(lgd1,'FontSize',mark_label_size);
lgd1.Position = [0.25  0.15  0.45  0.2];
legend boxoff;

ah2=axes('position',get(gca,'position'),'visible','off');
lgd2=legend(ah2,fig2train_obs,'Training data','orientation','horizontal','location','North');
set(lgd2,'FontSize',mark_label_size);
lgd2.Position = [0.52  0.15  0.45  0.2];
legend boxoff;
