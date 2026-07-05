clc;
clear all
close all
ftsz = 18;
mark_label_size = 15;

train_points = load('solu2train.mat');
X_Obs_train = double(train_points.xu_train);
Solu_Obs_train = double(train_points.Utrain_sin);

test_results = load('solu2test.mat');
X = double(test_results.x);
Solu_UTrue = double(test_results.Uexact_sin);
Solu_MAP_PINN = double(test_results.Umean_sin);


figure('name','Exact_Predict_Solu_Train_Points')
fig2true = plot(X, Solu_UTrue, 'b-', 'linewidth', 2);
hold on
figure2MAP_PINN = plot(X, Solu_MAP_PINN, 'r-', 'linewidth', 2);
hold on
fig2train_obs = plot(X_Obs_train, Solu_Obs_train, 'kX', 'linewidth', 1.5);
hold on
xlabel('$x$', 'Fontsize', ftsz, 'Interpreter', 'latex')
% ylabel('$u_\varepsilon(x)$', 'Fontsize', ftsz, 'Interpreter', 'latex')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', ftsz);
set(gcf, 'Renderer', 'zbuffer');
hold on

lgd0=legend(fig2true,'Exact','orientation','horizontal','location','North');
set(lgd0,'FontSize',mark_label_size);
lgd0.Position = [0.3  0.40  0.45  0.2];
legend boxoff;

ah1=axes('position',get(gca,'position'),'visible','off');
lgd1=legend(ah1,figure2MAP_PINN,'MAP-PINN','orientation','horizontal','location','North');
set(lgd1,'FontSize',mark_label_size);
lgd1.Position = [0.3  0.3  0.45  0.2];
legend boxoff;

ah2=axes('position',get(gca,'position'),'visible','off');
lgd2=legend(ah2,fig2train_obs,'Training data','orientation','horizontal','location','North');
set(lgd2,'FontSize',mark_label_size);
lgd2.Position = [0.3  0.2  0.45  0.2];
legend boxoff;
