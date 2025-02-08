clear all;
close all;

TOTSEEDNUM = 20;

Lambda_Sweep = [1e-1, 1, 10];
LR_Sweep = [1e-4, 1e-3, 1e-2];

avg_FedMLS = {};
avg_FedAvg = {};
for i = 1:length(Lambda_Sweep)
    avg_FedMLS{i}.relerr = 0;
    avg_FedMLS{i}.numLS = 0;
end
for i = 1:length(LR_Sweep)
    avg_FedAvg{i}.relerr = 0;
    avg_FedAvg{i}.numLS = 0;
end

for SEED = 1:TOTSEEDNUM

    load(['results/main_svm_stochastic_',num2str(SEED),'.mat']);
    rel_err = @(val) (val - out.cvx_optval) ./ out.cvx_optval;
    
    for i = 1:length(Lambda_Sweep)
        avg_FedMLS{i}.relerr = avg_FedMLS{i}.relerr + (1/TOTSEEDNUM) * rel_err(out.infoFedMLS{i}.obj);
        avg_FedMLS{i}.numLS = avg_FedMLS{i}.numLS + (1/TOTSEEDNUM) * out.infoFedMLS{i}.numLS;
    end

    for i = 1:length(LR_Sweep)
        avg_FedAvg{i}.relerr = avg_FedAvg{i}.relerr + (1/TOTSEEDNUM) * rel_err(out.infoFedAvg{i}.obj);
        avg_FedAvg{i}.numLS = avg_FedAvg{i}.numLS + (1/TOTSEEDNUM) * out.infoFedAvg{i}.numLS;
    end
end

%%

% [xSG, infoSG] = SubgradDescent(trainFeatures_aug, trainLabels, obj, grad, 1e6, 0.01);
% 
% 

%%

hfig = figure('Position',[100,100,1000,320]);
set(hfig,'name','svm-results','numbertitle','off');


colors = get(gca, 'ColorOrder');

subplot(122);
hsc1 = {};
legendText1 = {};
legendHandles1 = [];
hold on
for i = 1:length(Lambda_Sweep)
    hsc1{i} = loglog( avg_FedMLS{i}.relerr ,'LineStyle','-', 'Color', colors(i,:),'LineWidth', 2.5);
    loglog( avg_FedMLS{i}.numLS, avg_FedMLS{i}.relerr , 'LineStyle',':', 'Color', colors(i,:),'LineWidth', 1.5);
    legendText1{i} = ['$\lambda_0 = ', num2str(out.infoFedMLS{i}.lambda0),'$'];
    legendHandles1(i) = hsc1{i};
end
% loglog( rel_err(infoSG.obj) ,'k:');
hl{1} = legend(legendHandles1, legendText1);


subplot(121);
hsc2 = {};
legendText2 = {};
legendHandles2 = [];
hold on
for i = 1:length(LR_Sweep)
    hsc2{i} = loglog( avg_FedAvg{i}.relerr ,'LineStyle','-', 'Color', colors(i,:),'LineWidth', 2.5);
    loglog( avg_FedAvg{i}.numLS, avg_FedAvg{i}.relerr , 'LineStyle',':', 'Color', colors(i,:),'LineWidth', 1.5);
    legendText2{i} = ['$\eta_0 = ', num2str(out.infoFedAvg{i}.lr0),'$'];
    legendHandles2(i) = hsc2{i};
end
hl{2} = legend(legendHandles2, legendText2);

subplot(1,2,1)
for i = 1:2
    subplot(1,2,i)

    ylim([1e-3, 1e3])
    xlim([1, 1e8])
    
    ax = gca;
%     set(findall(ax, 'Type', 'Line'),'LineWidth', 2.5);
    
    ax.XScale = 'log';
    ax.YScale = 'log';
    grid on,
    grid minor; grid minor;

    set(gca,'TickDir','out')
    set(gca,'LineWidth',1,'TickLength',[0.02 0.02]);

    ax.YTick = 10.^(-20:20);
    ax.XTick = 10.^(-20:20);

    ax.TickLabelInterpreter = 'latex';
    ax.FontSize = 14;
    ax.Box = 'on';

    ax.YRuler.MinorTick = 'off';
    ax.XRuler.MinorTick = 'off';

    xlabel({'communication round (line)', 'local iterations (dots)'}, 'Interpreter','latex', 'FontSize', 16);
    ylabel('$(F(x_k) - F^*) / F^*$', 'Interpreter','latex', 'FontSize', 16);

    hl{i}.Interpreter = 'latex';
    hl{i}.FontSize = 14;

end

subplot(122)
title('FedMLS', 'Interpreter', 'latex', 'FontSize', 18);
subplot(121)
title('FedAvg', 'Interpreter', 'latex', 'FontSize', 18);


