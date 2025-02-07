clear all;
close all;

load resultalp.mat

% [xSG, infoSG] = SubgradDescent(trainFeatures_aug, trainLabels, obj, grad, 1e6, 0.01);
% 
% 
rel_err = @(val) (val - cvx_optval) ./ cvx_optval;

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
    hsc1{i} = loglog( rel_err(infoFedMLS{i}.obj) ,'LineStyle','-', 'Color', colors(i,:));
    loglog( infoFedMLS{i}.numLS, rel_err(infoFedMLS{i}.obj) , 'LineStyle',':', 'Color', colors(i,:));
    legendText1{i} = ['$\lambda_0 = ', num2str(infoFedMLS{i}.lambda0),'$'];
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
    hsc2{i} = loglog( rel_err(infoFedAvg{i}.obj) ,'LineStyle','-', 'Color', colors(i,:));
    loglog( infoFedAvg{i}.numLS, rel_err(infoFedAvg{i}.obj) , 'LineStyle',':', 'Color', colors(i,:));
    legendText2{i} = ['$\eta_0 = ', num2str(infoFedAvg{i}.lr0),'$'];
    legendHandles2(i) = hsc2{i};
end
hl{2} = legend(legendHandles2, legendText2);

subplot(1,2,1)
for i = 1:2
    subplot(1,2,i)

    ylim([1e-4, 1e3])
    xlim([1, 1e8])
    
    ax = gca;
    set(findall(ax, 'Type', 'Line'),'LineWidth', 2.5);
    
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


