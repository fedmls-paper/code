%% To save the figures

% First download altmany-export_fig-d966721 and add to path
addpath('/Users/alp/Documents/MATLAB/export_fig/')

% Determine and create the path to save the figures
figPath = 'figs/';
mkdir(figPath);

figHandles = findall(groot, 'Type', 'figure');
for rr = 1:length(figHandles)
    sName = [figPath,figHandles(rr).Name];
%     savefig(figHandles(rr), sName)
    set(0, 'CurrentFigure', figHandles(rr))
    figHandles(rr).Color = [1,1,1];
    if exist('export_fig','file')
        export_fig(sName,'-pdf','-dCompatibilityLevel=1.4')
    end
end
