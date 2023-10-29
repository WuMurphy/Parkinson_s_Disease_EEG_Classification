function singleChannelEvaluationFigure_EyesClose(xvector1, channelName, yvector1)
%CREATEFIGURE(xvector1, yvector1)
%  XVECTOR1:  bar xvector
%  YVECTOR1:  bar yvector

%  由 MATLAB 于 10-May-2023 15:38:47 自动生成

% 创建 figure
figure1 = figure('PaperSize',[20.99999864 29.69999902]);

% 创建 axes
axes1 = axes('Parent',figure1,...
    'Position',[0.13 0.0576407506702413 0.77528233151184 0.910187667560322]);
hold(axes1,'on');

% 创建 bar
bar1 = bar(xvector1,yvector1,'DisplayName','values','Horizontal','on',...
    'BarLayout','stacked');
baseline1 = get(bar1,'BaseLine');
set(baseline1,'Visible','on');

% 创建 ylabel
ylabel('Channel Name');

% 创建 xlabel
xlabel('R2 Score');

% 创建 title
title('Single Channel Evaluation on EyesClose Dataset');

% 取消以下行的注释以保留坐标区的 X 范围
xlim(axes1,[0.35 0.9]);
box(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'FontName','Times New Roman','FontSize',15,'GridAlpha',0.3,...
    'MinorGridAlpha',0.3,'XGrid','on','XMinorGrid','on','YTick',...
    [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63],...
    'YTickLabel',...
    channelName);
%     {'TP9','P3','P8','FC3','P1','PO7','O2','P4','CP3','P6','FC5','Pz','O1','Oz','TP10','C4','F5','TP7','P5','PO3','PO4','PO8','FC4','Fp1','F3','C3','T7','P7','CP6','AF7','POz','P2','CP4','FT8','Fz','F7','FT9','CP5','CP1','CP2','FT10','FC2','F4','C5','F2','AF8','FC1','Cz','T8','FC6','Fp2','F1','FT7','FCz','C1','C6','F6','AF4','AF3','TP8','C2','AFz','F8'} ...
%     );
