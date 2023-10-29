function demo(yvector1)
%CREATEFIGURE(yvector1)
%  YVECTOR1:  bar yvector

%  由 MATLAB 于 16-Oct-2023 02:09:41 自动生成

% 创建 figure
figure1 = figure;

% 创建 axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% 创建 bar
bar(yvector1,'Horizontal','on',...
    'FaceColor',[0.850980392156863 0.325490196078431 0.0980392156862745],...
    'EdgeColor','none');

% 创建 ylabel
ylabel({'Channel Name'});

% 创建 xlabel
xlabel({'R2 Score'});

% 创建 title
title({'(b)'},'FontWeight','bold');

% 取消以下行的注释以保留坐标区的 X 范围
xlim(axes1,[0.7 0.9]);
box(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'FontName','Times New Roman','FontSize',15,'XGrid','on',...
    'XMinorTick','on','YTick',...
    [1 2 3 4 5 6 7],'YTickLabel',...
    {'FC5','CP5','P4','P8','AF3','P3','TP7'});
