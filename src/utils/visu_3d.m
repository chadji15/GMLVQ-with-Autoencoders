function visu_3d(this)
% Visualizes the 3D-projection of the system

[ev,ew]=eig(this.lambda);       % determine eigenvectors and -values of lambda

omat(1,:)= ev(:,end);   % leading eigenvector of lambda
omat(2,:)= ev(:,end-1); % second eigenvector of lambda 
omat(3,:)= ev(:,end-2); % third eigenvector of lambda 

scale1=sqrt(ew(end,end));
scale2=sqrt(ew(end-1,end-1));  % scale projections with sqrt(eigenvalue) 
scale3=sqrt(ew(end-2,end-2));  % scale projections with sqrt(eigenvalue) 
% scale1=1; scale2=2;  % plot projections on orthonormal eigenvectors

proj= zeros(this.trainingData.nFeatureVectors,3);     % projections of all feature vectors
for i=1:this.trainingData.nFeatureVectors
    proj(i,1) = scale1*dot(omat(1,:),this.trainingData.featureVectors(i,:));  % proj. on eigenvector 1
    proj(i,2) = scale2*dot(omat(2,:),this.trainingData.featureVectors(i,:));  % proj. on eigenvector 2 
    proj(i,3) = scale3*dot(omat(3,:),this.trainingData.featureVectors(i,:));  % proj. on eigenvector 3
end 

projw= zeros(this.nPrototypes,3);  % projections of prototypes
for i=1:this.nPrototypes
    projw(i,1) = scale1*dot(omat(1,:),this.prototypes(i,:));  % proj. on eigenvector 1
    projw(i,2) = scale2*dot(omat(2,:),this.prototypes(i,:));  % proj. on eigenvector 2
    projw(i,3) = scale3*dot(omat(3,:),this.prototypes(i,:));  % proj. on eigenvector 3
end

% define symbols and colors for up to 7 classes  (prototypes)
symbstrings = GMLVQ.Util.consistentColorList(); %['b';'r';'g';'c';'m';'y';'k']; FIX Consistent Colors  
lgstr = [];   % initialize legend as empty text
for ip=1:this.nPrototypes
    scatter3(projw(ip,1), projw(ip,2),projw(ip,3),...            % plot prototype
        'MarkerFaceColor',symbstrings{this.gmlvq.plbl(ip),1});   % color acc. to 
    hold on;                                          % prototype label
    lgstr = [lgstr;num2str(this.gmlvq.plbl(ip))]; 
end

% define symbols and colors for up to 7 classes  (data points)
%symbstrings = ['bo';'ro';'go';'co';'mo';'yo';'ko']; 
for i=1:this.nClasses
    idx = this.trainingData.labels == i;
    scatter3(proj(idx,1), proj(idx,2), proj(idx,3),...
        'MarkerFaceColor', symbstrings{i});
end

% for i=1:this.trainingData.nFeatureVectors
%     scatter3(proj(i,1),proj(i,2), proj(i,3),... % symbstrings(lbl(i),:),...
%         'MarkerFaceColor',symbstrings{this.trainingData.labels(i)}); % plot data points 
%     % optional: mark samples by their number 
%     %  text(proj(i,1)+0.1,proj(i,2),num2str(i));
% end

% plot prototypes again to make them visible on top of data, larger now
% symbstrings = ['b';'r';'g';'c';'m';'y';'k']; % up to 7 classes
symbstrings = {'g', 'y'};
for ip=1:this.nPrototypes
    scatter3(projw(ip,1), projw(ip,2),projw(ip,3), 500,...
        'MarkerFaceColor',symbstrings{this.gmlvq.plbl(ip)});
    %  plot(projw(ip,1), projw(ip,2),'wp',...
    %      'MarkerSize',8); % 'MarkerFaceColor','w');     
end

if scale1==1 && scale2==1
    xlabel('projection on first eigenvector of \Lambda');
    ylabel('projection on second eigenvector of \Lambda');
else
    %   xlabel('projection on first eigenvector of \Lambda');
    %  ylabel('projection on second eigenvector of \Lambda');
    xlabel('proj. on first eigenvector of \Lambda, scaled by sqrt of eigenvalue');
    ylabel('proj. on second eigenvector of \Lambda, scaled by sqrt of eigenvalue');   
end

if this.gmlvq.params.showlegend
    legend(lgstr, 'Location','NorthEastOutside'); % legend of prototype symbols
end

axis tight; axis square;
title ('visualization of all data',...
    'FontName','LucidaSans','FontWeight','bold');
hold off; 
end