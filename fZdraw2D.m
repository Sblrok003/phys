function fZdraw2D( I3D, recVarPack, func_for_I3D, pointDraw, fontSize, valueFromMultiDataset, plotParametersStr, rowNumber, totalRows)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


hl=recVarPack.hl;
kl=recVarPack.kl;
ll=recVarPack.ll;
hr=recVarPack.hr;
kr=recVarPack.kr;
lr=recVarPack.lr;
nh = recVarPack.make3DArray.nh;
nk = recVarPack.make3DArray.nk;
nl = recVarPack.make3DArray.nl;

if nh==-1 nh=2;elseif nk==-1 nk=2;elseif nl==-1 nl=2;end
hArray = linspace(hl, hr, nh+1) + (hr-hl)/nh/2;
kArray = linspace(kl, kr, nk+1) + (kr-kl)/nk/2;
lArray = linspace(ll, lr, nl+1) + (lr-ll)/nl/2;


if isfloat(I3D)
    I3D=double(I3D);
end



multiDatasetTypeShort=valueFromMultiDataset.multiDatasetTypeShort;
units=valueFromMultiDataset.units;
currentValue=valueFromMultiDataset.currentValue;


projections=plotParametersStr.projections;
if isempty(projections)
    projections = [1 2 3];
end

SH = pointDraw(1); SK = pointDraw(2); SL = pointDraw(3);
FS = fontSize;

% make slices >>>
[~,nH]=min(abs(hArray-SH));
[~,nK]=min(abs(kArray-SK));
[~,nL]=min(abs(lArray-SL));
% zaplatka
if size(I3D,1)==1
    nK=1;
end
if size(I3D,2)==1
    nH=1;
end
if size(I3D,3)==1
    nL=1;
end
% zaplatka
okl=squeeze(I3D(:,nH,:))';
hol=squeeze(I3D(nK,:,:))';
hko=squeeze(I3D(:,:,nL));
% make slices <<<

slices2D=cell(3,1);
xAxes=cell(3,1);
yAxes=cell(3,1);
xLimits=cell(3,1);
yLimits=cell(3,1);
xLabels=cell(3,1);
yLabels=cell(3,1);
titles=cell(3,1);

slices2D{1}=okl;
xAxes{1}=kArray;
yAxes{1}=lArray;

xLabels{1}='K (r.l.u.)';
yLabels{1}='L (r.l.u.)';

titles{1}=['{\it' multiDatasetTypeShort '} = ' num2str(currentValue) ' ' units];
xLimits{1}=[kArray(1) kArray(end)];
yLimits{1}=[lArray(1) lArray(end)];


slices2D{2}=hol;
xAxes{2}=hArray;
yAxes{2}=lArray;

xLabels{2}='H (r.l.u.)';
yLabels{2}='L (r.l.u.)';

titles{2}=['{\it' multiDatasetTypeShort '} = ' num2str(currentValue) ' ' units];



slices2D{3}=hko;
xAxes{3}=hArray;
yAxes{3}=kArray;

xLabels{3}='H (r.l.u.)';
yLabels{3}='K (r.l.u.)';

titles{3}=['{\it' multiDatasetTypeShort '} = ' num2str(currentValue) ' ' units];



figure
iPlot=0;
for iSlice=projections
    iPlot=iPlot+1;
    subplot(totalRows,length(projections),iPlot+length(projections)*(rowNumber-1))
    hklSliceCurrent_orig=single(slices2D{iSlice});
    hklSliceCurrent = single(func_for_I3D(hklSliceCurrent_orig));
    
    xAxeCurrent=xAxes{iSlice};
    yAxeCurrent=yAxes{iSlice};
    titleCurrent=titles{iSlice};
    xLabelCurrent=xLabels{iSlice};
    yLabelCurrent=yLabels{iSlice};
    
    
    h=imagesc(xAxeCurrent([1 end]), yAxeCurrent([1 end]), hklSliceCurrent);
    
    
    grid on
    
    caxis(plotParametersStr.caxis);
    
    set(h, 'AlphaData', ~isnan(hklSliceCurrent))
    
    titleStr=titleCurrent;
    
    xlabel(xLabelCurrent,'FontSize',FS);
    ylabel(yLabelCurrent,'FontSize',FS);
    
    
    title_hkl_full = 'HKL';
    title_hkl = [title_hkl_full(iSlice) '=' num2str(pointDraw(iSlice))];
    title([titleStr ', ' title_hkl] ,'FontSize',FS)
    
    
    set(gca, 'DataAspectRatio',   [1 1 1])
    
    set(gca,'FontSize',FS);
    set(gca,'YDir','normal')
end

end




