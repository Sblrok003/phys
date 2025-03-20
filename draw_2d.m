%% Draw 2D

% PARAMETERS >>> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colorlimits=[1 4];

fontSize=14;

load(matFileFullName)

peak=recVarPack.peak;delta=recVarPack.delta;
build3D=recVarPack.make3DArray;



ccp4I1rotated = permute(ccp4I1, [2 1 3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DRAW PLANES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
func_for_I3D = @(x)(log(abs(x)+1));
% plot parameters >>>
currentFigureName=['[' num2str(peak) ']'];
plotParametersStr=[];
plotParametersStr.projections=[];
%plotParametersStr.writeCenterInTitle=true;
plotParametersStr.caxis=colorlimits;
plotParametersStr.useGrid=true;

if abs(build3D.nh)==1 dt=1;elseif abs(build3D.nk)==1 dt=2;elseif abs(build3D.nl)==1 dt=3;
else dt = plotParametersStr.projections; end
plotParametersStr.projections=dt;
% plot parameters <<<

fZdraw2D( ccp4I1rotated, recVarPack,...
    func_for_I3D, peak, fontSize, recVarPack.valueFromMultiDataset, plotParametersStr ,...
    1, 1);
