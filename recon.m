clear('temperatureArray', 'mainFolder', 'temperature', 'folderNameTemplate')

folderForSavingMatFiles=['output\' datestr(now,'yyyy.mm.dd') '\recon_01\recon\'];

experimentIdentifierString = 'PZT2.4 N03b (ramp_cool_0kV) sum by temperature';
databasepath = '';

[experimentMetadata, isOkGetDescription] = fZgetExperimentDescription(experimentIdentifierString, databasepath);
experimentInfo=experimentMetadata.experimentInfo;experimentNames=experimentMetadata.experimentNames;
backgroundWindow=experimentMetadata.backgroundWindow;multiDataset=experimentMetadata.multiDataset;
detectorImageStoreGeometry=experimentMetadata.detectorImageStoreGeometry;
if ~isOkGetDescription return; end



useub = false;

normalization.normalizeOnFirstFrame=false;
normalization.useNormalizationOn = 2; % 0 - none, 1 - bg, 2 - flux
normalization.returnFrameNumbers=0;
normalization.rememberFlux=0;
normalization.rememberBg=0;

cutFrames=[0 inf];

additionalTools=[];
additionalTools.solid_angle = true;
additionalTools.excludenegativepixels = true;
additionalTools.detector_area = 1; % 0 - use both , 1 - first , 2 - second

maskfilename = 'c:\Temp\var_1\test\mask_1475x1679_2023.08.06.cbf';
detectormask = [];
detectormask.useDetectorMask = true;
detectormask.IbadPixels = logical(fZcbfRead(maskfilename));

useScan=1;

dataset_index = 1;
valueFromMultiDataset=multiDataset;
valueFromMultiDataset.currentValue=multiDataset.multiDatasetValues(dataset_index);
valueFromMultiDataset.currentIndex=dataset_index;

chosenVolume=[];
chosenVolume.peak=[1 0 -1];
chosenVolume.delta=[0.6 0.6 0.6];

make3DArray=[];
make3DArray.makeReconstructionImmediately=1;
make3DArray.makeAxes=false;
make3DArray.makeCounts=false;
make3DArray.nh=100;
make3DArray.nk=100;
make3DArray.nl=100;

pixelFragmentation.usePixelFragmentation=1;
pixelFragmentation.pixelFragmentationX=1;
pixelFragmentation.pixelFragmentationY=1;
pixelFragmentation.pixelFragmentationAngle=7;



clear('recVarPack', 'I1D', 'F1D', 'H1D', 'K1D', 'L1D', 'ccp4I1', 'ccp4I0', 'ccp4H', 'ccp4K', 'ccp4L', 'ccp4Info')
[ recVarPack, I1D, F1D, H1D, K1D, L1D, ccp4I1, ccp4F, ccp4I0, ccp4H, ccp4K, ccp4L, warnings, isOk] = ...
    fZmakeReconstruction(experimentInfo, experimentNames, detectorImageStoreGeometry, valueFromMultiDataset ,...
    normalization, backgroundWindow, cutFrames, chosenVolume, 'single', pixelFragmentation, make3DArray ,...
    detectormask, additionalTools, useScan);
if ~isOk break; end
f=find(I1D<0 | isnan(I1D));I1D(f)=[];H1D(f)=[];K1D(f)=[];L1D(f)=[]; if ~isempty(F1D) F1D(f)=[]; end; clear('f')
if isempty(I1D) && ~recVarPack.make3DArray.makeReconstructionImmediately
    return
end

matFolder=fullfile(experimentNames.mainFolder, folderForSavingMatFiles);
mkdir(matFolder)
matFileName = ['T=' num2str(recVarPack.currentValueFromMultiDataset) ', cen=[' num2str(recVarPack.peak) ']'];
matFileFullName=fullfile(matFolder, matFileName);
save(matFileFullName, 'recVarPack', 'I1D', 'F1D', 'H1D', 'K1D', 'L1D', 'ccp4I1', 'ccp4F', 'ccp4I0', 'ccp4H', 'ccp4K', 'ccp4L');
disp(['File ''' matFileName ''' was successfully saved!'])
% Make 3D Arrays <<<
clear('I1D', 'F1D', 'H1D', 'K1D', 'L1D', 'ccp4I1', 'ccp4F', 'ccp4I0', 'ccp4H', 'ccp4K', 'ccp4L')






