function [ recVarPack, I1D, F1D, H1D, K1D, L1D, ccp4I1, ccp4F, ccp4I0, ccp4H, ccp4K, ccp4L, warningsStruct, isOkMain ] =...
fZmakeReconstruction(...
experimentInfo, experimentNames, detectorImageStoreGeometry,...
valueFromMultiDataset, normalization, ...
backgroundWindow, cutFrames, chosenVolume, outputType, ...
pixelFragmentation, make3DArray, detectorMask, additionalTools, useScan)
isOkMain=false; recVarPack=[]; I1D=[]; F1D=[]; H1D=[]; K1D=[]; L1D=[]; ccp4I1=[]; ccp4I0=[]; ccp4F=[]; ccp4H=[]; ccp4K=[]; ccp4L=[];
warningsStruct=iJBkBsNQ8LyCT();
[isOk, experimentInfo, experimentNames, valueFromMultiDataset, normalization, backgroundWindow, chosenVolume, pixelFragmentation, make3DArray, additionalTools] =...
Ezg29UNgW2(...
experimentInfo, experimentNames, valueFromMultiDataset, normalization, backgroundWindow, chosenVolume, pixelFragmentation, make3DArray, additionalTools);
if ~isOk
return
end
outputTypeSingle=false;
if ~ischar(outputType) || ~strcmp(outputType, 'single')
outputType='double';oneInoutputType=1;else
outputType='single';outputTypeSingle=true;oneInoutputType=single(1);
end
returnFrameNumbers=      normalization.returnFrameNumbers;
rememberFluxInEachFrame= normalization.rememberFlux;
rememberBgInEachFrame=   normalization.rememberBg;
bgWindowX1=      backgroundWindow.bgWindowUpperLeftCornerCoordinates(1);
bgWindowX2=      backgroundWindow.bgWindowBottomRightCornerCoordinates(1);
bgWindowY1=      backgroundWindow.bgWindowUpperLeftCornerCoordinates(2);
bgWindowY2=      backgroundWindow.bgWindowBottomRightCornerCoordinates(2);
bgWindowCoordinateSystem= backgroundWindow.bgWindowCoordinateSystem;
if ~isempty(detectorMask) && isfield(detectorMask ,'useDetectorMask') &&...
detectorMask.useDetectorMask
detectorMaskMatrix = detectorMask.IbadPixels;else
detectorMask=struct('useDetectorMask', false);detectorMaskMatrix = [];
end
useDetectorMask=detectorMask.useDetectorMask;
if isfield(additionalTools, 'excludenegativepixels')
excludenegativepixels = additionalTools.excludenegativepixels;else
excludenegativepixels = false;
end
if isfield(normalization, 'useNormalizationOn')
useNormalizationOn=normalization.useNormalizationOn;switch useNormalizationOn
case 0
useNormalizationString='none';case 1
useNormalizationString='background';case 2
useNormalizationString='flux';
end
else
useNormalizationOn=0;useNormalizationString='none';
end
bg2DarrayMeanZero=1;clear('headZero', 'Izero', 'bg2DarrayZero', 'IciZero')
usePixelFragmentation=false;
usePixelFragmentationDetector=false;pixelFragmentationAngle=1;
if pixelFragmentation.usePixelFragmentation==true
usePixelFragmentation=true;
pixelFragmentationAngle=pixelFragmentation.pixelFragmentationAngle;
if pixelFragmentation.pixelFragmentationX>1 || pixelFragmentation.pixelFragmentationY>1
usePixelFragmentationDetector=true;
end
end
makeReconstructionImmediately=false;
if make3DArray.makeReconstructionImmediately
makeReconstructionImmediately=true;
end
[ currentDatasetName, currentDatasetFolderFullName,...
warningsStruct ] =...
OK26MDiA05sPl74(...
experimentNames, valueFromMultiDataset, warningsStruct);
if ischar(additionalTools) && ~isempty(additionalTools)...
&& size(additionalTools,1) ==1 && strcmp(additionalTools, 'find in experiment folder')
additionalTools_read = fullfile(currentDatasetFolderFullName, 'additional_parameters.m');
try
run(additionalTools_read)
catch
warning('Error in additional tools')
end
end
if isstruct(additionalTools) && isfield(additionalTools, 'solid_angle') &&...
isscalar(additionalTools.solid_angle)
solid_angle_use = additionalTools.solid_angle;else
solid_angle_use = true;
end
[  crackerData, runFileData, useless, imagesData, exp_name_real, isOkExperimentConfigFiles] =...
mYIMtqDYewJeExY( currentDatasetFolderFullName, currentDatasetName, ...
experimentNames, additionalTools );
if ~isOkExperimentConfigFiles
return
end
imageFormat=imagesData(1).imagesFormat;
if strcmp(imageFormat, 'cbf')
runFileData=s_RCJe3(runFileData);excludenegativepixels = true;
end
useubarray = false;ub_array_txt = [];
if isstruct(additionalTools) && isfield(additionalTools, 'ub') &&...
ischar(additionalTools.ub) && ~isempty(additionalTools.ub)...
&& size(additionalTools.ub,1) ==1
if strncmpi(additionalTools.ub, 'read UB matrix from "ub_matrix.txt"', 21) &&...
strcmpi(additionalTools.ub(end), '"')
ub_matrix_filename = fullfile(currentDatasetFolderFullName, additionalTools.ub(22:end-1));
[ub_txt, isok_ub_txt] = ITmX6D_( ub_matrix_filename );
if isok_ub_txt
crackerData.ub  =     ub_txt ;crackerData.iub = inv(ub_txt);else
warning(['Error occured while reading UB matrix from file "' ub_matrix_filename '"'])
return
end
end
if strncmpi(additionalTools.ub, 'read UB matrices from "ub_matrix.txt"', 23) &&...
strcmpi(additionalTools.ub(end), '"')
ub_matrix_filename = fullfile(currentDatasetFolderFullName, additionalTools.ub(24:end-1));
[ub_array_txt, isok_ub_txt] = fZub_master_read_ub_array_from_txt( ub_matrix_filename );
if isok_ub_txt && runFileData.scansStruct.done== size(ub_array_txt, 3)
useubarray = true;ub_txt = squeeze(mean(ub_array_txt, 3));
crackerData.ub  = ub_txt;crackerData.iub = inv(ub_txt);else
warning(['Error occured while reading UB matrix from file "' ub_matrix_filename '"'])
return
end
end
end
[useScan, isScanIndexOk]=OKuPF1r8_h5e4B(useScan, runFileData, imagesData);
if ~isScanIndexOk
return
end
imagesData=imagesData(useScan);runFileData.scansStruct=runFileData.scansStruct(useScan);
runFileData.runFileHead.number_of_scans=1;
[ experimentGeometryInfo, isOkExperimentGeometryInfo ] = gOxS6ShVOiGI( crackerData, runFileData );
if ~isOkExperimentGeometryInfo
return
end
imagesFullNames=imagesData.imagesFullNames;fluxZero=backgroundWindow.defaultFlux;
[useless, fD, sD, headZero, fluxZero_fromFirstFrame, isOkFirstFrame]=...
MNzXp6am(imageFormat, imagesFullNames{1});
if ~isOkFirstFrame
return
end
if detectorImageStoreGeometry.auto
[detectorImageStoreGeometry, isOkDetectorType]=...
GYYlBYDrgEZbpN(imageFormat, headZero);
if ~isOkDetectorType
return
end
end
fdB=[];sdB=[];
if usePixelFragmentation==true
switch detectorImageStoreGeometry.imageFastestDimentionOrientation
case 'strings'
fdB=double(pixelFragmentation.pixelFragmentationX);
sdB=double(pixelFragmentation.pixelFragmentationY);otherwise
fdB=double(pixelFragmentation.pixelFragmentationY);
sdB=double(pixelFragmentation.pixelFragmentationX);
end
end
split_detector_on_2_parts = false;
if isfield(additionalTools ,'detector_area') && ...
~isempty(additionalTools.detector_area) &&...
isnumeric(additionalTools.detector_area) &&...
additionalTools.detector_area>0
split_detector_on_2_parts = true;
end
iub=crackerData.iub;
[ rArray     , surfacesInResiprocalSpace, solidAngle, detectorPerimeterCoordinates, detectorFragments ] =...
ij_8VJaiB24( experimentGeometryInfo, [], ...
detectorMask, detectorImageStoreGeometry, true, (~usePixelFragmentationDetector) || split_detector_on_2_parts);
if split_detector_on_2_parts
detector_area_to_block = Etm4PNOzs3s(experimentGeometryInfo, rArray);
if additionalTools.detector_area==2
detector_area_to_block = ~detector_area_to_block;
end
if useDetectorMask
detectorMaskMatrix(detector_area_to_block) = true;else
detectorMaskMatrix = detector_area_to_block;useDetectorMask = true;
end
end
if usePixelFragmentationDetector==true
clear('rArray')
end
normOnSolidAngle=1./solidAngle;clear('solidAngle')
numOfBins=10000;chosenVolumeShape=chosenVolume.shape;chosenVolumeUseSphere=false;
chosenVolumeUseCylinder=false;chosenVolumeUseParallelepiped=false;
chosenVolumeUseSimpleParallelepiped=false;switch chosenVolumeShape
case  'sphere'
chosenVolumeUseSphere=true;case 'cylinder'
chosenVolumeUseCylinder=true;case 'parallelepiped'
chosenVolumeUseParallelepiped=true;case 'simple parallelepiped'
chosenVolumeUseSimpleParallelepiped=true;
end
if strcmp(chosenVolumeShape, 'simple parallelepiped')
if ~all(isfinite(chosenVolume.delta))
[centerSpace, deltaMAX]=...
ypT3SS5fNB(...
experimentGeometryInfo, detectorPerimeterCoordinates, iub);
infiniteEdges= ~isfinite(chosenVolume.delta) | ~isfinite(chosenVolume.peak);
chosenVolume.peak(infiniteEdges)=centerSpace(infiniteEdges);
chosenVolume.delta(infiniteEdges)=deltaMAX(infiniteEdges);
end
end
[centerOfTheVolumeInAdditionalBasis,...
iTparallelepiped, cylinderRadiusSqr,...
volumeEdges, additionalInformationForGettingApplicableFrames,...
peak, delta, deltaFull, hl, hr, kl, kr, ll, lr]=...
f0jAPbb3EVLPtS(chosenVolume);useNewAxes=chosenVolume.useNewAxes;
[ applicableFrames, rotMatricesFull, isOkGetApplicableFrames      ] = h0tBk6Fi6MhHvT( surfacesInResiprocalSpace, volumeEdges, numOfBins, crackerData.ub, experimentGeometryInfo, chosenVolume, additionalInformationForGettingApplicableFrames );
if ~isOkGetApplicableFrames
return
end
[applicableFrames, isOkCutFrames]=I8xq0asezb(cutFrames, applicableFrames);
if ~isOkCutFrames
return
end
Icell=cell(length(applicableFrames), pixelFragmentation.pixelFragmentationAngle);
HKLhCell=cell(length(applicableFrames), pixelFragmentation.pixelFragmentationAngle);
HKLkCell=cell(length(applicableFrames), pixelFragmentation.pixelFragmentationAngle);
HKLlCell=cell(length(applicableFrames), pixelFragmentation.pixelFragmentationAngle);
if rememberFluxInEachFrame
fluxInEachFrameArrayCell=num2cell(zeros(1,length(applicableFrames))*nan);else
fluxInEachFrameArrayCell=[];
end
if rememberBgInEachFrame
bgInEachFrameArrayCell=num2cell(zeros(1,length(applicableFrames))*nan);else
bgInEachFrameArrayCell=[];
end
numOfPixelsInSubFrameCell = cell(length(applicableFrames), pixelFragmentation.pixelFragmentationAngle);
nh=[];nk=[];nl=[];
if makeReconstructionImmediately
nh = abs(make3DArray.nh);nk = abs(make3DArray.nk);
nl = abs(make3DArray.nl);ccp4I1 = zeros(nh*nk*nl, 1, outputType);ccp4I0 = ccp4I1;
if returnFrameNumbers
ccp4F = ccp4I1;ccp4F(:)=nan;
end
end
recVarPack=struct('delta', deltaFull,...
'peak', peak, 'hl', hl, 'hr', hr, 'kl', kl, 'kr', kr, 'll', ll, 'lr', lr, ...
'cutFrames', cutFrames, 'folderNameTemplate', experimentNames.folderNameTemplate, ...
'experimentInfo', experimentInfo, 'experimentNames', experimentNames, 'valueFromMultiDataset', valueFromMultiDataset, ...
'normalization', normalization, 'backgroundWindow', backgroundWindow, ...
'experimentIdentifierString', experimentNames.experimentIdentifierString, ...
'currentValueFromMultiDataset', valueFromMultiDataset.currentValue, 'useNormalizationString', useNormalizationString, ...
'imagesData', imagesData, 'runFileData', runFileData,...
'crackerData', crackerData, 'chosenVolume', chosenVolume, ...
'make3DArray', make3DArray, 'pixelFragmentation', pixelFragmentation, 'additionalTools', additionalTools, ...
'applicableFrames', applicableFrames, 'warnings', warningsStruct, 'meta', struct(), 'version', IRTmq__v() );
recVarPack.meta.datasetfullpath = currentDatasetFolderFullName;
recVarPack.meta.date = char(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z'));
if usePixelFragmentationDetector
[ rArray, ~, ~ ] = ij_8VJaiB24( ...
experimentGeometryInfo, pixelFragmentation, detectorMask, ...
detectorImageStoreGeometry, false, true);
end
rArrayBinning=pixelFragmentation.useLessMemoryCoefficient;
rArrayCell=cell(1, rArrayBinning);
rArrayCellNumOfElements = floor(length(rArray)/rArrayBinning);
rArrayCellShifts=[0:1:(rArrayBinning-1)]*rArrayCellNumOfElements;
for i=1:rArrayBinning
if i==rArrayBinning
rArrayCell{i}=rArray( :, (i-1)*rArrayCellNumOfElements + 1 : end );else
rArrayCell{i} = rArray( :, rArrayCellShifts(i) + 1 : i*rArrayCellNumOfElements );
end
end
rArrayCell=rArrayCell(:);clear('rArray');
RarrayExtended = wuDvdC1( experimentGeometryInfo, applicableFrames, pixelFragmentation);
vsd1=1;vsd2=3;vsd3=2;KArrayCenteredAbsLevel=[];LArrayCenteredAbsLevel=[];
if chosenVolumeUseSimpleParallelepiped ||chosenVolumeUseParallelepiped || chosenVolumeUseCylinder
if [delta(1) delta(3)]/delta(2)>2 | delta(3)./[delta(1) delta(2)]>2
vsd1=2;vsd2=1;vsd3=3;
if chosenVolumeUseCylinder
KArrayCenteredAbsLevel=3;LArrayCenteredAbsLevel=1;
end
elseif [delta(1) delta(2)]/delta(3)>2 | delta(1)./[delta(3) delta(2)]>2
vsd1=3;vsd2=2;vsd3=1;
if chosenVolumeUseCylinder
KArrayCenteredAbsLevel=2;LArrayCenteredAbsLevel=3;
end
else
vsd1=1;vsd2=3;vsd3=2;
if chosenVolumeUseCylinder
KArrayCenteredAbsLevel=1;LArrayCenteredAbsLevel=2;
end
end
centerVSD1=centerOfTheVolumeInAdditionalBasis(vsd1);
centerVSD2=centerOfTheVolumeInAdditionalBasis(vsd2);
centerVSD3=centerOfTheVolumeInAdditionalBasis(vsd3);
deltaVSD1=delta(vsd1);deltaVSD2=delta(vsd2);deltaVSD3=delta(vsd3);
end
RarrayExtendedSliced=RarrayExtended(applicableFrames, :);
imagesFullNamesSliced=imagesFullNames(applicableFrames);
if useubarray
iub_array = zeros(3,3, length(applicableFrames));
for i=1:length(applicableFrames)
iub_array(:,:,i) = inv(...
ub_array_txt(:,:,applicableFrames(i))...
);
end
else
iub_array = repmat(iub, 1, 1, length(applicableFrames));
end
timeReadingBegin=now;
for i=1:length(applicableFrames)
frameNumber = applicableFrames(i);
imageFileWasAlreadyRead = false;numOfPixelsInFrameCell=zeros(1, rArrayBinning);
I1DinCurrentFrameCell=cell(1, rArrayBinning);
H1DinCurrentFrameCell=cell(1, rArrayBinning);
K1DinCurrentFrameCell=cell(1, rArrayBinning);
L1DinCurrentFrameCell=cell(1, rArrayBinning);iub_ = squeeze(iub_array(:,:,i));
for an=1:pixelFragmentationAngle
R = RarrayExtendedSliced{i,an};iubIr = (iub_/R);
if chosenVolumeUseParallelepiped || chosenVolumeUseCylinder
TPariubIr=iTparallelepiped\iub_/R;
end
numOfPixelsInFrameCell=numOfPixelsInFrameCell*0;
for r=1:rArrayBinning
foundIsNotEmpty = false;
if chosenVolumeUseSimpleParallelepiped
HKL1Array=iubIr(vsd1,:)*rArrayCell{r};HKL1ArrayCenteredAbs=abs(HKL1Array-centerVSD1);
found1 = find(HKL1ArrayCenteredAbs<deltaVSD1);HKL1ArrayCenteredAbs=[];
if ~isempty(found1)
rArray2=rArrayCell{r}(:,found1);
HKL2Array=iubIr(vsd2,:)*rArray2;HKL2ArrayCenteredAbs=abs(HKL2Array-centerVSD2);
found2 = find(HKL2ArrayCenteredAbs<deltaVSD2);HKL2ArrayCenteredAbs=[];
if ~isempty(found2)
rArray3=rArray2(:, found2);
HKL3Array=iubIr(vsd3,:)*rArray3;HKL3ArrayCenteredAbs=abs(HKL3Array-centerVSD3);
found3 = find(HKL3ArrayCenteredAbs<deltaVSD3);HKL3ArrayCenteredAbs=[];
if ~isempty(found3)
found2=found2(found3);found = found1(found2);found1=[];
if vsd1==1
H1DinCurrentFrameCell{r}=HKL1Array(found);
elseif vsd1==2
K1DinCurrentFrameCell{r}=HKL1Array(found);
elseif vsd1==3
L1DinCurrentFrameCell{r}=HKL1Array(found);else
disp('CRYTICAL ERROR')
pause
continue
end
if vsd2==1
H1DinCurrentFrameCell{r}=HKL2Array(found2);
elseif vsd2==2
K1DinCurrentFrameCell{r}=HKL2Array(found2);
elseif vsd2==3
L1DinCurrentFrameCell{r}=HKL2Array(found2);else
disp('CRYTICAL ERROR')
pause
continue
end
found2=[];
if vsd3==1
H1DinCurrentFrameCell{r}=HKL3Array(found3);
elseif vsd3==2
K1DinCurrentFrameCell{r}=HKL3Array(found3);
elseif vsd3==3
L1DinCurrentFrameCell{r}=HKL3Array(found3);else
disp('CRYTICAL ERROR')
pause
continue
end
found3=[];found = found + rArrayCellShifts(r);foundIsNotEmpty=true;
end
end
end
elseif chosenVolumeUseParallelepiped || chosenVolumeUseCylinder
HKL1Array=TPariubIr(vsd1,:)*rArrayCell{r};HKL1ArrayCenteredAbs=abs(HKL1Array-centerVSD1);
found1 = find(HKL1ArrayCenteredAbs<deltaVSD1);
if chosenVolumeUseCylinder || vsd1~=1
if vsd1==2
KArrayCenteredAbs=HKL1ArrayCenteredAbs;
elseif vsd1==3
LArrayCenteredAbs=HKL1ArrayCenteredAbs;
end
end
HKL1ArrayCenteredAbs=[];
if ~useNewAxes
HKL1Array=[];
end
if ~isempty(found1)
rArray2=rArrayCell{r}(:,found1);
HKL2Array=TPariubIr(vsd2,:)*rArray2;HKL2ArrayCenteredAbs=abs(HKL2Array-centerVSD2);
found2 = find(HKL2ArrayCenteredAbs<deltaVSD2);
if chosenVolumeUseCylinder || vsd2~=1
if vsd2==2
KArrayCenteredAbs=HKL2ArrayCenteredAbs;
elseif vsd2==3
LArrayCenteredAbs=HKL2ArrayCenteredAbs;
end
end
HKL2ArrayCenteredAbs=[];
if ~useNewAxes
HKL2Array=[];
end
if ~isempty(found2)
rArray3=rArray2(:, found2);
HKL3Array=TPariubIr(vsd3,:)*rArray3;HKL3ArrayCenteredAbs=abs(HKL3Array-centerVSD3);
found3 = find(HKL3ArrayCenteredAbs<deltaVSD3);
if chosenVolumeUseCylinder || vsd1~=1
if vsd3==2
KArrayCenteredAbs=HKL3ArrayCenteredAbs;
elseif vsd3==3
LArrayCenteredAbs=HKL3ArrayCenteredAbs;
end
end
HKL3ArrayCenteredAbs=[];
if chosenVolumeUseCylinder
if LArrayCenteredAbsLevel==1
LArrayCenteredAbs=LArrayCenteredAbs(found3);
elseif LArrayCenteredAbsLevel==2
LArrayCenteredAbs=LArrayCenteredAbs(found2(found3));
elseif LArrayCenteredAbsLevel==3
LArrayCenteredAbs=LArrayCenteredAbs(found1(found2(found3)));
end
if KArrayCenteredAbsLevel==1
KArrayCenteredAbs=KArrayCenteredAbs(found3);
elseif KArrayCenteredAbsLevel==2
KArrayCenteredAbs=KArrayCenteredAbs(found2(found3));
elseif KArrayCenteredAbsLevel==3
KArrayCenteredAbs=KArrayCenteredAbs(found1(found2(found3)));
end
HKLarrayCenteredAbsSqr=KArrayCenteredAbs.^2+LArrayCenteredAbs.^2;
KArrayCenteredAbs=[];LArrayCenteredAbs=[];
found3=found3(HKLarrayCenteredAbsSqr<cylinderRadiusSqr);HKLarrayCenteredAbsSqr=[];
end
if ~useNewAxes
HKL3Array=[];
end
if ~isempty(found3)
if useNewAxes
found2=found2(found3);
found = found1(found2);HKL1Array=HKL1Array(found);HKL2Array=HKL2Array(found2);
HKL3Array=HKL3Array(found3);found1=[];found2=[];found3=[];else
found = found1(found2(found3));
found1=[];found2=[];found3=[];rArrayCellFounded=rArrayCell{r}(:, found);
HKL1Array=iubIr(vsd1,:)*(rArrayCellFounded);HKL2Array=iubIr(vsd2,:)*(rArrayCellFounded);
HKL3Array=iubIr(vsd3,:)*(rArrayCellFounded);rArrayCellFounded=[];
end
switch vsd1
case 1
H1DinCurrentFrameCell{r}=HKL1Array;case 2
K1DinCurrentFrameCell{r}=HKL1Array;case 3
L1DinCurrentFrameCell{r}=HKL1Array;otherwise
disp('ERROR')
pause
end
switch vsd2
case 1
H1DinCurrentFrameCell{r}=HKL2Array;case 2
K1DinCurrentFrameCell{r}=HKL2Array;case 3
L1DinCurrentFrameCell{r}=HKL2Array;otherwise
disp('ERROR')
pause
end
switch vsd3
case 1
H1DinCurrentFrameCell{r}=HKL3Array;case 2
K1DinCurrentFrameCell{r}=HKL3Array;case 3
L1DinCurrentFrameCell{r}=HKL3Array;otherwise
disp('ERROR')
pause
end
HKL1Array=[];
HKL2Array=[];HKL3Array=[];found = found + rArrayCellShifts(r);foundIsNotEmpty=true;
end
end
end
end
if ~foundIsNotEmpty
numOfPixelsInFrameCell(r) = 0;
I1DinCurrentFrameCell{r} = [];H1DinCurrentFrameCell{r} = [];
K1DinCurrentFrameCell{r} = [];L1DinCurrentFrameCell{r} = [];continue
end
numOfPixelsInFrameCell(r) = length(H1DinCurrentFrameCell{r});
if ~imageFileWasAlreadyRead
switch imageFormat
case 'cbf'
[I2D, imageHead] = bqdopOyfbsiQElql(imagesFullNamesSliced{i});
imageHeadFlux=imageHead.Flux;
if isempty(imageHeadFlux)
imageHeadFlux=nan;
end
case 'img'
[I2D, imageHead] = OZNnACLMtb1Y0vkK(imagesFullNamesSliced{i});imageHeadFlux=1;case 'edf'
[I2D, imageHead] = CZ09aDxngjTvV3V(imagesFullNamesSliced{i});
imageHeadFlux=1;case 'esperanto'
[I2D, imageHead] = AWo1Glkgc8Ljn(imagesFullNamesSliced{i});imageHeadFlux=1;
if imageHead.monitor(1)~=0
imageHeadFlux=imageHead.monitor(1);
end
end
I2D=double(I2D);
if useDetectorMask
I2D(detectorMaskMatrix)=nan;
end
if excludenegativepixels
I2D (I2D<0) = nan;
end
if solid_angle_use
I2D=I2D.*normOnSolidAngle;
end
imageFileWasAlreadyRead = true;
if useNormalizationOn==1 || rememberBgInEachFrame
if strcmpi(bgWindowCoordinateSystem, 'snblAlbula')
bg2Darray = I2D(bgWindowX1:bgWindowX2, bgWindowY1:bgWindowY2);
elseif any(strcmpi(bgWindowCoordinateSystem, {'snblCrysalis', 'crysalis'}))
Isz = size(I2D);bg_d1_s = Isz(1) - bgWindowY2 +1;
bg_d1_e = Isz(1) - bgWindowY1 +1;bg_d2_s = Isz(2) - bgWindowX2 +1;
bg_d2_e = Isz(2) - bgWindowX1 +1;bg2Darray = I2D(bg_d1_s:bg_d1_e, bg_d2_s:bg_d2_e);else
bg2Darray=nan;
end
meanBg2Darray=mean(bg2Darray(:));
end
if rememberFluxInEachFrame
fluxInEachFrameArrayCell{i}=imageHeadFlux;
end
if rememberBgInEachFrame
bgInEachFrameArrayCell{i}=meanBg2Darray;
end
end
if usePixelFragmentation==true
found=floor((found-0.01)/(fD*fdB*sdB))*fD + ceil(rem((found-0.01), fD*fdB)/fdB);
I1DinCurrentFrameCell{r}=I2D(found);else
I1DinCurrentFrameCell{r}=I2D(found);
end
end
numOfPixelsInFrameSubAngle = sum(numOfPixelsInFrameCell);
numOfPixelsInSubFrameCell{i, an} = numOfPixelsInFrameSubAngle;
if numOfPixelsInFrameSubAngle>0
I1DinCurrentFrameSubAngle=cell2mat(I1DinCurrentFrameCell)';
H1DinCurrentFrameSubAngle=cell2mat(H1DinCurrentFrameCell)';
K1DinCurrentFrameSubAngle=cell2mat(K1DinCurrentFrameCell)';
L1DinCurrentFrameSubAngle=cell2mat(L1DinCurrentFrameCell)';
if useNormalizationOn==1
bgMultiplier 	  = bg2DarrayMeanZero/meanBg2Darray;
I1DinCurrentFrameSubAngle = I1DinCurrentFrameSubAngle*bgMultiplier 	 ;
elseif useNormalizationOn==2
bgMultiplier 	  = fluxZero/imageHeadFlux;
I1DinCurrentFrameSubAngle = I1DinCurrentFrameSubAngle*bgMultiplier 	 ;else
bgMultiplier=nan;
end
if useNormalizationOn==1
disp(['Reading frame: ' num2str(frameNumber) '     found=' num2str(numOfPixelsInFrameSubAngle) '     normalization on:' num2str(1/bgMultiplier 	 )])
elseif useNormalizationOn==2
disp(['Reading frame: ' num2str(frameNumber) '     found=' num2str(numOfPixelsInFrameSubAngle) '     normalization on (Flux):' num2str(1/bgMultiplier 	 )])
else
disp(['Reading frame: ', num2str(frameNumber), '     found=', num2str(numOfPixelsInFrameSubAngle)])
end
else
disp(['Reading frame: ', num2str(frameNumber)])
I1DinCurrentFrameSubAngle=[];
H1DinCurrentFrameSubAngle=[];K1DinCurrentFrameSubAngle=[];L1DinCurrentFrameSubAngle=[];
end
if makeReconstructionImmediately==1 && ~isempty(I1DinCurrentFrameSubAngle)
if useDetectorMask || excludenegativepixels
f=find(isnan(I1DinCurrentFrameSubAngle));
I1DinCurrentFrameSubAngle(f)=[];H1DinCurrentFrameSubAngle(f)=[];
K1DinCurrentFrameSubAngle(f)=[];L1DinCurrentFrameSubAngle(f)=[];
end
if ~isempty(I1DinCurrentFrameSubAngle)
if outputTypeSingle
I1DinCurrentFrameSubAngle=single(I1DinCurrentFrameSubAngle);
end
HKLnCur = floor(abs(L1DinCurrentFrameSubAngle-ll)/(2*delta(3)/nl))*(nh*nk)+...
floor(abs(K1DinCurrentFrameSubAngle-kl)/(2*delta(2)/nk))*nh+...
ceil(abs(H1DinCurrentFrameSubAngle-hl)/(2*delta(1)/nh));
foundPixelsOutOfBorders=find(HKLnCur>nh*nk*nl);
if ~isempty(foundPixelsOutOfBorders)
f=foundPixelsOutOfBorders;
HKLnCur(f) = floor(abs(L1DinCurrentFrameSubAngle(f)-1e-5-ll)/(2*delta(3)/nl))*(nh*nk)+...
floor(abs(K1DinCurrentFrameSubAngle(f)-1e-5-kl)/(2*delta(2)/nk))*nh+...
ceil(abs(H1DinCurrentFrameSubAngle(f)-1e-5-hl)/(2*delta(1)/nh));
end
for ci=1:numel(I1DinCurrentFrameSubAngle)
ccp4I1(HKLnCur(ci)) = ccp4I1(HKLnCur(ci)) + ...
I1DinCurrentFrameSubAngle(ci);ccp4I0(HKLnCur(ci)) = ccp4I0(HKLnCur(ci))+1;
end
if returnFrameNumbers
ccp4Ftmp=zeros(nh*nk*nl, 1, outputType)+nan;
ccp4Ftmp(HKLnCur)=(frameNumber);ccp4F=min(ccp4F, ccp4Ftmp);
end
end
else
if outputTypeSingle
Icell{i, an} = single(I1DinCurrentFrameSubAngle);
HKLhCell{i, an} = single(H1DinCurrentFrameSubAngle);
HKLkCell{i, an} = single(K1DinCurrentFrameSubAngle);
HKLlCell{i, an} = single(L1DinCurrentFrameSubAngle);else
Icell{i, an} = I1DinCurrentFrameSubAngle;HKLhCell{i, an} = H1DinCurrentFrameSubAngle;
HKLkCell{i, an} = K1DinCurrentFrameSubAngle;HKLlCell{i, an} = L1DinCurrentFrameSubAngle;
end
end
end
end
clear('rArrayCell', 'HKLhArray', 'HKLkArray', 'HKLlArray', 'found', 'found1', 'found2', 'found3',...
'H1DinCurrentFrameSubAngle', 'K1DinCurrentFrameSubAngle', 'L1DinCurrentFrameSubAngle', 'I1DinCurrentFrameSubAngle',...
'H1DinCurrentFrameCell', 'K1DinCurrentFrameCell', 'L1DinCurrentFrameCell', 'I1DinCurrentFrameCell',...
'HKLnCur', 'f', 'ccp4Ftmp')
timeReading=(now-timeReadingBegin)*24*3600;
disp(['elapsing time for reading and analysing all the frames: ', num2str(timeReading)])
if makeReconstructionImmediately==false
I1D=cell2mat(reshape(transpose(Icell),[],1));clear('Icell')
H1D=cell2mat(reshape(transpose(HKLhCell),[],1));clear('HKLhCell')
K1D=cell2mat(reshape(transpose(HKLkCell),[],1));clear('HKLkCell')
L1D=cell2mat(reshape(transpose(HKLlCell),[],1));clear('HKLlCell')
if returnFrameNumbers
numOfPixelsInSubFrame=cell2mat(numOfPixelsInSubFrameCell);
numOfPixelsInFrame=sum(numOfPixelsInSubFrame, 2);
numOfPixelsInVolume = sum(numOfPixelsInFrame);
applicableFramesInt16=int16(applicableFrames);
F1D = zeros(numOfPixelsInVolume, 1, 'int16');frameIndexBegin=1;
for i=1:length(applicableFramesInt16)
if numOfPixelsInFrame(i)>0
frameIndexEnd=frameIndexBegin+numOfPixelsInFrame(i)-1;
F1D(frameIndexBegin:frameIndexEnd) = applicableFramesInt16(i);
frameIndexBegin=frameIndexEnd+1;
end
end
end
else
I1D = [];H1D = [];K1D = [];L1D = [];
if returnFrameNumbers
ccp4Fnan=isnan(ccp4F);
ccp4F=int16(round(ccp4F));ccp4F(ccp4Fnan)=-1;ccp4F=reshape(ccp4F, nh, nk, nl);
end
ccp4I1=ccp4I1./ccp4I0;
ccp4I1=reshape(ccp4I1, nh, nk, nl);ccp4I0=reshape(ccp4I0, nh, nk, nl);
if make3DArray.makeAxes==true
if make3DArray.nh==-1
ccp4I1=repmat(ccp4I1, [2,1,1]);ccp4I0=repmat(ccp4I0, [2,1,1]);
elseif make3DArray.nk==-1
ccp4I1=repmat(ccp4I1, [1,2,1]);ccp4I0=repmat(ccp4I0, [1,2,1]);
elseif make3DArray.nl==-1
ccp4I1=repmat(ccp4I1, [1,1,2]);ccp4I0=repmat(ccp4I0, [1,1,2]);
end
end
if strcmp(outputType, 'single')
ccp4I1=single(ccp4I1);
end
if make3DArray.makeCounts==false
ccp4I0=[];
end
end
if rememberFluxInEachFrame
fluxInEachFrameArray=zeros(max(applicableFrames), 1)*nan;
for i=1:length(fluxInEachFrameArrayCell)
fluxInEachFrameArray(applicableFrames(i))=fluxInEachFrameArrayCell{i};
end
recVarPack.fluxInEachFrameArray=fluxInEachFrameArray;
end
if rememberBgInEachFrame
bgInEachFrameArray=zeros(max(applicableFrames), 1)*nan;
for i=1:length(bgInEachFrameArrayCell)
bgInEachFrameArray(applicableFrames(i))=bgInEachFrameArrayCell{i};
end
recVarPack.bgInEachFrameArray=bgInEachFrameArray;
end
warningsCellArray=warningsStruct.warningsCellArray(1:warningsStruct.numOfWarnings);
recVarPack.warnings=warningsCellArray;isOkMain=true;
end
function [ currentDatasetName, currentDatasetFolderFullName, warningsStruct ] =...
OK26MDiA05sPl74(experimentNames, valueFromMultiDataset, warningsStruct)
if valueFromMultiDataset.thereIsOnlyOneFolderInDataset
currentDatasetFolderFullName=experimentNames.mainFolder;
[~,currentDatasetName,~,~]=fZfileParts(currentDatasetFolderFullName);else
[nameTemplateStrBegin, nameTemplateStrEnd] = SN5XEBn7UuSH(experimentNames.folderNameTemplate);
if valueFromMultiDataset.digitsAreDynamic
currentDatasetName = [nameTemplateStrBegin, num2str(valueFromMultiDataset.currentValue), nameTemplateStrEnd];
else
currentDatasetName = [nameTemplateStrBegin, num2str(valueFromMultiDataset.currentValue, ...
['%0.' num2str(valueFromMultiDataset.digits) 'd']), nameTemplateStrEnd];
end
if valueFromMultiDataset.typeIndexInName
if valueFromMultiDataset.indexDigitsAreDynamic
indexStr=num2str(valueFromMultiDataset.currentIndex);else
indexStr=num2str(valueFromMultiDataset.currentIndex, ...
['%0.' num2str(valueFromMultiDataset.indexDigits) 'd']);
end
currentDatasetName=[ valueFromMultiDataset.indexSuffix indexStr currentDatasetName ];
end
additionalExperimentSubPath=experimentNames.additionalExperimentSubPath;
currentDatasetFolderFullName=fullfile(experimentNames.mainFolder, currentDatasetName, additionalExperimentSubPath);
end
end
function [centerOfTheVolumeInAdditionalBasis, ...
iTparallelepiped, cylinderRadiusSqr,...
volumeEdges, additionalInformationForGettingApplicableFrames,...
peak, delta, deltaFull, hl, hr, kl, kr, ll, lr]=...
f0jAPbb3EVLPtS(chosenVolume)
chosenVolumeShape=chosenVolume.shape;
additionalInformationForGettingApplicableFrames=struct;
volumeEdges=[];cylinderRadiusSqr=[];iTparallelepiped=[];switch chosenVolumeShape
case 'sphere'
peak=chosenVolume.sphere.center;delta=chosenVolume.sphere.radius*[1 1 1];
ch=peak(1);ck=peak(2);cl=peak(3);dh=delta(1);dk=delta(2);dl=delta(3);
hl=ch-dh;hr=ch+dh;kl=ck-dk;kr=ck+dk;ll=cl-dl;lr=cl+dl;
case {'parallelepiped', 'simple parallelepiped', 'cylinder'}
switch chosenVolumeShape
case {'parallelepiped', 'cylinder'}
switch chosenVolumeShape
case 'parallelepiped'
zeroVertex=chosenVolume.parallelepiped.zeroVertex(:)';
axe1=chosenVolume.parallelepiped.axe1(:)';
axe2=chosenVolume.parallelepiped.axe2(:)';axe3=chosenVolume.parallelepiped.axe3(:)';
axe1Norm=norm(axe1);axe2Norm=norm(axe2);axe3Norm=norm(axe3);
axe1Normed=axe1/axe1Norm;axe2Normed=axe2/axe2Norm;axe3Normed=axe3/axe3Norm;
centerOfTheVolume=(zeroVertex+(axe1+axe2+axe3)/2);case 'cylinder'
cylinderRadius=chosenVolume.cylinder.radius;cylinderRadiusSqr=cylinderRadius^2;
if ~chosenVolume.cylinder.useTopBottomPoints
cylinderDirection=chosenVolume.cylinder.direction;
cylinderLength=chosenVolume.cylinder.length;
cylinderCenter=chosenVolume.cylinder.center;else
cylinderBottomToTopVector=chosenVolume.cylinder.topPoint-chosenVolume.cylinder.bottomPoint;
cylinderDirection=cylinderBottomToTopVector;
cylinderLength=norm(cylinderBottomToTopVector);
cylinderCenter=(chosenVolume.cylinder.topPoint+chosenVolume.cylinder.bottomPoint)/2;
end
[~, axe1Normed, axe2Normed, axe3Normed]=twmBsAr3b(cylinderDirection);
axe1Norm=cylinderLength;axe2Norm=cylinderRadius*2;
axe3Norm=cylinderRadius*2;axe1=axe1Normed/norm(axe1Normed)*axe1Norm;
axe2=axe2Normed/norm(axe2Normed)*axe2Norm;axe3=axe3Normed/norm(axe3Normed)*axe3Norm;
zeroVertex=cylinderCenter-(axe1+axe2+axe3)/2;centerOfTheVolume=cylinderCenter;
end
additionalInformationForGettingApplicableFrames.axe1=axe1;
additionalInformationForGettingApplicableFrames.axe2=axe2;
additionalInformationForGettingApplicableFrames.axe3=axe3;
additionalInformationForGettingApplicableFrames.axe1Norm=axe1Norm;
additionalInformationForGettingApplicableFrames.axe2Norm=axe2Norm;
additionalInformationForGettingApplicableFrames.axe3Norm=axe3Norm;
additionalInformationForGettingApplicableFrames.zeroVertex=zeroVertex;
iTparallelepiped=[axe1Normed;axe2Normed;axe3Normed]';
centerOfTheVolumeInAdditionalBasis=(iTparallelepiped\(centerOfTheVolume'))';
chosenVolume.centerOfTheVolume=centerOfTheVolume;
chosenVolume.centerOfTheVolumeInAdditionalBasis=centerOfTheVolumeInAdditionalBasis;
peak=centerOfTheVolume;delta=abs([axe1Norm, axe2Norm, axe3Norm]/2);
ch=centerOfTheVolumeInAdditionalBasis(1);ck=centerOfTheVolumeInAdditionalBasis(2);cl=centerOfTheVolumeInAdditionalBasis(3);
dh=delta(1);dk=delta(2);dl=delta(3);
hl=ch-dh;hr=ch+dh;kl=ck-dk;kr=ck+dk;ll=cl-dl;lr=cl+dl;
tmpVertexOfTheParallelepiped=zeros(8, 3);
tmpVertexOfTheParallelepiped(1, :)=abs(zeroVertex-peak);
tmpVertexOfTheParallelepiped(2, :)=abs(zeroVertex+axe1-peak);
tmpVertexOfTheParallelepiped(3, :)=abs(zeroVertex+axe2-peak);
tmpVertexOfTheParallelepiped(4, :)=abs(zeroVertex+axe3-peak);
tmpVertexOfTheParallelepiped(5, :)=abs(zeroVertex+axe1+axe2-peak);
tmpVertexOfTheParallelepiped(6, :)=abs(zeroVertex+axe2+axe3-peak);
tmpVertexOfTheParallelepiped(7, :)=abs(zeroVertex+axe3+axe1-peak);
tmpVertexOfTheParallelepiped(8, :)=abs(zeroVertex+axe1+axe2+axe3-peak);
deltaFull=[max(tmpVertexOfTheParallelepiped(:, 1)) max(tmpVertexOfTheParallelepiped(:, 2)) max(tmpVertexOfTheParallelepiped(:, 3))];
chosenVolume.deltaFull=deltaFull;
chosenVolume.iTparallelepiped=iTparallelepiped;case 'simple parallelepiped'
peak=chosenVolume.peak;delta=chosenVolume.delta;
centerOfTheVolume=peak;centerOfTheVolumeInAdditionalBasis=centerOfTheVolume;
deltaFull=delta;chosenVolume.deltaFull=deltaFull;
ch=peak(1);ck=peak(2);cl=peak(3);dh=delta(1);dk=delta(2);dl=delta(3);
hl=ch-dh;hr=ch+dh;kl=ck-dk;kr=ck+dk;ll=cl-dl;lr=cl+dl;
volumeEdges = [[hl, hr];[kl, kr];[ll, lr]];
end
end
end
function [Izero, fD, sD, headZero, fluxZero, isOk]=MNzXp6am(imageFormat, firstImageFullName)
Izero=[];fD=[];sD=[];headZero=[];fluxZero=[];isOk=false;switch imageFormat
case 'cbf'
[Izero, headZero, ~] = bqdopOyfbsiQElql(firstImageFullName);
fD=headZero.X_Binary_Size_Fastest_Dimension;sD=headZero.X_Binary_Size_Second_Dimension;
fluxZero = 200000*headZero.Exposure_time;case 'img'
[Izero, headZero, ~] = OZNnACLMtb1Y0vkK(firstImageFullName);
fD=headZero.nx;sD=headZero.ny;fluxZero = 1;case 'edf'
[Izero, headZero, ~] = CZ09aDxngjTvV3V(firstImageFullName);
fD=headZero.Dim_1;sD=headZero.Dim_2;fluxZero = 1;case 'esperanto'
[Izero, headZero, ~] = AWo1Glkgc8Ljn(firstImageFullName);
fD=headZero.dimension_1;sD=headZero.dimension_2;fluxZero = 1;otherwise
warning('wrong image format')
return
end
fD=double(fD);sD=double(sD);isOk=true;
end
function [centerSpace, deltaMAX]=ypT3SS5fNB(experimentGeometryInfo, detectorPerimeterCoordinates, iub)
RarrayTmp = wuDvdC1( experimentGeometryInfo, [], [] );
HKLmax=zeros(3,length(RarrayTmp))*nan;HKLmin=HKLmax;
for i=1:length(RarrayTmp)
HKL=(iub/RarrayTmp{i})*detectorPerimeterCoordinates;
HKLmax(1,i)=max(HKL(1,:));HKLmin(1,i)=min(HKL(1,:));HKLmax(2,i)=max(HKL(2,:));
HKLmin(2,i)=min(HKL(2,:));HKLmax(3,i)=max(HKL(3,:));HKLmin(3,i)=min(HKL(3,:));
end
HKLMAX=[nan nan nan];HKLMIN=[nan nan nan];
HKLMAX(1)=max(HKLmax(1,:));HKLMAX(2)=max(HKLmax(2,:));HKLMAX(3)=max(HKLmax(3,:));
HKLMIN(1)=min(HKLmin(1,:));HKLMIN(2)=min(HKLmin(2,:));HKLMIN(3)=min(HKLmin(3,:));
deltaMAX=(HKLMAX-HKLMIN)/2;centerSpace=(HKLMAX+HKLMIN)/2;clear('RarrayTmp')
end
function [applicableFrames, isOkCutFrames]=I8xq0asezb(cutFrames, applicableFrames)
isOkCutFrames=false;
if size(cutFrames,1)==1 && size(cutFrames,2)==2
applicableFrames=...
applicableFrames(applicableFrames>=cutFrames(1) & applicableFrames<=cutFrames(2));
elseif size(cutFrames,1)>1 && size(cutFrames,2)==2
applicableFramesOverflowing=...
applicableFrames(applicableFrames>=cutFrames(1,1) & applicableFrames<=cutFrames(1,2));
for cuti=2:size(cutFrames,1)
applicableFramesOverflowing=...
[applicableFramesOverflowing;...
applicableFrames(applicableFrames>=cutFrames(cuti,1) & applicableFrames<=cutFrames(cuti,2))...
];
end
applicableFrames=unique(applicableFramesOverflowing);else
return
end
isOkCutFrames=true;
end
function runFileData=s_RCJe3(runFileData)
if strcmp(runFileData.scansStruct.scanned_axe, 'phi') && runFileData.scansStruct.phi~=0
warning('SHOULD BE REWRITTEN CAREFULLY: Fixing snbl *.run file')
for i=1:numel(runFileData.scansStruct)
runFileData.scansStruct(i).omega=runFileData.scansStruct(i).phi;
runFileData.scansStruct(i).phi=0;
end
end
end
function [useScan, isScanIndexOk]=OKuPF1r8_h5e4B(useScan, runFileData, imagesData)
isScanIndexOk=false;useScan=abs(round(useScan(1)));
if isempty(useScan) || ~isnumeric(useScan)...
|| numel(useScan)~=1 || ~isreal(useScan) || useScan<1
useScan=1;
end
if useScan>length(runFileData.scansStruct)
warning(['scan number ' num2str(useScan) ' is absent in the *.run file'])
return
end
if useScan>length(imagesData)
warning(['scan number ' num2str(useScan) ' is absent in the images folder'])
return
end
if ~imagesData(useScan).scanIsFull
warning(['cann''t find some frames for the scan number ' num2str(useScan)])
end
if runFileData.scansStruct(useScan).done~=imagesData(useScan).requiredNumberOfImagesInScan
warning('number of images in folder & *.run file are different')
return
end
isScanIndexOk=true;
end
function [detectorImageStoreGeometry, isOkDetectorType]=GYYlBYDrgEZbpN(imageFormat, headZero)
detectorGeometryType=imageFormat;
if strcmp(imageFormat, 'cbf')
if ~isempty(headZero) && isfield(headZero, 'Detector') &&...
(strcmpi(headZero.Detector, 'PILATUS 2M, 24-0111') || strcmpi(headZero.Detector, 'PILATUS 2M 24-0111'))
detectorGeometryType='PILATUS 2M, 24-0111';
end
end
[detectorImageStoreGeometry, isOkDetectorType]=...
KAgjA40wxXJWo( detectorGeometryType );
if ~isOkDetectorType
warning('cann''t find information about the geometrical location of the images on the detector')
end
end
function detector_second_area = Etm4PNOzs3s(experimentGeometryInfo, rArray)
g = experimentGeometryInfo;
s0 = MZTpYH9gfUM2Qh(3, g.beam.b3)*MZTpYH9gfUM2Qh(2, g.beam.b2)*[-1;0;0];
switch g.scanned_axe
case 'omega'
rotation_axe = [0 0 1]';case 'phi'
rotation_axe = ...
MZTpYH9gfUM2Qh(3,g.omega)*...
MZTpYH9gfUM2Qh(2,g.alpha)*MZTpYH9gfUM2Qh(3,g.kappa)*MZTpYH9gfUM2Qh(2,-g.alpha)*...
MZTpYH9gfUM2Qh(2,g.beta)*...
[0 0 1]';otherwise
warning('error')
return
end
test_vector = cross(s0 ,rotation_axe)';detector_second_area = test_vector*rArray <= 0;
end
function [ coordsArray, surfacesInResiprocalSpace, solidAngle, detectorPerimeterCoordinates, detectorFragments ] =...
ij_8VJaiB24(...
experimentGeometryInfo, pixelFragmentation, detectorMask,  ...
detectorImageStoreGeometry, ...
calculateAdditionalStaff, calculateMainCoordinatesArray)
coordsArray=[];surfacesInResiprocalSpace=[];
solidAngle=[];detectorPerimeterCoordinates=[];detectorFragments=[];
beam=experimentGeometryInfo.beam;detector=experimentGeometryInfo.detector;xdb=1;ydb=1;
if ~calculateAdditionalStaff && ~isempty(pixelFragmentation) &&...
pixelFragmentation.usePixelFragmentation
xdb=pixelFragmentation.pixelFragmentationX;ydb=pixelFragmentation.pixelFragmentationY;
end
rArrayBinning = 100;d=detector;detector_theta=experimentGeometryInfo.detector_thetaArm;
E0E1E2 = MZTpYH9gfUM2Qh(3, detector_theta)*MZTpYH9gfUM2Qh(3, d.d3)*MZTpYH9gfUM2Qh(2, d.d2)*MZTpYH9gfUM2Qh(1, d.d1);
E1E2E0 = E0E1E2(:, [2,3,1]);eDetector = E1E2E0;
s0 = MZTpYH9gfUM2Qh(3, beam.b3)*MZTpYH9gfUM2Qh(2, beam.b2)*[-1;0;0];fastestDimention=...
detectorImageStoreGeometry.imageFastestDimentionOrientation;
[x_start, x___end, y_start, y___end] = ...
ZtsdKy3I7(...
detectorImageStoreGeometry, detector, xdb, ydb);
xd=d.dimentionX;yd=d.dimentionY;XD=xd*xdb;YD=yd*ydb;xLine=linspace(x_start, x___end, XD);
yLine=linspace(y_start, y___end, YD);switch fastestDimention
case 'strings'
xReshape=repmat(xLine',YD,1);
yReshape=repmat(yLine ,XD,1);yReshape=yReshape(:);case 'columns'
xReshape=repmat(xLine ,YD,1);xReshape=xReshape(:);yReshape=repmat(yLine',XD,1);otherwise
warning('bad detector image store parameters')
return
end
coordsArray=[xReshape  yReshape];clear('xReshape', 'yReshape')
coordsArray(:,3)=-d.d;coordsArray=coordsArray';
if calculateAdditionalStaff
[surfacesInResiprocalSpace, solidAngle, detectorPerimeterCoordinates] =...
AAOsrQUFCvU(...
detectorImageStoreGeometry, coordsArray, eDetector, E0E1E2, s0, d);
end
if calculateMainCoordinatesArray
numOfPixels = size(coordsArray, 2);
numOfPixelsInOneCell = floor(numOfPixels/rArrayBinning);
for r=1:rArrayBinning
indexBegin=(r-1)*numOfPixelsInOneCell+1;
if r~=rArrayBinning
indexEnd=r*numOfPixelsInOneCell;else
indexEnd=numOfPixels;
end
normSp=sqrt(sum(coordsArray(:, indexBegin:indexEnd).^2,1));
rArray = (eDetector*coordsArray(:, indexBegin:indexEnd));
rArray(1, :)=rArray(1, :)./normSp-s0(1);
rArray(2, :)=rArray(2, :)./normSp-s0(2);rArray(3, :)=rArray(3, :)./normSp-s0(3);
coordsArray(:, indexBegin:indexEnd)=rArray;clear('rArray')
end
else
clear('coordsArray')
coordsArray=[];
end
end
function [x_start, x___end, y_start, y___end] = ...
ZtsdKy3I7(...
detectorImageStoreGeometry, detector, pixel_fragmentation_x, pixel_fragmentation_y)
xdb = pixel_fragmentation_x;ydb = pixel_fragmentation_y;
firstPixelPosition=detectorImageStoreGeometry.firstPixelPosition;
switch firstPixelPosition
case 'lower left corner'
rotate90=0;case 'upper left corner'
rotate90=1;case 'upper right corner'
rotate90=2;case 'lower right corner'
rotate90=3;
end
d=detector;pixelSizeX=d.pixelSizeX;
pixelSizeY=d.pixelSizeY;xd=d.dimentionX;yd=d.dimentionY;cornersX_default=[
-d.x0 (xd-1)-d.x0
-d.x0 (xd-1)-d.x0
];cornersY_default=[
(yd-1)-d.y0 (yd-1)-d.y0
-d.y0 -d.y0
];cornersX_default(:,1)=cornersX_default(:,1)+(-0.5+0.5/xdb);
cornersX_default(:,2)=cornersX_default(:,2)+( 0.5-0.5/xdb);
cornersY_default(1,:)=cornersY_default(1,:)+( 0.5-0.5/ydb);
cornersY_default(2,:)=cornersY_default(2,:)+(-0.5+0.5/ydb);
cornersX_default=cornersX_default*pixelSizeX;
cornersY_default=cornersY_default*pixelSizeY;
cornersX=rot90(cornersX_default,rotate90);cornersY=rot90(cornersY_default,rotate90);
x_start=cornersX(2,1);x___end=cornersX(1,2);y_start=cornersY(2,1);y___end=cornersY(1,2);
end
function [surfacesInResiprocalSpace, solidAngle, detectorPerimeterCoordinates] =...
AAOsrQUFCvU(...
detectorImageStoreGeometry, coordsArray, eDetector, E0E1E2, s0, detector)
switch detectorImageStoreGeometry.imageFastestDimentionOrientation
case 'strings'
fd=detector.dimentionX;sd=detector.dimentionY;case 'columns'
sd=detector.dimentionX;fd=detector.dimentionY;otherwise
warning('bad detector image store parameters')
return
end
surfacesInResiprocalSpace=struct;detectorVerticesIndexes=[1; fd*sd; fd; ( (sd-1)*fd+1 )];
coordsArrayDetectorVertices=coordsArray(:,detectorVerticesIndexes);
normSp=sqrt(sum(coordsArrayDetectorVertices.^2,1));
rArrayDetectorVertices = (eDetector*coordsArrayDetectorVertices);
rArrayDetectorVertices(1, :)=rArrayDetectorVertices(1, :)./normSp-s0(1);
rArrayDetectorVertices(2, :)=rArrayDetectorVertices(2, :)./normSp-s0(2);
rArrayDetectorVertices(3, :)=rArrayDetectorVertices(3, :)./normSp-s0(3);
coordsArrayDetectorVertices=rArrayDetectorVertices;
surfacesInResiprocalSpace.detectorVertices = coordsArrayDetectorVertices;
surfacesInResiprocalSpace.evaldSphereCenter = -s0;
surfacesInResiprocalSpace.evaldSphereTopPoint = -E0E1E2(:,1)-s0;
solidAngle=(detector.d^3)./sqrt(sum(coordsArray.^2,1)).^3;
solidAngle=reshape(solidAngle,[fd,sd]);
detectorPerimeterCoordinates=coordsArray(:, [1:fd (fd*(1:(sd-2))+1) ((2:(sd-1))*fd) ((end-fd+1):end)]);
normSp=sqrt(sum(detectorPerimeterCoordinates.^2,1));
rArrayDetectorPerimeter = (eDetector*detectorPerimeterCoordinates);
rArrayDetectorPerimeter(1, :)=rArrayDetectorPerimeter(1, :)./normSp-s0(1);
rArrayDetectorPerimeter(2, :)=rArrayDetectorPerimeter(2, :)./normSp-s0(2);
rArrayDetectorPerimeter(3, :)=rArrayDetectorPerimeter(3, :)./normSp-s0(3);
detectorPerimeterCoordinates=rArrayDetectorPerimeter;
end
function [I, cbfHead, isOk] = bqdopOyfbsiQElql( cbfFileName, varargin )
use_java_decompressor = 1;I=[];cbfHead=[];isOk=false;javaclassexists = false;
if use_java_decompressor
if exist('com.company.Cbf', 'class')==8
javaclassexists = true;else
func_folder=fileparts(mfilename('fullpath'));
javafilename = fullfile(func_folder, 'formatscoder.jar');
if exist(javafilename, 'file')==2
javaaddpath(javafilename)
if exist('com.company.Cbf', 'class')==8
javaclassexists = true;
end
end
end
end
if ~javaclassexists
use_java_decompressor=false;
end
if nargin>1 && ischar(varargin{1}) && ...
(strcmpi(varargin{1}, 'minimal') || strcmpi(varargin{1}, 'min') || strcmpi(varargin{1}, 'm') )
useminimalheader = true;else
useminimalheader = false;
end
if isempty(cbfFileName)
warning('input parameter is an empty variable')
return;
end
if ~ischar(cbfFileName)
warning('input type isn''t correct')
return;
end
if exist(cbfFileName, 'file')~=2
warning(['file ''', cbfFileName, ''' doesn''t exist'])
return;
end
g = fopen(cbfFileName, 'r', 'l');
masInt8 = fread(g,  '*int8');mas=double(masInt8);fclose(g);
lmas=length(mas);isBinExist=false;isTextExist=false;k=0;binIdentifierTale=[26;4;-43];
while k<(lmas-3)
k=k+1;
if mas(k)==12
if all(mas(k+1:k+3)==binIdentifierTale)
isBinExist=true;
if k>1
isTextExist=true;lastTextSymbolPosition=k-1;
end
k=k+3;break;
end
end
end
if ~isTextExist
warning('cann''''t find text header');return;
end
if ~isBinExist
warning('cann''''t find any binary identifier');return;
end
if useminimalheader
cbfHead=MqWgrlfw(mas(1:lastTextSymbolPosition));else
cbfHead=bUbw_QxICcj(mas(1:lastTextSymbolPosition));
end
fd=cbfHead.X_Binary_Size_Fastest_Dimension;sd=cbfHead.X_Binary_Size_Second_Dimension;
if isnan(fd) || isnan (sd) || isnan(cbfHead.X_Binary_Size) ||...
isempty(fd) || isempty (sd) || isempty(cbfHead.X_Binary_Size)
warning([filename ': corrupted file header'])
return
end
if use_java_decompressor
K = com.company.Cbf;
K.putcompressedstream(masInt8(k+1:end),fd, sd, cbfHead.X_Binary_Size);
isOkDecode = K.getstatus();
if ~isOkDecode
warning(K.getmessage()');return
end
I=reshape(K.getvalues,[fd, sd]);isOk = true;return
end
if double(fd*sd)*1.03<double(cbfHead.X_Binary_Size)
[I, isOk_decomp]  = pm6oDrQr(mas(k+1:end), cbfHead.X_Binary_Size, fd, sd);
isOk=isOk_decomp;return
end
x=find(mas==-128);lx=length(x);
if lx==0
I=reshape(cumsum((mas(k+1:k+sd*fd))), fd, sd);isOk=true;return
end
i=find(x>k, 1, 'first');
if isempty(i)
I=reshape(cumsum((mas(k+1:k+sd*fd))), fd, sd);isOk=true;return
end
i=i-1;xLogical=true(1,lmas);xLogical(x)=false;
xLogical(1:k)=false;int16_indexs=zeros(1,lx);int16_numbers=zeros(1,2*lx);
int16_ind=0;int32_indexs=zeros(1,lx);int32_numbers=zeros(1,4*lx);int32_ind=0;
while i<=(lx-1)
i=i+1;curIndex=x(i);
if curIndex-6*int32_ind-2*int16_ind-k>sd*fd
disp('Warning: Last Element read (bqdopOyfbsiQElql.m)')
break
else
end
if lx>=i+1 && (x(i+1)==curIndex+2 && mas(curIndex+1)==0)
int32_numbers(int32_ind*4+1)=( curIndex+3 );int32_numbers(int32_ind*4+2)=( curIndex+4 );
int32_numbers(int32_ind*4+3)=( curIndex+5 );int32_numbers(int32_ind*4+4)=( curIndex+6 );
int32_indexs(int32_ind+1)=x(i)-6*int32_ind-2*int16_ind-k;
int32_ind=int32_ind+1;xLogical(curIndex+1) = false;
xLogical(curIndex+2) = false;xLogical(curIndex+3) = false;xLogical(curIndex+4) = false;
xLogical(curIndex+5) = false;xLogical(curIndex+6) = true;i=i+1;curIndex=x(i);
for j=1:4
if lx>=i+1 && x(i+1)-curIndex<5
i=i+1;else
break
end
end
else
int16_numbers(int16_ind*2+1)=( curIndex+1 );int16_numbers(int16_ind*2+2)=( curIndex+2 );
int16_indexs(int16_ind+1)=x(i)-6*int32_ind-2*int16_ind-k;
int16_ind=int16_ind+1;xLogical(curIndex+1) = false;xLogical(curIndex+2) = true;
if lx>=i+1 && x(i+1)-curIndex<3
i=i+1;
end
if lx>=i+1 && x(i+1)-curIndex<3
i=i+1;
end
end
end
xLogical((k+fd*sd+2*int16_ind+6*int32_ind+1):end)=false;I_delta=mas(xLogical);
if int16_ind>0
int16_numbers=int16_numbers(1:int16_ind*2);int16indexs=int16_indexs(1:int16_ind);
I_2=double(typecast(int8(mas(int16_numbers)), 'int16'));I_delta(int16indexs)=I_2;
end
if int32_ind>0
int32_numbers=int32_numbers(1:int32_ind*4);int32indexs=int32_indexs(1:int32_ind);
I_4=double(typecast(int8(mas(int32_numbers)), 'int32'));I_delta(int32indexs)=I_4;
end
I=reshape(cumsum((I_delta)), fd, sd);isOk = true;
end
function  cbfHead  = bUbw_QxICcj( mas )
cellStr=regexp(char(mas'), ['(?:', sprintf('\r\n'), ')+'], 'split');
k=length(cellStr);fndA=cell(100,6);
for i=1:size(fndA, 1)
fndA{i,5}='%f %s';
end
n=1;fndA{n,1}='# Detector: ';fndA{n,6}='Detector';fndA{n,5}='%s';
n=n+2;fndA{n,1}='# Pixel_size';fndA{n,6}='Pixel_size';fndA{n,5}='%f m x %f %s';
n=n+1;fndA{n,1}='# Exposure_time';fndA{n,6}='Exposure_time';n=n+1;
fndA{n,1}='# Exposure_period';fndA{n,6}='Exposure_period';n=n+1;fndA{n,1}='# Wavelength';
fndA{n,6}='Wavelength';n=n+1;fndA{n,1}='# Flux';fndA{n,6}='Flux';n=n+1;
fndA{n,1}='X-Binary-Size-Fastest-Dimension:';fndA{n,6}='X_Binary_Size_Fastest_Dimension';
n=n+1;fndA{n,1}='X-Binary-Size-Second-Dimension:';
fndA{n,6}='X_Binary_Size_Second_Dimension';n=n+1;fndA{n,1}='# Start_angle';
fndA{n,6}='Start_angle';n=n+1;fndA{n,1}='# Angle_increment';fndA{n,6}='Angle_increment';
n=n+1;fndA{n,1}='# Omega ';fndA{n,6}='Omega';n=n+1;fndA{n,1}='# Omega_increment';
fndA{n,6}='Omega_increment';n=n+1;fndA{n,1}='# Phi ';fndA{n,6}='Phi';
n=n+1;fndA{n,1}='# Phi_increment';fndA{n,6}='Phi_increment';n=n+1;fndA{n,1}='# Kappa';
fndA{n,6}='Kappa';n=n+1;fndA{n,1}='# Oscillation_axis';fndA{n,6}='Oscillation_axis';
fndA{n,5}='%s';n=n+1;fndA{n,1}='# Temperature';fndA{n,6}='Temperature';
n=n+1;fndA{n,1}='# Blower';fndA{n,6}='Blower';n=n+1;fndA{n,1}='# Detector_distance';
fndA{n,6}='Detector_distance';n=n+1;fndA{n,1}='# Detector_Voffset';
fndA{n,6}='Detector_Voffset';n=n+1;fndA{n,1}='X-Binary-Size:';fndA{n,6}='X_Binary_Size';
n=n+1;fndA{n,1}='# Beam_xy';fndA{n,6}='Beam_xy';fndA{n,5}='(%f, %f) %s';
n=n+1;fndA{n,1}='Content-MD5:';fndA{n,6}='MD5';fndA{n,5}='%s';fndA(n+1:end,:)=[];
for i=1:n
fndA{i, 2}=length(fndA{i, 1});
if strncmp(fndA{i,5},'%s',2)
fndA{i, 3}='';else
fndA{i, 3}=nan;
end
end
for i=1:k
m=cellStr{i};fndA{1, 3}='';fndA{2, 3}='';
if strncmp(fndA{1, 1}, m, fndA{1, 2})
c=m(fndA{1, 2}+1:end);fndA{1, 3}=c;
if k>i
m=cellStr{i+1};
if length(m)==28 && strcmp(m(7),'-') && strcmp(m(10),'-')...
&& strcmp(m(13),'T') && strcmp(m(16),':') &&...
strcmp(m(19),':') && strcmp(m(22),'.')
c=m(3:end);fndA{2, 3}=c;else
strL=length(m);
if strL>14 && strcmp(m(end-3), '.')  &&...
strcmp(m(end-6), ':') && strcmp(m(end-9), ':') ...
&& strcmp(m(1:2),'# ')
c=m(3:end);fndA{2, 3}=c;
end
end
end
break
end
end
for i=1:k
m=cellStr{i};
for j=1:n
if strncmp(fndA{j, 1}, m, fndA{j, 2})
c=textscan(m(fndA{j, 2}+1:end), fndA{j, 5},'delimiter','','CollectOutput',1);
if ~isempty(c)
if ~iscell(c{1})
fndA{j, 3}=c{1};else
fndA{j, 3}=c{1}{1};
end
end
if numel(c)>1
fndA{j, 4}=c{2};
end
break;
end
end
end
fndA{2,6}='Data';cbfHead=cell2struct(fndA(:,3)',fndA(:,6)',2);
end
function  cbfHead  = MqWgrlfw( mas )
k = 0;
for k=1:numel(mas)
if mas(k)==88
if mas(k+1)==45 && mas(k+2)==66
break
end
end
end
M = mas(k:end)';l = numel(M);u=find(M==88);
d = 10.^(9:-1:0);b = [88 45 66 105 110 97 114 121 45 83 105 122 101 58 32];n = 15;
for i=1:numel(u)
if all(M(u(i):u(i)+n-1)==b)
for j=u(i)+n:l
if M(j)<48 || M(j)>57
break
end
end
X_Binary_Size = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));break
end
end
b = [88 45 66 105 110 97 114 121 45 83 105 122 101 45 70 97 115 116 101 115 116 45 68 105 109 101 110 115 105 111 110 58 32];
n = 33;
for i=1:numel(u)
if all(M(u(i):u(i)+n-1)==b)
for j=u(i)+n:l
if M(j)<48 || M(j)>57
break
end
end
X_Binary_Size_Fastest_Dimension = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));break
end
end
b = [88 45 66 105 110 97 114 121 45 83 105 122 101 45 83 101 99 111 110 100 45 68 105 109 101 110 115 105 111 110 58 32];
n = 32;
for i=1:numel(u)
if all(M(u(i):u(i)+n-1)==b)
for j=u(i)+n:l
if M(j)<48 || M(j)>57
break
end
end
X_Binary_Size_Second_Dimension = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));break
end
end
cbfHead = struct(...
'X_Binary_Size_Fastest_Dimension'   ,X_Binary_Size_Fastest_Dimension  ,...
'X_Binary_Size_Second_Dimension'    ,X_Binary_Size_Second_Dimension   ,...
'X_Binary_Size'                     ,X_Binary_Size);
end
function  [I]  = j0pkB0H0Vb(mas_Double, k, fd, sd)
I_delta=zeros(1,fd*sd);I_2=zeros(2e3, 1);I_4=zeros(4e3, 1);
I_2_i=zeros(1e3, 1);I_4_i=zeros(1e3, 1);k_int16=0;k_int32=0;k_int16_2=0;k_int32_4=0;
for i=1:sd*fd
k=k+1;
if mas_Double(k)==-128;k=k+1;a1=mas_Double(k);k=k+1;a2=mas_Double(k);
if a1==0 && a2==-128
k_int32=k_int32+1;k_int32_4=k_int32_4+4;k=k+1;a1=mas_Double(k);
k=k+1;a2=mas_Double(k);k=k+1;a3=mas_Double(k);k=k+1;a4=mas_Double(k);I_4(k_int32_4-3)=a1;
I_4(k_int32_4-2)=a2;I_4(k_int32_4-1)=a3;I_4(k_int32_4  )=a4;I_4_i(k_int32)=i;else
k_int16=k_int16+1;
k_int16_2=k_int16_2+2;I_2(k_int16_2-1)=a1;I_2(k_int16_2  )=a2;I_2_i(k_int16)=i;
end
else
I_delta(i)=mas_Double(k);
end
end
if k_int16>0
I_2=I_2(1:k_int16_2);I_2_i=I_2_i(1:k_int16);I_2=double(typecast(int8(I_2), 'int16'));
for i=1:k_int16
I_delta(I_2_i(i))=I_2(i);
end
end
if k_int32>0
I_4=I_4(1:k_int32_4);I_4_i=I_4_i(1:k_int32);I_4=double(typecast(int8(I_4), 'int32'));
for i=1:k_int32
I_delta(I_4_i(i))=I_4(i);
end
end
I=reshape(cumsum((I_delta)), fd, sd);isOk = 1;
end
function  [I, isok]  = pm6oDrQr(mas_Double, X_Binary_Size, fd, sd)
I = zeros(fd,sd);isok = false;
dat_in=typecast(int8(mas_Double), 'uint8');ind_out = 1;ind_in = 1;val_curr = 0;
while (ind_in <= X_Binary_Size)
val_diff = double(dat_in(ind_in));ind_in = ind_in +1;
if (val_diff ~= 128)
if (val_diff >= 128)
val_diff = val_diff - 256;
end
else
if ((dat_in(ind_in) ~= 0) || (dat_in(ind_in+1) ~= 128))
val_diff = double(dat_in(ind_in)) + ...
256 * double(dat_in(ind_in+1));
if (val_diff >= 32768)
val_diff = val_diff - 65536;
end
ind_in = ind_in +2;else
ind_in = ind_in +2;val_diff = double(dat_in(ind_in)) + ...
256 * double(dat_in(ind_in+1)) + ...
65536 * double(dat_in(ind_in+2)) + ...
16777216 * double(dat_in(ind_in+3));
if (val_diff >= 2147483648)
val_diff = val_diff - 4294967296;
end
ind_in = ind_in +4;
end
end
val_curr = val_curr + val_diff;I(ind_out) = val_curr;ind_out = ind_out +1;
end
if (ind_out-1 ~= fd*sd)
warning(filename,[ 'mismatch between ' num2str(ind_out-1) ...
' bytes after decompression with ' num2str(fd*sd) ...
' expected' ]);return
end
isok=true;
end
function [ beginStr, endStr, useAllFoldersNames ] = SN5XEBn7UuSH( str )
if isempty(str) || strcmp(str, '*')
beginStr='';
endStr='';useAllFoldersNames=true;return
end
asterixArray=strfind(str,'*');
if isempty(asterixArray)
beginStr=str;
endStr='';useAllFoldersNames=false;return
end
asterix1=asterixArray(1);asterix2=asterixArray(end);beginStr=str( 1:(asterix1-1) );
endStr=str( (asterix2+1):end );useAllFoldersNames=false;
end
function [cellArrayOfNames, listLength, list, isOk ] = a9SljmjRJu( folderPath, varargin )
isOk=false;list=[];listLength=[];cellArrayOfNames=[];
if ~isvector(folderPath) || ~ischar(folderPath) || isempty(folderPath)
warning('wrong first input parameter')
return
end
useMultiTemplates=false;useSeparateOutputCellArrays=false;
if nargin>=2
template=varargin{1};else
template=false;
end
str='*';
if ischar(template) && ~isempty(template) && isvector(template) && size(template, 1)==1
str=template;str=strtrim(str);
elseif iscell(template)
lng=numel(template);isInputCellArrayOk=true;
for k=1:lng
currentCell=template{k};
if isempty(currentCell) || ~ischar(currentCell) || ~isvector(currentCell) || size(currentCell, 1)~=1
isInputCellArrayOk=false;break
end
end
if isInputCellArrayOk
useMultiTemplates=true;cellArrayOftemplates=template;
if nargin>=3 && varargin{2}
useSeparateOutputCellArrays=true;
end
else
str='*';
end
else
str='*';
end
if exist(folderPath, 'dir')~=7
warning(['folder ''' folderPath ''' doesn''t exist'])
return
end
folderList=dir(folderPath);N = numel(folderList);goodindexes=true(1 ,N) ;
for k=numel(folderList):-1:1
if folderList(k).isdir==true && ( strcmp(folderList(k).name, '.') || strcmp(folderList(k).name, '..'))
goodindexes(k)=false;
end
end
folderList=folderList(goodindexes);folderListLng=length(folderList);
if ~useMultiTemplates
goodIndexes=zeros(1,folderListLng);
if strcmpi(str, 'dir')
for k=1:folderListLng
if folderList(k).isdir
goodIndexes(k)=1;
end
end
elseif strcmpi(str, 'all') || strcmpi(str, '*')
goodIndexes(:)=1;
elseif strcmpi(str, 'files') || strcmpi(str, 'allfiles') || strcmpi(str, 'all files') || strcmpi(str, '*.*') || strcmpi(str, '.*')
for k=1:folderListLng
if ~folderList(k).isdir
goodIndexes(k)=1;
end
end
elseif strcmpi(str, 'zero')
for k=1:folderListLng
[useless,useless,ext]=fileparts(folderList(k).name);
if isempty(ext)
goodIndexes(k)=1;
end
end
else
if strncmp(str, '*.', 2)
extention=str(2:end);
elseif strncmp(str, '.', 1)
extention=str;else
extention=['.' str];
end
for k=1:folderListLng
[useless,useless,ext]=fileparts(folderList(k).name);
if strcmpi(ext,extention)
goodIndexes(k)=1;
end
end
end
list=folderList(logical(goodIndexes));
listLength=length(list);cellArrayOfNames=cell(listLength, 1);
for k=1:listLength
cellArrayOfNames{k}=list(k).name;
end
else
lng=numel(cellArrayOftemplates);goodIndexesFull=zeros(lng,folderListLng);
for j=1:lng
str=cellArrayOftemplates{j};
if strncmp(str, '*.', 2)
extention=str(2:end);
elseif strncmp(str, '.', 1)
extention=str;else
extention=['.' str];
end
for k=1:folderListLng
[useless,useless,ext]=fileparts(folderList(k).name);
if strcmpi(ext,extention)
goodIndexesFull(j,k)=1;
end
end
end
if ~useSeparateOutputCellArrays
list=folderList(logical(any(goodIndexesFull,1)));
listLength=length(list);cellArrayOfNames=cell(listLength, 1);
for k=1:listLength
cellArrayOfNames{k}=list(k).name;
end
else
listBig=cell(1,lng);listLengthBig=zeros(1,lng);cellArrayOfNamesBig=cell(1,lng);
for j=1:lng
list=folderList(logical(goodIndexesFull(j,:)));
listLength=length(list);cellArrayOfNames=cell(listLength, 1);
for k=1:listLength
cellArrayOfNames{k}=list(k).name;
end
listBig{j}=list;listLengthBig(j)=listLength;cellArrayOfNamesBig{j}=cellArrayOfNames;
end
list=listBig;listLength=listLengthBig;cellArrayOfNames=cellArrayOfNamesBig;
end
end
isOk=true;
end
function [I, edfHead, isOk] = CZ09aDxngjTvV3V( edfFile, varargin )
I=[];edfHead=[];isOk = false;
if exist(edfFile, 'file')~=2
warning(['file ''', edfFile, ''' doesn''t exist'])
return;
end
outputType='double';
if ~isempty(varargin)
if ischar(varargin{1})
switch varargin{1}
case 'uint8'
outputType='UnsignedByte';case 'single'
outputType='single';case 'int32'
outputType='int32';case 'int16'
outputType='int16';case {'origin', 'original', 'default'}
outputType='origin';otherwise
outputType='double';
end
end
end
g = fopen(edfFile, 'r', 'l');masInt8 = fread(g,  '*int8');
mas=double(masInt8(1:1e4));fclose(g);lmas=length(mas);k=0;isCurlyBraceLeft=0;
while k<(lmas)
k=k+1;
if mas(k)==123
curlyBraceLeft=k;isCurlyBraceLeft=1;break;
end
end
if isCurlyBraceLeft==0;
warning(['file ''', edfFile, ''' doesn''t contain header identifier ''{'''])
return;
end
isCurlyBraceRight=0;
while k<(lmas)
k=k+1;
if mas(k)==125
curlyBraceRight=k;isCurlyBraceRight=1;break;
end
end
if isCurlyBraceRight==0;
warning(['file ''', edfFile, ''' doesn''t contain any binary identifier ''}'''])
return;
end
masHead=mas( (curlyBraceLeft+1):(curlyBraceRight-1) );
edfHead=pDZ8aUo2xhuuc1f(masHead);fd=edfHead.Dim_1;sd=edfHead.Dim_2;
if strcmpi(edfHead.DataType, 'UnsignedByte')
numOfBytesForPixel=1;dataInputType='uint8';
elseif strcmpi(edfHead.DataType, 'UnsignedShort')
numOfBytesForPixel=2;dataInputType='uint16';
elseif any(strcmpi(edfHead.DataType, {'SignedInteger', 'SignedLong'}))
numOfBytesForPixel=4;dataInputType='int32';
elseif any(strcmpi(edfHead.DataType, 'UnsignedInteger'))
numOfBytesForPixel=4;dataInputType='uint32';else
warning(['unknown data type in file ''' edfFile ''''])
return
end
if (~isfinite(edfHead.Dim_1) || ~isfinite(edfHead.Dim_2)) || (edfHead.Dim_1==0 || edfHead.Dim_2==0);
warning(['dimentions in the header of the file ''', edfFile, ''' are wrong ''}'''])
return;
end
if lmas<curlyBraceRight+1 || mas(curlyBraceRight+1)~=10
warning(['cann''t find ''LF'' symbol after end of the header ''}'' in the file ''', edfFile ''''])
return;
end
firstSymbolOfTheBinaryPart = curlyBraceRight+2;
lmasInt8 = length(masInt8);lengthOfTheBinaryPart = lmasInt8-(curlyBraceRight+1);
if lengthOfTheBinaryPart~=fd*sd*numOfBytesForPixel
warning(['WARNING: length of the binary part in the file ''', edfFile, 'doesn''t correspond to it''s resolution'])
return;
end
switch outputType
case 'UnsignedByte'
I=reshape(uint8(typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType)), fd, sd);
case 'single'
I=reshape(single(typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType)), fd, sd);
case 'int32'
I=reshape(int32((typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType))), fd, sd);
case 'int16'
I=reshape(int16(typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType)), fd, sd);
case 'double'
I=reshape(double(typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType)), fd, sd);
case 'origin'
I=reshape(typecast(masInt8(firstSymbolOfTheBinaryPart:end), dataInputType), fd, sd);
end
isOk = 1;
end
function  edfHead  = pDZ8aUo2xhuuc1f( mas )
cellStr=regexp(char(mas'), ['(?:', sprintf('\r\n|\n'), ')+'], 'split');
numOfStr=numel(cellStr);badStringsIndexes=false(1, numOfStr);
for s=1:numOfStr
if isempty(cellStr{s}) || numel(cellStr{s})<3
badStringsIndexes(s)=true;else
if  ~strcmp(cellStr{s}(end),';')
badStringsIndexes(s)=true;else
cellStr{s}(end)=[];
end
end
end
cellStr(badStringsIndexes)=[];numOfStr=numel(cellStr);fndA=cell(100,6);
for i=1:size(fndA, 1)
fndA{i,5}='%d';
end
n=1;fndA{n,1}='HeaderID';fndA{n,6}='HeaderID';fndA{n,5}='%s';n=n+1;fndA{n,1}='ByteOrder';
fndA{n,6}='ByteOrder';fndA{n,5}='%s';n=n+1;fndA{n,1}='DataType';fndA{n,6}='DataType';
fndA{n,5}='%s';n=n+1;fndA{n,1}='Dim_1';fndA{n,6}='Dim_1';n=n+1;fndA{n,1}='Dim_2';
fndA{n,6}='Dim_2';n=n+1;fndA{n,1}='Size';fndA{n,6}='Size';n=n+1;fndA{n,1}='time_of_day';
fndA{n,6}='time_of_day';fndA{n,5}='%s';n=n+1;fndA{n,1}='time_of_frame';
fndA{n,6}='time_of_frame';fndA{n,5}='%s';n=n+1;fndA{n,1}='count_time';
fndA{n,6}='count_time';fndA{n,5}='%f';n=n+1;fndA{n,1}='point_no';fndA{n,6}='point_no';
n=n+1;fndA{n,1}='scan_no';fndA{n,6}='scan_no';n=n+1;fndA{n,1}='preset';
fndA{n,6}='preset';n=n+1;fndA{n,1}='counter_pos';fndA{n,6}='counter_pos';
fndA{n,5}='%s';n=n+1;fndA{n,1}='counter_mne';fndA{n,6}='counter_mne';
fndA{n,5}='%s';n=n+1;fndA{n,1}='motor_pos';fndA{n,6}='motor_pos';fndA{n,5}='%s';
n=n+1;fndA{n,1}='motor_mne';fndA{n,6}='motor_mne';fndA{n,5}='%s';n=n+1;fndA{n,1}='dir';
fndA{n,6}='dir';fndA{n,5}='%s';n=n+1;fndA{n,1}='run';fndA{n,6}='run';fndA(n+1:end,:)=[];
for i=1:n
fndA{i, 2}=length(fndA{i, 1});
if strncmp(fndA{i,5},'%s',2)
fndA{i, 3}='';else
fndA{i, 3}=nan;
end
end
for i=1:numOfStr
m=cellStr{i};
for j=1:n
if strncmp(fndA{j, 1}, m, fndA{j, 2})
c=textscan(m(fndA{j, 2}+1:end), ['=' fndA{j, 5}],'delimiter','','CollectOutput',1);
if ~isempty(c)
if ~iscell(c{1})
fndA{j, 3}=c{1};else
if ischar(c{1}{1})
fndA{j, 3}=strtrim(c{1}{1});else
fndA{j, 3}=c{1}{1};
end
end
end
break;
end
end
end
edfHead=cell2struct(fndA(:,3)',fndA(:,6)',2);
if isfield(edfHead, 'counter_pos') && isfield(edfHead, 'counter_mne') &&...
~isempty(edfHead.counter_pos) && ~isempty(edfHead.counter_mne)
counter_pos=str2num(edfHead.counter_pos);
counter_mne=regexp(edfHead.counter_mne, ['(?:', ' ', ')+'], 'split');
if length(counter_pos)==length(counter_mne) &&...
length(counter_mne)>1 && ~isempty(counter_mne{1})
edfHead.countersStruct=cell2struct(num2cell(counter_pos), counter_mne, 2);
end
end
if isfield(edfHead, 'motor_pos') && isfield(edfHead, 'motor_mne') &&...
~isempty(edfHead.motor_pos) && ~isempty(edfHead.motor_mne)
motor_pos=str2num(edfHead.motor_pos);
motor_mne=regexp(edfHead.motor_mne, ['(?:', ' ', ')+'], 'split');
if length(motor_pos)==length(motor_mne) &&...
length(motor_mne)>1 && ~isempty(motor_mne{1})
edfHead.motorsStruct=cell2struct(num2cell(motor_pos), motor_mne, 2);
end
end
end
function [ nameCoreString, isOk ] = xIOIhDd9q2fSL8( varargin )
nameCoreString='';isOk=false;
if nargin<1
disp(['ERROR (' mfilename '): there should be at least one input argument'])
return
end
if ~isstruct(varargin{1})
disp(['ERROR (' mfilename '): first input argument should be a struct'])
return
end
R=varargin{1};
if nargin==2 && iscell(varargin{2})
keys = varargin{2};else
keys = varargin(2:end);
end
numOfBlocks = numel(keys);
if numOfBlocks>0
for i=1:numOfBlocks
if ~ischar(keys{i})
disp(['ERROR (' mfilename '): all input arguments accept the first and the last should be strings'])
return
end
end
else
switch R.chosenVolume.shape
case {'parallelepiped', 'simple parallelepiped'}
keys={'value', 'center', 'edges', 'norm', 'binning', '3D'};case 'cylinder'
keys={'center', 'direction', 'length', 'bottom point', 'top point', 'radius'};
end
end
blocksStrCellOut=cell(1, numOfBlocks);
for i=1:numOfBlocks
[blocksStrCellOut{i}, isOkStrPart]=nqLSDCRKq4WGxdrr( R, keys{i} );
if ~isOkStrPart
return
end
end
for i=numOfBlocks:-1:1
if isempty(blocksStrCellOut{i})
blocksStrCellOut(i)=[];
end
end
nameCoreString=strjoin(blocksStrCellOut, ', ');isOk=true;
end
function [ namePart, isOk ] = nqLSDCRKq4WGxdrr( recVarPack ,stringIn )
namePart='';isOk=false;R = recVarPack;
V = R.valueFromMultiDataset;chosenVolume=R.chosenVolume;shape=chosenVolume.shape;
if strcmp(shape, 'parallelepiped')
parallelepiped=chosenVolume.parallelepiped;
elseif strcmp(shape, 'cylinder')
cylinder=chosenVolume.cylinder;useTopBottomPoints=cylinder.useTopBottomPoints;
end
switch stringIn
case {'index', 'nindex', '¹index'}
namePart=[ num2str(...
V.currentIndex, ['%0.' num2str(V.indexDigits) 'd'])];
if strcmp(stringIn, 'nindex')
namePart = ['N' namePart];
elseif strcmp(stringIn, '¹index')
namePart = ['¹' namePart];
end
case {'index?', 'nindex?', '¹index?'}
if V.typeIndexInName
namePart=num2str(...
V.currentIndex, ['%0.' num2str(V.indexDigits) 'd']);
if strcmp(stringIn, 'nindex')
namePart = ['N' namePart];
elseif strcmp(stringIn, '¹index')
namePart = ['¹' namePart];
end
end
case 'value'
if ~isfield(V, 'multiDatasetTypeShort')
V.multiDatasetTypeShort=upper(V.multiDatasetType(1));
end
var    = V.multiDatasetTypeShort;
val    = V.currentValue;dig    = V.digits;units_ = V.units;
if val==round(val)
namePart=[var '=' num2str( val, ['%0.' num2str(dig) 'i']), units_];else
val_ = round(val*1e5)/1e5;
valr = floor(val_);valo = val_-valr;nmpart2 = num2str(valo, '%0.5g');nmpart2(1)=[];
namePart=[var '=' num2str( valr, ['%0.' num2str(dig) 'i']), nmpart2, units_];
end
case 'valueshort'
if ~isfield(V, 'multiDatasetTypeShort')
V.multiDatasetTypeShort=upper(V.multiDatasetType(1));
end
val    = V.currentValue;dig    = V.digits;units_ = V.units;
if val==round(val)
namePart=[num2str( val, ['%0.' num2str(dig) 'i']), units_];else
val_ = round(val*1e5)/1e5;
valr = floor(val_);valo = val_-valr;nmpart2 = num2str(valo, '%0.5g');
nmpart2(1)=[];namePart=[num2str( valr, ['%0.' num2str(dig) 'i']), nmpart2, units_];
end
case {'center', 'center2', 'center3' }
digits = BjXadTOHI7fvT(stringIn);switch shape
case 'simple parallelepiped'
namePart = ['cen=[' VnhMgvaur8TVY( R.chosenVolume.peak, digits ) ']'];
case 'parallelepiped'
namePart = ['cen=[' VnhMgvaur8TVY( R.peak, digits ) ']'];case 'cylinder'
if useTopBottomPoints
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''center'''])
namePart='';else
namePart = ['cen=[' VnhMgvaur8TVY( cylinder.center, digits ) ']'];
end
otherwise
disp(['ERROR (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''center'''])
return
end
case 'length'
if strcmp(shape, 'cylinder') && ~useTopBottomPoints
namePart=['length=[' num2str(cylinder.length, ' %1.2f') ']'];else
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''length'''])
namePart='';
end
case 'direction'
if strcmp(shape, 'cylinder') && ~useTopBottomPoints
namePart = ['direction=[' VnhMgvaur8TVY( cylinder.direction, [] ) ']'];else
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''direction'''])
namePart='';
end
case 'radius'
if strcmp(shape, 'cylinder')
namePart=['radius=[' num2str(cylinder.radius, ' %1.2f') ']'];else
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''radius'''])
namePart='';
end
case 'bottom point'
if strcmp(shape, 'cylinder') && useTopBottomPoints
namePart=['p1=[' num2str(cylinder.bottomPoint, ' %1.2f') ']'];else
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''bottom point'''])
namePart='';
end
case 'top point'
if strcmp(shape, 'cylinder') && useTopBottomPoints
namePart=['p2=[' num2str(cylinder.topPoint, ' %1.2f') ']'];else
disp(['WARNING (' mfilename '): first input parameter ''R.chosenVolume'' is wrong - no field ''top point'''])
namePart='';
end
case {'edges', 'edges2', 'edges3'}
digits = BjXadTOHI7fvT(stringIn);switch R.chosenVolume.shape
case 'simple parallelepiped'
delta_edges = R.chosenVolume.delta;
if delta_edges(2)~=delta_edges(1) || delta_edges(3)~=delta_edges(1)
namePart = ['edges=[' VnhMgvaur8TVY( delta_edges, digits ) ']'];else
namePart = ['edges=[' VnhMgvaur8TVY( delta_edges(1), digits ) ']'];
end
case 'parallelepiped'
namePart=['edges=[' num2str(R.delta, ' %1.2f') ']'];otherwise
disp(['WARNING (' mfilename '): first input parameter ''R'' or ''R.chosenVolume'' is wrong - no field ''delta'''])
namePart='';
end
case 'norm'
namePart=['norm=' num2str(R.normalization.useNormalizationOn)];case 'binning'
pixelFragmentation=R.pixelFragmentation;
if pixelFragmentation.usePixelFragmentation==false
pixelFragmentation.pixelFragmentationX=1;pixelFragmentation.pixelFragmentationY=1;
pixelFragmentation.pixelFragmentationAngle=1;namePart=['bin=[1]'];else
namePart=['bin=[' num2str(R.pixelFragmentation.pixelFragmentationX) ...
num2str(R.pixelFragmentation.pixelFragmentationY)...
num2str(R.pixelFragmentation.pixelFragmentationAngle) ']'];
end
case '3D'
if R.make3DArray.makeReconstructionImmediately==true
namePart=['3D=[' num2str(R.make3DArray.nh) ' '...
num2str(R.make3DArray.nk) ' ' num2str(R.make3DArray.nl) ']'];else
namePart='';
end
case {'hkl', 'HKL'}
hkl = stringIn;
peak = R.peak;delta = R.delta;ind = find(delta==min(delta), 1, 'first');namePart='';
for k=1:3
if k==ind
namePart = [namePart num2str(peak(ind))];else
namePart = [namePart hkl(k)];
end
end
case {'name'}
namePart = R.experimentNames.experimentIdentifierString;otherwise
if numel(stringIn)>2 && stringIn(1)=='"' && stringIn(end)=='"'
namePart = stringIn(2:end-1);else
disp(['ERROR (' mfilename '): bad input parameter - ' stringIn])
return
end
end
isOk=true;
end
function [ string_ ] = VnhMgvaur8TVY( array, digits )
string_ = '';
if isempty(digits)
d = 0;
end
for i=1:numel(array)
if i>1
string_ = [string_ ' '];
end
num = array(i);
if num==fix(num)
string_ = [string_ int2str(num)];else
if isempty(digits)
if round(num*10)==num*10
d = 1;
elseif round(num*100)==num*100
d = 2;else
d = 3;
end
else
d = digits;
end
string_ = [string_ num2str(num, ['%1.' int2str(d) 'f'])];
end
end
end
function [ digits ] = BjXadTOHI7fvT( string_ )
digits = str2double(string_(end));
if ~isfinite(digits)
digits = [];
end
end
function [pixels_2D, header, isOk] = AWo1Glkgc8Ljn( esperantoFileName )
use_only_java = 0;use_java_agi_bitfield_decompressor = 1;
use_only_matlab = 0;pixels_2D=[];header=[];isOk=false;javaclassexists = false;
if use_only_java || use_java_agi_bitfield_decompressor
if exist('com.company.Esperanto', 'class')==8
javaclassexists = true;else
func_folder=fileparts(mfilename('fullpath'));
javafilename = fullfile(func_folder, 'formatscoder.jar');
if exist(javafilename, 'file')==2
javaaddpath(javafilename)
if exist('com.company.Esperanto', 'class')==8
javaclassexists = true;
end
end
end
end
if ~javaclassexists
use_only_java=false;use_java_agi_bitfield_decompressor=false;
end
if use_only_java
[pixels_2D, header, isOk] = qpcyWBN(esperantoFileName);return
end
if isempty(esperantoFileName)
warning('input variable is empty')
return;
end
if ~ischar(esperantoFileName)
warning('input type is incorrect')
return;
end
if exist(esperantoFileName, 'file')~=2
warning(['file ''', esperantoFileName, ''' doesn''t exist'])
return;
end
g = fopen(esperantoFileName, 'r', 'l');
file_stream_uint8 = fread(g,  '*uint8')';fclose(g);file_length=length(file_stream_uint8);
if file_length<256*25
warning(['file size of the ''', esperantoFileName, ''' is too small'])
return;
end
header_stream_uint8=file_stream_uint8(1:256*25);
[header, isOkHeader]  = U0JqvAIB( header_stream_uint8, esperantoFileName );
if ~isOkHeader
return
end
indexOfTheBinaryPartStarts=256*25+1;
sizeOfTheBinaryPart=file_length-256*25;switch header.compression_method
case 'AGI_BITFIELD'
if use_java_agi_bitfield_decompressor
K = com.company.Esperanto;
K.putcompressedstream(typecast(file_stream_uint8,'int8'), header.dimension_2, header.dimension_1);
isOkDecode = K.getstatus();
if ~isOkDecode
warning(K.getmessage()');return
end
pixels_2D=reshape(K.getvalues,[header.dimension_1, header.dimension_2]);else
isOkDecode=YojpHqq60L();
end
case '4BYTE_LONG'
isOkDecode=B0YbFk2NFLqun();otherwise
warning(['unknown compression method (' header.compression_method ') in the file ''' esperantoFileName ''''])
return
end
if ~isOkDecode
return
end
isOk=true;return
function isOk=B0YbFk2NFLqun()
isOk=false;sizeX=header.dimension_1;sizeY=header.dimension_2;
if sizeOfTheBinaryPart>4*sizeX*sizeY;
warning(['binary part in the file ''', esperantoFileName, ''' is too big'])
return;
elseif sizeOfTheBinaryPart<4*sizeX*sizeY;
warning(['binary part in the file ''', esperantoFileName, ''' is too small'])
return;
end
pixels_2D=reshape(double(typecast((file_stream_uint8(indexOfTheBinaryPartStarts:end)), 'int32')), sizeX, sizeY);
isOk=true;
end
function isOkDecode=YojpHqq60L()
isOkDecode=false;dim_1=header.dimension_1;
dim_2=header.dimension_1;numOfRows=uint32(dim_2);numOfColumns=uint32(dim_1);
numOfFields=numOfColumns/16;pixels=zeros(1, numOfRows*numOfRows);
datasize=typecast(file_stream_uint8(indexOfTheBinaryPartStarts:indexOfTheBinaryPartStarts+3),'uint32');
data_stream_start_index=uint32(indexOfTheBinaryPartStarts+4);
file_stream_db=double(file_stream_uint8);
adressesstream_uint8=file_stream_uint8(data_stream_start_index+datasize:(data_stream_start_index-1)+datasize+4*numOfRows);
adressesstream_uint32=typecast(adressesstream_uint8, 'uint32');
for row_index=1:numOfRows
ind=adressesstream_uint32(row_index)+(data_stream_start_index-1);
pxl=uint32(numOfColumns*(row_index-1));xNASRkYL();field2_lng=uint8(0);e=false;ind2=ind;
for i=1:(numOfFields-1)*2
field_lng=field2_lng;e=~e;
if e
ind=ind2+1;fieldheader=file_stream_uint8(ind);
field1_lng=mod(fieldheader, 16);field2_lng=(fieldheader-field1_lng) / 2^4;
field_lng=field1_lng;ind2=ind+uint32(field2_lng)+uint32(field1_lng);
end
if field_lng==8
val=file_stream_db(ind+1);
if val<254
pixels(pxl+1)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+1)=val;
end
val=file_stream_db(ind+2);
if val<254
pixels(pxl+2)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+2)=val;
end
val=file_stream_db(ind+3);
if val<254
pixels(pxl+3)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+3)=val;
end
val=file_stream_db(ind+4);
if val<254
pixels(pxl+4)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+4)=val;
end
val=file_stream_db(ind+5);
if val<254
pixels(pxl+5)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+5)=val;
end
val=file_stream_db(ind+6);
if val<254
pixels(pxl+6)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+6)=val;
end
val=file_stream_db(ind+7);
if val<254
pixels(pxl+7)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+7)=val;
end
val=file_stream_db(ind+8);
if val<254
pixels(pxl+8)=val-127;else
if val==254
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256;
if val>=32768
val=val-65536;
end
ind2=ind2+2;else
val=file_stream_db(ind2+1)+...
file_stream_db(ind2+2)*256+...
file_stream_db(ind2+3)*65536+...
file_stream_db(ind2+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind2=ind2+4;
end
pixels(pxl+8)=val;
end
ind=ind+8;pxl=pxl+8;
end
if field_lng==1
ind=ind+1;t1=mod(file_stream_uint8(ind), 2);t2=mod(file_stream_uint8(ind), 4);
t3=mod(file_stream_uint8(ind), 8);t4=mod(file_stream_uint8(ind), 16);
t5=mod(file_stream_uint8(ind), 32);t6=mod(file_stream_uint8(ind), 64);
t7=mod(file_stream_uint8(ind), 128);pixels(pxl+1)=double(t1);
pixels(pxl+2)=double((t2-t1)/02);pixels(pxl+3)=double((t3-t2)/04);
pixels(pxl+4)=double((t4-t3)/08);pixels(pxl+5)=double((t5-t4)/16);
pixels(pxl+6)=double((t6-t5)/32);pixels(pxl+7)=double((t7-t6)/64);
pixels(pxl+8)=double((file_stream_uint8(ind)-t7)/128);pxl=pxl+8;
elseif field_lng==2
ind=ind+1;t1=mod(file_stream_uint8(ind), 4);pixels(pxl+1)=double(t1) -1;
t2=mod(file_stream_uint8(ind), 16);pixels(pxl+2)=double((t2-t1)/04) -1;
t3=mod(file_stream_uint8(ind), 64);pixels(pxl+3)=(double(t3-t2)/16) -1;
pixels(pxl+4)=double((file_stream_uint8(ind)-t3)/64) -1;
ind=ind+1;t4=mod(file_stream_uint8(ind), 4);pixels(pxl+5)=double(t4) -1;
t5=mod(file_stream_uint8(ind), 16);pixels(pxl+6)=double((t5-t4)/04) -1;
t6=mod(file_stream_uint8(ind), 64);pixels(pxl+7)=double((t6-t5)/16) -1;
pixels(pxl+8)=double((file_stream_uint8(ind)-t6)/64) -1;pxl=pxl+8;
elseif field_lng==3
t1=mod(file_stream_uint8(ind+1),8);
pixels(pxl+1)=double(t1)-3;t2=mod(file_stream_uint8(ind+1),64);
pixels(pxl+2)=double((t2-t1)/8)-3;t3=mod(file_stream_uint8(ind+2), 2);
pixels(pxl+3)=double(t3*04+(file_stream_uint8(ind+1)-t2)/64)-3;
t4=mod(file_stream_uint8(ind+2), 16);
pixels(pxl+4)=double((t4-t3)/2)-3;t5=mod(file_stream_uint8(ind+2), 128);
pixels(pxl+5)=double((t5-t4)/16)-3;t6=mod(file_stream_uint8(ind+3), 04);
pixels(pxl+6)=double(t6*2+(file_stream_uint8(ind+2)-t5)/128)-3;
t7=mod(file_stream_uint8(ind+3), 32);pixels(pxl+7)=double((t7-t6)/4)-3;
pixels(pxl+8)=double((file_stream_uint8(ind+3)-t7)/32)-3;ind=ind+3;pxl=pxl+8;
elseif field_lng==4
t1=mod(file_stream_uint8(ind+1), 16);
pixels(pxl+1)=double(t1) -7;pixels(pxl+2)=double((file_stream_uint8(ind+1)-t1)/16) -7;
t2=mod(file_stream_uint8(ind+2), 16);
pixels(pxl+3)=double(t2) -7;pixels(pxl+4)=double((file_stream_uint8(ind+2)-t2)/16) -7;
t3=mod(file_stream_uint8(ind+3), 16);
pixels(pxl+5)=double(t3) -7;pixels(pxl+6)=double((file_stream_uint8(ind+3)-t3)/16) -7;
t4=mod(file_stream_uint8(ind+4), 16);pixels(pxl+7)=double(t4) -7;
pixels(pxl+8)=double((file_stream_uint8(ind+4)-t4)/16) -7;ind=ind+4;pxl=pxl+8;
elseif field_lng==5
t1=mod(file_stream_uint8(ind+1),32);
pixels(pxl+1)=double(t1)-15;t2=mod(file_stream_uint8(ind+2),4);
pixels(pxl+2)=double(t2*8+(file_stream_uint8(ind+1)-t1)/32)-15;
t3=mod(file_stream_uint8(ind+2), 128);
pixels(pxl+3)=double((t3-t2)/4)-15;t4=mod(file_stream_uint8(ind+3), 16);
pixels(pxl+4)= double(t4*2 +  (file_stream_uint8(ind+2)-t3)/128)-15;
t5=mod(file_stream_uint8(ind+4),2);
pixels(pxl+5)=double(t5*16+(file_stream_uint8(ind+3)-t4)/16)-15;
t6=mod(file_stream_uint8(ind+4),64);
pixels(pxl+6)=double((t6-t5)/2)-15;t7=mod(file_stream_uint8(ind+5),8);
pixels(pxl+7)=double(t7*4+(file_stream_uint8(ind+4)-t6)/64)-15;
pixels(pxl+8)=double((file_stream_uint8(ind+5)-t7)/8)-15;ind=ind+5;pxl=pxl+8;
elseif field_lng==6
t1=mod(file_stream_uint8(ind+1),64);
pixels(pxl+1)=double(t1)-31;t2=mod(file_stream_uint8(ind+2),16);
pixels(pxl+2)=double(t2*4+(file_stream_uint8(ind+1)-t1)/64)-31;
t3=mod(file_stream_uint8(ind+3), 4);
pixels(pxl+3)=double(t3*16+(file_stream_uint8(ind+2)-t2)/16)-31;
pixels(pxl+4)= double((file_stream_uint8(ind+3)-t3)/4)-31;
t4=mod(file_stream_uint8(ind+4),64);
pixels(pxl+5)=double(t4)-31;t5=mod(file_stream_uint8(ind+5),16);
pixels(pxl+6)=double(t5*4+(file_stream_uint8(ind+4)-t4)/64)-31;
t6=mod(file_stream_uint8(ind+6), 4);
pixels(pxl+7)=double(t6*16+(file_stream_uint8(ind+5)-t5)/16)-31;
pixels(pxl+8)= double((file_stream_uint8(ind+6)-t6)/4)-31;ind=ind+6;pxl=pxl+8;
elseif field_lng==7
t1=mod(file_stream_uint8(ind+1),128);
pixels(pxl+1)=double(t1)-63;t2=mod(file_stream_uint8(ind+2),64);
pixels(pxl+2)=double(t2*02+(file_stream_uint8(ind+1)-t1)/128)-63;
t3=mod(file_stream_uint8(ind+3), 32);
pixels(pxl+3)=double(t3*04+(file_stream_uint8(ind+2)-t2)/64)-63;
t4=mod(file_stream_uint8(ind+4), 16);
pixels(pxl+4)=double(t4*08+(file_stream_uint8(ind+3)-t3)/32)-63;
t5=mod(file_stream_uint8(ind+5), 08);
pixels(pxl+5)=double(t5*16+(file_stream_uint8(ind+4)-t4)/16)-63;
t6=mod(file_stream_uint8(ind+6), 04);
pixels(pxl+6)=double(t6*32+(file_stream_uint8(ind+5)-t5)/08)-63;
t7=mod(file_stream_uint8(ind+7), 02);
pixels(pxl+7)=double(t7*64+(file_stream_uint8(ind+6)-t6)/04)-63;
pixels(pxl+8)=double((file_stream_uint8(ind+7)-t7)/02)-63;ind=ind+7;pxl=pxl+8;
end
end
ind=ind2;
for pxl=pxl+1:pxl+15
ind=ind+1;val=file_stream_db(ind)-127;
if val==127
val=file_stream_db(ind+1)+...
file_stream_db(ind+2)*256;
if val>=32768
val=val-65536;
end
ind=ind+2;
elseif val==128
val=file_stream_db(ind+1)+...
file_stream_db(ind+2)*256+...
file_stream_db(ind+3)*65536+...
file_stream_db(ind+4)*16777216;
if val>=2147483648
val=val-4294967296;
end
ind=ind+4;
end
pixels(pxl)=val;
end
end
pixels_2D=cumsum(reshape(pixels,numOfColumns, numOfRows));
isOkDecode=true;function [] = xNASRkYL()
ind=ind+1;val1=file_stream_db(ind)-127;
if val1==127
val1=file_stream_db(ind+1)+...
file_stream_db(ind+2)*256;
if val1>=32768
val1=val1-65536;
end
ind=ind+2;
elseif val1==128
val1=file_stream_db(ind+1)+...
file_stream_db(ind+2)*256+...
file_stream_db(ind+3)*65536+...
file_stream_db(ind+4)*16777216;
if val1>=2147483648
val1=val1-4294967296;
end
ind=ind+4;
end
pxl=pxl+1;pixels(pxl)=val1;
end
end
end
function  [esperantoHeadOut, isOkHeader]  = U0JqvAIB( header_stream_uint8, filename )
esperantoHeadOut=[];isOkHeader=false;header_stream_char=char(header_stream_uint8);
if ~strcmp(header_stream_char(1:65), ...
'ESPERANTO FORMAT   1 CONSISTING OF   25 LINES OF   256 BYTES EACH')
warning(['unknown header in the file ''' filename ''''])
return
end
header_stream_char_2D=reshape(header_stream_char, 256,25)';header=struct(...
'dimension_1', nan,...
'dimension_2', nan,...
'binningX', nan,...
'binningY', nan,...
'compression_method', '',...
'time', [0 0 0],...
'monitor', [0 0 0 0],...
'pixelsize', [0 0],...
'timestamp', '',...
'gridpattern', '',...
'startanglesindeg', [0 0 0 0],...
'endanglesindeg', [0 0 0 0],...
'goniomodel_1', [0 0 0 0 0 0 0 0 0 0],...
'goniomodel_2', [0 0 0 0],...
'wavelength', [0 0 0 0],...
'monochromator', '',...
'special_ccd_1', [0 0 0 0 0 0],...
'special_ccd_2', [0 0 0 0 0],...
'special_ccd_3', [0 0 0 0 0],...
'special_ccd_4', [0 0 0 0 0 0 0 0],...
'special_ccd_5', [0 0 0 0],...
'history', '');ind=1;ind=ind+1;string_=header_stream_char_2D(ind,:);
[A, count, errmsg, nextindex] = sscanf(string_, 'IMAGE %d%d%d%d', 4);
if ~isempty(errmsg) || count~=4
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.dimension_1=A(1);
header.dimension_2=A(2);header.binningX=A(3);header.binningY=A(4);
compressionMethod=strtrim(string_(nextindex:end));switch compressionMethod
case '"AGI_BITFIELD"'
compressionMethod='AGI_BITFIELD';case '"4BYTE_LONG"'
compressionMethod='4BYTE_LONG';otherwise
end
header.compression_method=compressionMethod;
ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'SPECIAL_CCD_1 ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(15:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.special_ccd_1=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'SPECIAL_CCD_2 ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(15:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.special_ccd_2=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'SPECIAL_CCD_3 ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(15:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.special_ccd_3=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'SPECIAL_CCD_4 ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(15:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.special_ccd_4=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'SPECIAL_CCD_5 ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(15:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.special_ccd_5=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'TIME ', 5)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(6:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.time=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'MONITOR ', 8)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(9:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.monitor=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'PIXELSIZE ', 10)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(11:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.pixelsize=A';ind=ind+1;
string_=header_stream_char_2D(ind,:);string_=deblank(string_);string_is_ok=false;
if strncmp(string_, 'TIMESTAMP "', 11)
string_=string_(12:end);
if string_(end)=='"'
string_(end)=[];string_is_ok=true;
end
end
if ~string_is_ok
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.timestamp=string_;ind=ind+1;string_=deblank(header_stream_char_2D(ind,:));
if ~strncmp(string_, 'GRIDPATTERN ', 12)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
if string_(13)=='"' && string_(end)=='"'
header.gridpattern=strtrim(string_(14:end-1));
end
ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'STARTANGLESINDEG ', 17)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(18:end), '%f');
if ~isempty(errmsg) || count~=4
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.startanglesindeg=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'ENDANGLESINDEG ', 15)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(16:end), '%f');
if ~isempty(errmsg) || count~=4
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.endanglesindeg=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'GONIOMODEL_1 ', 13)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(14:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.goniomodel_1=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'GONIOMODEL_2 ', 13)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(14:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.goniomodel_2=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'WAVELENGTH  ', 11)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
[A, count, errmsg] = sscanf(string_(12:end), '%f');
if ~isempty(errmsg)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.wavelength=A';ind=ind+1;string_=header_stream_char_2D(ind,:);
if ~strncmp(string_, 'MONOCHROMATOR ', 14)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
header.monochromator=strtrim(string_(15:end));
ind=ind+1;ind=ind+1;string_=deblank(header_stream_char_2D(ind,:));
if ~strncmp(string_, 'HISTORY ', 8)
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
if string_(9)=='"' && string_(end)=='"'
header.history=strtrim(string_(10:end-1));
end
esperantoHeadOut=header;isOkHeader=true;
end
function [pixels_2D, header, isOk_java] = qpcyWBN(esperantoFileName)
pixels_2D=[];
header=[];isOk_java=false;K = com.company.Esperanto;K.readfile(esperantoFileName);
if ~K.getstatus()
warning(K.getmessage()');return
end
dimentions=K.getdimentions();binning=K.getbinning();
pixels_2D=reshape(K.getvalues,[dimentions(1), dimentions(2)]);header=struct(...
'dimension_1', dimentions(1),...
'dimension_2', dimentions(2),...
'binningX', binning(1),...
'binningY', binning(2),...
'compression_method', (K.getcompressiontype()'),...
'time', [0 0 0],...
'monitor', [0 0 0 0],...
'pixelsize', [0 0],...
'timestamp', '',...
'startanglesindeg', [0 0 0 0],...
'endanglesindeg', [0 0 0 0],...
'goniomodel_1', [0 0 0 0 0 0 0 0 0 0],...
'goniomodel_2', [0 0 0 0],...
'wavelength', [0 0 0 0],...
'rawheader', (K.getrawheader().toCharArray'),...
'monochromator', '');isOk_java = true;
end
function [imageFormat, imagesData, isOk]=k2uG9uAqeCHE(imagesFolder, imageFormats, runFileData)
imageFormat=[];imagesData=[];isOk=false;
if isempty(imageFormats)
imageFormats={'img', 'cbf', 'edf', 'esperanto'};
elseif ischar(imageFormats)
imageFormats={imageFormats};
elseif ~iscellstr(imageFormats)
warninng(['wrong class of the input parameter ''imageFormats''.'...
' Should be ''char'' or ''cellstr'''])
return
end
[cellArrayOfCellstrOfNames, listLength,   useless, isOkDir] = ...
a9SljmjRJu(imagesFolder, imageFormats, 1);
if ~isOkDir
return
end
for frmt=1:length(imageFormats)
imageFormat=imageFormats{frmt};imagesNamesCellArray=cellArrayOfCellstrOfNames{frmt};
if listLength(frmt)~=0
switch imageFormat
case 'img'
[scanStructArray, isOk]=AB6eycfa(...
imagesNamesCellArray, runFileData);
if ~isOk && ~isempty(runFileData)
warning(['no *.img files were found with the '...
'nameprefix written in the ''run file'' '...
'struct. Starting force method of search '...
'of any ''*.img'' files.'])
[scanStructArray, isOk]=AB6eycfa(...
imagesNamesCellArray, []);
end
case 'cbf'
[scanStructArray, isOk]=fIh6mrONC(imagesNamesCellArray);case 'edf'
[scanStructArray, isOk]=ie28vPqvkz6o(imagesNamesCellArray);case 'esperanto'
[scanStructArray, isOk]=...
unSSPswzyPynOYc(imagesNamesCellArray, runFileData);otherwise
warning(['unknown image file format ''' imageFormat ''''])
return
end
if ~isOk
return
end
imagesData=scanStructArray;break
end
end
if isOk
for sc=1:length(imagesData)
imagesData(sc).imagesFolder=imagesFolder;
end
end
end
function [scanStructArray, isOk]=fIh6mrONC(imagesNamesCellArray)
scanStructArray=createImageDataTemplate();isOk=false;
if isempty(imagesNamesCellArray)
return
end
imagesNamesCellArrayLng=length(imagesNamesCellArray);
randomFileNameHasAppropriateLength=false;
for im=1:imagesNamesCellArrayLng
randomFileName=imagesNamesCellArray{im};
randomFileNameLng=length(randomFileName);randomFilenamePrefix=randomFileName(1:end-16);
randomFilenamePrefixUnderscore=randomFileName(1:end-15);
randomFilenamePrefixUnderscoreLng=length(randomFilenamePrefixUnderscore);
if randomFileNameLng>=16
randomFileNameHasAppropriateLength=true;break
end
end
if ~randomFileNameHasAppropriateLength
warning(['cann''t find filenames with appropriate length of name. Example of the proper filename: ''prefix_0002p_00003.cbf'', where run index=1 & frame index=3.'])
return
end
filenamePrefix=randomFilenamePrefix;
logicalIndexesOfFilesWithAppropriateNameLength=true(imagesNamesCellArrayLng, 1);
for im=1:imagesNamesCellArrayLng
if numel(imagesNamesCellArray{im})~=randomFileNameLng
logicalIndexesOfFilesWithAppropriateNameLength(im)=false;
end
end
framesNamesCellArray=imagesNamesCellArray(logicalIndexesOfFilesWithAppropriateNameLength);
framesNamesCellArrayLng=numel(framesNamesCellArray);
imagesNamesCharArray=char(framesNamesCellArray);
logicalIndexesOfFilesWithSameNamePrefix=all(...
imagesNamesCharArray(:,1:randomFilenamePrefixUnderscoreLng)==...
repmat(randomFilenamePrefixUnderscore, [framesNamesCellArrayLng, 1])...
,2);framesNamesCellArray=framesNamesCellArray(logicalIndexesOfFilesWithSameNamePrefix);
imagesNamesCharArray=imagesNamesCharArray(logicalIndexesOfFilesWithSameNamePrefix, (randomFilenamePrefixUnderscoreLng+1):end-4);
logicalIndexesOfFilesWithAppropriateDilimeter=...
imagesNamesCharArray(:,5)=='p' & imagesNamesCharArray(:,6)=='_';
framesNamesCellArray=framesNamesCellArray(logicalIndexesOfFilesWithAppropriateDilimeter);
runsCharArray=imagesNamesCharArray(logicalIndexesOfFilesWithAppropriateDilimeter, 1:4);
framesCharArray=imagesNamesCharArray(logicalIndexesOfFilesWithAppropriateDilimeter, 7:11);
logicalIndexesOfFilesWithAccurateScanAndFrameIndexes=...
all(runsCharArray>47 & runsCharArray<58, 2) & all(framesCharArray>47 & framesCharArray<58, 2);
framesNamesCellArray=framesNamesCellArray(logicalIndexesOfFilesWithAccurateScanAndFrameIndexes);
runsCharArray=runsCharArray(logicalIndexesOfFilesWithAccurateScanAndFrameIndexes,:);
framesCharArray=framesCharArray(logicalIndexesOfFilesWithAccurateScanAndFrameIndexes,:);
runsUint8Array=double(runsCharArray)-48;framesUint8Array=double(framesCharArray)-48;
scansIndexesRawArray=  runsUint8Array(:,1)*1e3+ runsUint8Array(:,2)*1e2+ runsUint8Array(:,3)*1e1+ runsUint8Array(:,4);
framesIndexesRawArray=framesUint8Array(:,1)*1e4+framesUint8Array(:,2)*1e3+framesUint8Array(:,3)*1e2+framesUint8Array(:,4)*1e1+framesUint8Array(:,5);
if isempty(scansIndexesRawArray)
[scanStructArray_test, isOk_test]=XciCyqMAwjpnu(imagesNamesCellArray, 'cbf');
if isOk_test
scanStructArray=scanStructArray_test;else
return
end
else
scanStructArray=uHscooqDOK6gq(...
scansIndexesRawArray, framesIndexesRawArray, framesNamesCellArray, filenamePrefix);
end
for sc=1:numel(scanStructArray)
scanStructArray(sc).imagesFormat='cbf';
end
isOk=true;
end
function [scanStructArray, isOk]=ie28vPqvkz6o(imagesNamesCellArray)
scanStructArray=createImageDataTemplate();isOk=false;
fullNumOfImages=length(imagesNamesCellArray);firstName=imagesNamesCellArray{1};
firstNameLng=length(firstName);brokenIndexes=false(fullNumOfImages,1);
for im=1:fullNumOfImages
if numel(imagesNamesCellArray{im})~=firstNameLng
brokenIndexes(im)=true;
end
end
imagesNamesCellArray(brokenIndexes)=[];
if isempty(imagesNamesCellArray)
return
end
numOfImages=length(imagesNamesCellArray);imageNamesChar=char(imagesNamesCellArray);
imageNamesChar=imageNamesChar(:,1:end-4);firstStringLng=firstNameLng-4;namePrefixLng=0;
for col=1:firstStringLng
if all(imageNamesChar(:,col)==firstName(col))
namePrefixLng=col;else
break
end
end
if namePrefixLng==0 || namePrefixLng==firstStringLng
return
end
filenamePrefix=firstName(1:namePrefixLng);
indexChar=imageNamesChar(:,namePrefixLng+1:end);
if any(any(indexChar<48 | indexChar>57))
return
end
framesIndexes=str2num(indexChar);
scansIndexes=ones(numOfImages, 1);scanStructArray=uHscooqDOK6gq(...
scansIndexes, framesIndexes, imagesNamesCellArray, filenamePrefix);
for sc=1:numel(scanStructArray)
scanStructArray(sc).imagesFormat='edf';
end
isOk=true;
end
function [scanStructArray, isOk]=AB6eycfa(...
imagesNamesCellArray, runFileData)
[scanStructArray, isOk]=aVSs6lluXcg6(...
imagesNamesCellArray, runFileData, 'img');
for sc=1:numel(scanStructArray)
scanStructArray(sc).imagesFormat='img';
end
end
function [scanStructArray, isOk]=unSSPswzyPynOYc(...
imagesNamesCellArray, runFileData)
[scanStructArray, isOk]=aVSs6lluXcg6(...
imagesNamesCellArray, runFileData, 'esperanto');
for sc=1:numel(scanStructArray)
scanStructArray(sc).imagesFormat='esperanto';
end
end
function [scanStructArray, isOk]=aVSs6lluXcg6(...
imagesNamesCellArray, runFileData, imageFormat)
scanStructArray=createImageDataTemplate();isOk=false;
if isempty(runFileData)
[scanStructArray, isOk_img]=XciCyqMAwjpnu(imagesNamesCellArray, imageFormat);
isOk=isOk_img;return
end
filenamePrefix=runFileData.runFileHead.name;
if ~ischar(filenamePrefix) ||...
(~isempty(filenamePrefix) && ...
~(isvector(filenamePrefix) && size(filenamePrefix, 1)==1))
warning('experiment name obtained from the crysalis ''run file'' struct is wrong')
return
end
numOfFramesInEachRun=zeros(1,numel(runFileData.scansStruct));
for i=1:length(numOfFramesInEachRun)
numOfFramesInEachRun(i)=runFileData.scansStruct(i).done;
end
numOfFramesInEachRun=numOfFramesInEachRun(:)';numOfScans=length(runFileData.scansStruct);
scansIndexes=zeros(sum(numOfFramesInEachRun),1);
framesIndexes=zeros(sum(numOfFramesInEachRun),1);e=0;
for sc=1:numOfScans
numOfFramesInRun=numOfFramesInEachRun(sc);scansIndexes(e+1:e+numOfFramesInRun)=sc;
framesIndexes(e+1:e+numOfFramesInRun)=1:numOfFramesInRun;e=e+numOfFramesInRun;
end
filesNamesConstructed=strtrim(cellstr(num2str([scansIndexes framesIndexes], [filenamePrefix '_%i_%i.' imageFormat])));
fileNamesInFolder=imagesNamesCellArray;indexesOfConstructedFilenamesThatExistInFolder=...
ismember(filesNamesConstructed, fileNamesInFolder);
if ~any(indexesOfConstructedFilenamesThatExistInFolder)
return
end
scansIndexes=scansIndexes(indexesOfConstructedFilenamesThatExistInFolder);
framesIndexes=framesIndexes(indexesOfConstructedFilenamesThatExistInFolder);
filesNamesConstructed=filesNamesConstructed(indexesOfConstructedFilenamesThatExistInFolder);
scanStructArray=uHscooqDOK6gq(...
scansIndexes, framesIndexes, filesNamesConstructed, filenamePrefix);
for sc=1:length(scanStructArray)
scan=scanStructArray(sc);
if ~scan.scanIsEmpty
numOfReqiiredFrames=numOfFramesInEachRun(sc);
if numOfReqiiredFrames~=scan.requiredNumberOfImagesInScan
scan.requiredNumberOfImagesInScan=numOfReqiiredFrames;
scan.indexesOfTheExistingFrames(numOfReqiiredFrames)=false;
scan.imagesNames{numOfReqiiredFrames}=[];scanStructArray(sc)=scan;
end
end
end
isOk=true;
end
function [scanStructArray, isOk]=XciCyqMAwjpnu(imagesNamesCellArray, imageFormat)
scanStructArray=createImageDataTemplate();isOk=false;
extLength=numel(imageFormat)+1;imagesNamesCharArray=char(imagesNamesCellArray);
imagesNamesCellArrayLng=length(imagesNamesCellArray);
for im=1:imagesNamesCellArrayLng
curImageName=imagesNamesCellArray{im};
indexPartOfTheCurrentNameIsOk=false;refIndex=strfind(curImageName, 'ref_');
if ~isempty(refIndex)
lastRefIndex=refIndex(end);
namePartWithIndexes=curImageName((lastRefIndex+4):end-extLength);
nameparts=regexp(namePartWithIndexes, '_', 'split');
if ~isempty(nameparts) && numel(nameparts)==4 &&...
~isempty(nameparts{1}) && ~isempty(nameparts{2}) &&...
~isempty(nameparts{3}) && ~isempty(nameparts{4}) &&...
all(nameparts{1}>47 & nameparts{1}<58) &&...
all(nameparts{2}>47 & nameparts{2}<58) &&...
all(nameparts{3}>47 & nameparts{3}<58) &&...
all(nameparts{4}>47 & nameparts{4}<58)
continue
end
end
UnderscoreIndex=find(curImageName=='_', 2, 'last');
if ~isempty(UnderscoreIndex) && numel(UnderscoreIndex)==2
namePartWithIndexes=curImageName(UnderscoreIndex(1)+1:end-extLength);
nameparts=regexp(namePartWithIndexes, '_', 'split');
if ~isempty(nameparts) && numel(nameparts)==2 &&...
~isempty(nameparts{1}) && ~isempty(nameparts{2}) &&...
all(nameparts{1}>47 & nameparts{1}<58) &&...
all(nameparts{2}>47 & nameparts{2}<58)
indexPartOfTheCurrentNameIsOk=true;
filenamePrefix=curImageName(1:(UnderscoreIndex(1)-1));filenamePrefix_underscore=...
curImageName(1:UnderscoreIndex(1));
end
end
if indexPartOfTheCurrentNameIsOk
break
end
end
if ~indexPartOfTheCurrentNameIsOk
return
end
baseFileNameLng=length(filenamePrefix_underscore);imagesNamesCharArray_cut=...
imagesNamesCharArray(:,1:baseFileNameLng);
baseFileName2D=repmat(filenamePrefix_underscore,...
[size(imagesNamesCharArray,1) 1]);indexesOfImagesWithBaseName=...
all(imagesNamesCharArray_cut==baseFileName2D,2);
indexPartChar=imagesNamesCharArray(indexesOfImagesWithBaseName, baseFileNameLng+1:end-extLength);
indexPartLng=size(indexPartChar, 2);
if indexPartLng<3
return
end
imagesNames=imagesNamesCellArray;
imagesNames=imagesNames(indexesOfImagesWithBaseName);numOfImages=length(imagesNames);
imagesStrings=cell(numOfImages,1);brokenNames=false(numOfImages, 1);
for im=1:numOfImages
curName=imagesNames{im}(baseFileNameLng+1:end-extLength);
if any(curName==' ');brokenNames(im)=true;continue
end
imagesStrings{im}=curName;
end
imagesStrings(brokenNames)=[];numOfImages=length(imagesStrings);
imagesNamesSplitted=regexp(imagesStrings, '_', 'split');
indexesOfBrokenNames=false(numOfImages,1);
for im=1:numOfImages
if numel(imagesNamesSplitted{im})~=2 || isempty(imagesNamesSplitted{im}{1}) || isempty(imagesNamesSplitted{im}{2})
indexesOfBrokenNames=true;
end
end
imagesNamesSplitted(indexesOfBrokenNames)=[];
imagesNames(indexesOfBrokenNames)=[];numOfImages=numel(imagesNamesSplitted);
scansCell=cell(numOfImages,1);framesCell=cell(numOfImages,1);
for im=1:numOfImages
scansCell{im}=imagesNamesSplitted{im}{1};framesCell{im}=imagesNamesSplitted{im}{2};
end
scansChar=char(scansCell);framesChar=char(framesCell);indexesOfAppropriateNames=all(...
(scansChar>47 & scansChar<58) | scansChar==32,2) &...
all( (framesChar>47 & framesChar<58) | framesChar==32 ...
,2);
if ~any(indexesOfAppropriateNames)
return
end
framesNamesCellArray=imagesNames(indexesOfAppropriateNames);
scansChar=scansChar(indexesOfAppropriateNames,:);
framesChar=framesChar(indexesOfAppropriateNames,:);
scansIndexesRawArray=str2num(scansChar);
framesIndexesRawArray=str2num(framesChar);scanStructArray=uHscooqDOK6gq(...
scansIndexesRawArray, framesIndexesRawArray, framesNamesCellArray, filenamePrefix);
isOk=true;
end
function scanStructArray=uHscooqDOK6gq(scansIndexesRawArray, framesIndexesRawArray, framesNamesCellArray, filenamePrefix)
[useless, sortIndexes]=sortrows(...
[scansIndexesRawArray framesIndexesRawArray]);
scansIndexesArray=scansIndexesRawArray(sortIndexes);
framesIndexesArray=framesIndexesRawArray(sortIndexes);
framesNamesCellArray=framesNamesCellArray(sortIndexes);
uniqueScans=unique(scansIndexesArray);
if uniqueScans(1)==0
uniqueScans(1)=[];warning('frames in zero indexed scan would be skipped')
end
numOfScans=uniqueScans(end);
frameIndexesInEachScan=cell(numOfScans, 1);imagesNamesInEachScan=cell(numOfScans, 1);
for sc=(uniqueScans')
curScanFramesIndexes=scansIndexesArray==sc;
frameIndexesInCurScanCompact=framesIndexesArray(   curScanFramesIndexes);
imagesNamesInCurScanCompact=framesNamesCellArray(curScanFramesIndexes);
firstFrameIndex=frameIndexesInCurScanCompact(1);
lastFrameIndex=frameIndexesInCurScanCompact(end);
if firstFrameIndex==0
frameIndexesInCurScan=false(lastFrameIndex+1, 1);
frameIndexesInCurScan(frameIndexesInCurScanCompact+1)=true;
imagesNamesInCurScan=cell(lastFrameIndex+1, 1);
imagesNamesInCurScan(frameIndexesInCurScanCompact+1)=imagesNamesInCurScanCompact;else
frameIndexesInCurScan=false(lastFrameIndex  , 1);
frameIndexesInCurScan(frameIndexesInCurScanCompact)=true;
imagesNamesInCurScan=cell(lastFrameIndex, 1);
imagesNamesInCurScan(frameIndexesInCurScanCompact)=imagesNamesInCurScanCompact;
end
frameIndexesInEachScan{sc}=frameIndexesInCurScan;
imagesNamesInEachScan{ sc}=imagesNamesInCurScan;
end
scanStructArray=createImageDataTemplate();scanStructArray.imageNamePrefix=filenamePrefix;
scanStructArray(1:numOfScans)=scanStructArray(1);
for sc=(uniqueScans')
framesIndexes=frameIndexesInEachScan{sc};
requiredNumberOfImagesInScan=numel(framesIndexes);
actualNumberOfImagesInScan=sum(framesIndexes);scanStructArray(sc).scanIsFull=...
(requiredNumberOfImagesInScan==actualNumberOfImagesInScan);
scanStructArray(sc).requiredNumberOfImagesInScan=requiredNumberOfImagesInScan;
scanStructArray(sc).actualNumberOfImagesInScan=actualNumberOfImagesInScan;
scanStructArray(sc).indexesOfTheExistingFrames=framesIndexes;
scanStructArray(sc).imagesNames=imagesNamesInEachScan{ sc};
scanStructArray(sc).scanIsEmpty=false;
end
end
function imageData=createImageDataTemplate()
commentString=['requiredNumberOfImagesInScan: number of the frames'...
' that should be in scan (according to *.run file data or to'...
' the maximum index found in the images'' filenames). '...
'actualNumberOfImagesInScan: number of the images that'...
' have been found in the images'' folder. '...
'scanIsFull: is true when required and actual numbers'...
' of the images are equal'];imageData=struct('imageNamePrefix', '',...
'scanIsEmpty', true, 'scanIsFull', false, ...
'requiredNumberOfImagesInScan', nan, 'actualNumberOfImagesInScan',...
nan, 'indexesOfTheExistingFrames', [], 'imagesNames', {{}},...
'imagesFormat', '',...
...
'comment', commentString);
end
function f = iOR5OPq(varargin)
matlabVersionReleaseDate=datenum(version('-date'));
if matlabVersionReleaseDate<datenum('August 22, 2012')
switch nargin
case 0
f = fullfile_lateVersion();case 1
f = fullfile_lateVersion(varargin{1});case 2
f = fullfile_lateVersion(varargin{1}, varargin{2});case 3
f = fullfile_lateVersion(varargin{1}, varargin{2}, varargin{3});case 4
f = fullfile_lateVersion(varargin{1}, varargin{2}, varargin{3}, varargin{4});case 5
f = fullfile_lateVersion(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5});
case 6
f = fullfile_lateVersion(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6});
otherwise
string=['f = fullfile_lateVersion(varargin{1}' sprintf(', varargin{%d}',[2:nargin]) ');'];
eval(string);
end
else
switch nargin
case 0
f = fullfile();case 1
f = fullfile(varargin{1});case 2
f = fullfile(varargin{1}, varargin{2});case 3
f = fullfile(varargin{1}, varargin{2}, varargin{3});case 4
f = fullfile(varargin{1}, varargin{2}, varargin{3}, varargin{4});case 5
f = fullfile(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5});case 6
f = fullfile(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6});
otherwise
string=['f = fullfile(varargin{1}' sprintf(', varargin{%d}',[2:nargin]) ');'];
eval(string);
end
end
end
function f = fullfile_lateVersion(varargin)
error(nargchk(1, Inf, nargin, 'struct'));persistent fileSeparator;
if isempty(fileSeparator)
fileSeparator = filesep;
end
argIsACell = cellfun('isclass', varargin, 'cell');
theInputs = varargin;f = theInputs{1};try
if nargin == 1
f = refinePath(f, fileSeparator);return;
elseif any(argIsACell)
theInputs(cellfun(@(x)~iscell(x)&&isempty(x), theInputs)) = [];else
theInputs(cellfun('isempty', theInputs)) = '';
end
if length(theInputs)>1
theInputs{1} = ensureTrailingFilesep(theInputs{1}, fileSeparator);
end
if ~isempty(theInputs)
theInputs(2,:) = {fileSeparator};theInputs{2,1} = '';theInputs(end) = '';
if any(argIsACell)
f = strcat(theInputs{:});else
f = [theInputs{:}];
end
end
f = refinePath(f,fileSeparator);catch exc
locHandleError(exc, theInputs(1,:));
end
end
function f = ensureTrailingFilesep(f,fileSeparator)
if iscell(f)
for i=1:numel(f)
f{i} = addTrailingFileSep(f{i},fileSeparator);
end
else
f = addTrailingFileSep(f,fileSeparator);
end
end
function str = addTrailingFileSep(str, fileSeparator)
persistent bIsPC
if isempty (bIsPC)
bIsPC = ispc;
end
if ~isempty(str) && (str(end) ~= fileSeparator && ~(bIsPC && str(end) == '/'))
str = [str, fileSeparator];
end
end
function f = refinePath(f, fs)
persistent singleDotPattern multipleFileSepPattern
if isempty(singleDotPattern)
singleDotPattern = [fs, '.', fs];multipleFileSepPattern = [fs, fs];
end
f = strrep(f, '/', fs);
if iscell(f)
hasSingleDotCell = ~cellfun('isempty',strfind(f, singleDotPattern));
if any(hasSingleDotCell)
f(hasSingleDotCell) = replaceSingleDots(f(hasSingleDotCell), fs);
end
hasMultipleFileSepCell = ~cellfun('isempty',strfind(f, multipleFileSepPattern));
if any(hasMultipleFileSepCell)
f(hasMultipleFileSepCell) = replaceMultipleFileSeps(f(hasMultipleFileSepCell), fs);
end
else
if ~isempty(strfind(f, singleDotPattern))
f = replaceSingleDots(f, fs);
end
if ~isempty(strfind(f, multipleFileSepPattern))
f = replaceMultipleFileSeps(f, fs);
end
end
end
function f = replaceMultipleFileSeps(f, fs)
persistent fsEscape multipleFileSepRegexpPattern
if isempty(fsEscape)
fsEscape = ['\', fs];
if ispc
multipleFileSepRegexpPattern = '(^(\\\\\?\\.*|(\w+:)?\\+))|(\\)\\+';else
multipleFileSepRegexpPattern = ['(?<!^(\w+:)?' fsEscape '*)(', fsEscape, ')', fsEscape '+'];
end
end
f = regexprep(f, multipleFileSepRegexpPattern, '$1');
end
function f = replaceSingleDots(f, fs)
persistent fsEscape singleDotRegexpPattern
if isempty(fsEscape)
fsEscape = ['\', fs];
if ispc
singleDotRegexpPattern = '(^\\\\(\?\\.*|\.(?=\\)))|(\\)(?:\.\\)+';else
singleDotRegexpPattern = ['(',fsEscape,')', '(?:\.', fsEscape, ')+'];
end
end
f = regexprep(f, singleDotRegexpPattern, '$1');
end
function locHandleError(theException, theInputs)
idToThrow = 'MATLAB:fullfile:UnknownError';switch theException.identifier
case {'MATLAB:catenate:dimensionMismatch', ...
'MATLAB:strcat:InvalidInputSize', 'MATLAB:strrep:MultiRowInput', 'MATLAB:strcat:NumberOfInputRows', 'MATLAB:UnableToConvert'}
firstNonscalarCellArg = struct('idx', 0, 'size', 0);
for argIdx = 1:numel(theInputs)
currentArg = theInputs{argIdx};
if isscalar(currentArg)
continue;
elseif ischar(currentArg) && ~isrow(currentArg)
idToThrow = 'MATLAB:fullfile:NumCharRowsExceeded';
elseif iscell(currentArg)
currentArgSize = size(currentArg);
if firstNonscalarCellArg.idx == 0
firstNonscalarCellArg.idx = argIdx;firstNonscalarCellArg.size = currentArgSize;
elseif ~isequal(currentArgSize, firstNonscalarCellArg.size)
idToThrow = 'MATLAB:fullfile:CellstrSizeMismatch';
end
end
end
otherwise
argIsInvalidType = ~cellfun(@(arg)isnumeric(arg)&&isreal(arg)||ischar(arg)||iscellstr(arg), theInputs);
if any(argIsInvalidType)
idToThrow = 'MATLAB:fullfile:InvalidInputType';
end
end
excToThrow = MException(message(idToThrow));
excToThrow = excToThrow.addCause(theException);throwAsCaller(excToThrow);
end
function [ applicableFrames, Rarray, isOk ] = h0tBk6Fi6MhHvT( ...
surfacesInResiprocalSpace, volumeEdgesInSampleBasis, numOfBinFull,  ...
UBmatrix, experimentGeometryInfo, chosenVolume, additionalInformationForGettingApplicableFrames )
applicableFrames=[];
Rarray=[];isOk=false;evaldSphereCenter = surfacesInResiprocalSpace.evaldSphereCenter;
evaldSphereTopPoint = surfacesInResiprocalSpace.evaldSphereTopPoint;
detectorVertices = surfacesInResiprocalSpace.detectorVertices;
if strcmp(chosenVolume.shape, 'parallelepiped') || strcmp(chosenVolume.shape, 'simple parallelepiped') || strcmp(chosenVolume.shape, 'cylinder')
if strcmp(chosenVolume.shape, 'parallelepiped') || strcmp(chosenVolume.shape, 'cylinder')
zeroVertex=additionalInformationForGettingApplicableFrames.zeroVertex(:)';
axe1=additionalInformationForGettingApplicableFrames.axe1(:)';
axe2=additionalInformationForGettingApplicableFrames.axe2(:)';
axe3=additionalInformationForGettingApplicableFrames.axe3(:)';
axe1Norm=additionalInformationForGettingApplicableFrames.axe1Norm;
axe2Norm=additionalInformationForGettingApplicableFrames.axe2Norm;
axe3Norm=additionalInformationForGettingApplicableFrames.axe3Norm;
axes123Norm=[axe1Norm axe2Norm axe3Norm];minAxeNorm=min(axes123Norm);
numOfBinsInTheThinestAxe=(nthroot(numOfBinFull/(prod(axes123Norm)/minAxeNorm^3), 3));
if numOfBinsInTheThinestAxe<1
numOfBinsHKL=round(axes123Norm/minAxeNorm);numOfBinsFullCur=prod(numOfBinsHKL);
if numOfBinsFullCur>numOfBinFull*1.2
coef=sqrt(numOfBinsFullCur/numOfBinFull);numOfBinsHKL=round(axes123Norm/minAxeNorm/coef);
numOfBinsHKL(numOfBinsHKL<1)=1;numOfBinsFullCur=prod(numOfBinsHKL);
if numOfBinsFullCur>numOfBinFull*1.2
coef=numOfBinsFullCur/numOfBinFull;
numOfBinsHKL=round(axes123Norm/minAxeNorm/coef);numOfBinsHKL(numOfBinsHKL<1)=1;
end
end
else
numOfBinsHKL=round(numOfBinsInTheThinestAxe*axes123Norm/minAxeNorm);
end
numOfBinsH=numOfBinsHKL(1);numOfBinsK=numOfBinsHKL(2);numOfBinsL=numOfBinsHKL(3);
axe1Delta=axe1/numOfBinsH;axe2Delta=axe2/numOfBinsK;axe3Delta=axe3/numOfBinsL;
firstSmallVolumeCenter=zeroVertex+(axe1Delta+axe2Delta+axe3Delta)/2;
centersOfTheSmallVolumesInTheSampleBasis=zeros(3,  numOfBinsH*numOfBinsK*numOfBinsL);
for i=1:3
HcoordinateAxe1Part1D=(0:numOfBinsH-1)*axe1Delta(i);
HcoordinateAxe1Part3D=repmat(HcoordinateAxe1Part1D, [numOfBinsK, 1, numOfBinsL]);
HcoordinateAxe1Part3DPermuted=permute(HcoordinateAxe1Part3D, [2 1 3]);
HcoordinateAxe2Part1D=(0:numOfBinsK-1)*axe2Delta(i);
HcoordinateAxe2Part3D=repmat(HcoordinateAxe2Part1D, [numOfBinsH, 1, numOfBinsL]);
HcoordinateAxe2Part3DPermuted=permute(HcoordinateAxe2Part3D, [1 2 3]);
HcoordinateAxe3Part1D=(0:numOfBinsL-1)*axe3Delta(i);
HcoordinateAxe3Part3D=repmat(HcoordinateAxe3Part1D, [numOfBinsH, 1, numOfBinsK]);
HcoordinateAxe3Part3DPermuted=permute(HcoordinateAxe3Part3D, [1 3 2]);
HcoordinateAxe0Part3DPermuted=HcoordinateAxe1Part3DPermuted+...
HcoordinateAxe2Part3DPermuted+...
HcoordinateAxe3Part3DPermuted;
centersOfTheSmallVolumesInTheSampleBasis(i, :)=(HcoordinateAxe0Part3DPermuted(:)'+firstSmallVolumeCenter(i));
end
vectorFromCenterToTopOfTheSmallVolumeInTheSampleBasis=(axe1Delta+axe2Delta+axe3Delta)'/2;
elseif strcmp(chosenVolume.shape, 'simple parallelepiped')
numOfBins=4;numOfBinsH=numOfBins;numOfBinsK=numOfBins;numOfBinsL=numOfBins;
hl=volumeEdgesInSampleBasis(1,1);   hr=volumeEdgesInSampleBasis(1,2);
kl=volumeEdgesInSampleBasis(2,1);   kr=volumeEdgesInSampleBasis(2,2);
ll=volumeEdgesInSampleBasis(3,1);   lr=volumeEdgesInSampleBasis(3,2);
deltaH=abs(hr-hl)/numOfBinsH/2;deltaK=abs(kr-kl)/numOfBinsK/2;
deltaL=abs(lr-ll)/numOfBinsL/2;hLineSpace = linspace(hl,hr-2*deltaH,numOfBinsH)+deltaH;
kLineSpace = linspace(kl,kr-2*deltaK,numOfBinsK)+deltaK;
lLineSpace = linspace(ll,lr-2*deltaL,numOfBinsL)+deltaL;
[hMeshgrid, kMeshgrid, lMeshgrid] = meshgrid(hLineSpace, kLineSpace, lLineSpace);
vectorFromCenterToTopOfTheSmallVolumeInTheSampleBasis = [deltaH; deltaK; deltaL];
centersOfTheSmallVolumesInTheSampleBasis = [hMeshgrid(:) kMeshgrid(:) lMeshgrid(:)]';
end
centersOfTheSmallVolumes0Basis = UBmatrix*centersOfTheSmallVolumesInTheSampleBasis;
vectorFromCenterToTopOfTheSmallVolume0Basis = UBmatrix*vectorFromCenterToTopOfTheSmallVolumeInTheSampleBasis;
radiusOfTheSmallVolume0Basis = norm(vectorFromCenterToTopOfTheSmallVolume0Basis);
radiusOfTheSmallVolumeInGoniometerBasis = radiusOfTheSmallVolume0Basis;else
disp(['ERROR in (' mfilename '): can''t manage the volume shape'])
return
end
pixelFragmentation=1;useOnlySpecialFrames=[];
Rarray = wuDvdC1( experimentGeometryInfo, useOnlySpecialFrames, pixelFragmentation );
applicableFrames=Js0rrhLfj0MhyWit(Rarray, ...
centersOfTheSmallVolumes0Basis, evaldSphereCenter, ...
evaldSphereTopPoint, radiusOfTheSmallVolumeInGoniometerBasis, ...
detectorVertices );isOk=true;
end
function [ applicableFrames ] = Js0rrhLfj0MhyWit(...
Rarray, centersOfTheSmallVolumes0Basis, evaldSphereCenter,...
evaldSphereTopPoint, radiusOfTheSmallVolumeInGoniometerBasis,...
detectorVertices )
numOfFrames=size(Rarray, 1);RarrayFullAxe1=zeros(numOfFrames,3);
RarrayFullAxe2=zeros(numOfFrames,3);RarrayFullAxe3=zeros(numOfFrames,3);
for i=1:numOfFrames
RarrayFullAxe1(i, :)=Rarray{i}(1,:);
RarrayFullAxe2(i, :)=Rarray{i}(2,:);RarrayFullAxe3(i, :)=Rarray{i}(3,:);
end
centersOfTheSmallVolumesInGoniometerBasisAxe1=RarrayFullAxe1*centersOfTheSmallVolumes0Basis;
centersOfTheSmallVolumesInGoniometerBasisAxe2=RarrayFullAxe2*centersOfTheSmallVolumes0Basis;
centersOfTheSmallVolumesInGoniometerBasisAxe3=RarrayFullAxe3*centersOfTheSmallVolumes0Basis;
clear('Rarray', 'RarrayFullAxe1', 'RarrayFullAxe2', 'RarrayFullAxe3', 'centersOfTheSmallVolumes0Basis')
detectorCenter = mean(detectorVertices, 2);
detectorRadiuses = detectorVertices-[detectorCenter detectorCenter detectorCenter detectorCenter];
detectorRadius=max(sqrt(sum(detectorRadiuses.^2, 1)));
volumeRadius = radiusOfTheSmallVolumeInGoniometerBasis;
volumesCentersAxe1 = centersOfTheSmallVolumesInGoniometerBasisAxe1;
volumesCentersAxe2 = centersOfTheSmallVolumesInGoniometerBasisAxe2;
volumesCentersAxe3 = centersOfTheSmallVolumesInGoniometerBasisAxe3;clear(...
'centersOfTheSmallVolumesInGoniometerBasisAxe1',...
'centersOfTheSmallVolumesInGoniometerBasisAxe2',...
'centersOfTheSmallVolumesInGoniometerBasisAxe3'...
);evaldSphereCenterToVolumesCentersAxe1 = volumesCentersAxe1-evaldSphereCenter(1);
evaldSphereCenterToVolumesCentersAxe2 = volumesCentersAxe2-evaldSphereCenter(2);
evaldSphereCenterToVolumesCentersAxe3 = volumesCentersAxe3-evaldSphereCenter(3);
distancesBtwSphereCenterAndVolumesCenters=sqrt(...
evaldSphereCenterToVolumesCentersAxe1.^2+...
evaldSphereCenterToVolumesCentersAxe2.^2+...
evaldSphereCenterToVolumesCentersAxe3.^2);
evaldSphereCenterToEvaldSphereTopPoint=(evaldSphereTopPoint-evaldSphereCenter)';
dotProducts =...
evaldSphereCenterToEvaldSphereTopPoint(1)*evaldSphereCenterToVolumesCentersAxe1+...
evaldSphereCenterToEvaldSphereTopPoint(2)*evaldSphereCenterToVolumesCentersAxe2+...
evaldSphereCenterToEvaldSphereTopPoint(3)*evaldSphereCenterToVolumesCentersAxe3;clear(...
'evaldSphereCenterToVolumesCentersAxe1',...
'evaldSphereCenterToVolumesCentersAxe2',...
'evaldSphereCenterToVolumesCentersAxe3'...
);detectorCenterToVolumesCentersAxe1 = volumesCentersAxe1-detectorCenter(1);
detectorCenterToVolumesCentersAxe2 = volumesCentersAxe2-detectorCenter(2);
detectorCenterToVolumesCentersAxe3 = volumesCentersAxe3-detectorCenter(3);clear(...
'volumesCentersAxe1',...
'volumesCentersAxe2',...
'volumesCentersAxe3'...
);distancesBtwDetectorCenterAndVolumesCenters = sqrt(...
detectorCenterToVolumesCentersAxe1.^2+...
detectorCenterToVolumesCentersAxe2.^2+...
detectorCenterToVolumesCentersAxe3.^2);clear(...
'detectorCenterToVolumesCentersAxe1',...
'detectorCenterToVolumesCentersAxe2',...
'detectorCenterToVolumesCentersAxe3'...
);isCrossing=any(...
(...
(distancesBtwSphereCenterAndVolumesCenters-volumeRadius)<1 &...
(distancesBtwSphereCenterAndVolumesCenters+volumeRadius)>1 &...
dotProducts>-volumeRadius & ...
(distancesBtwDetectorCenterAndVolumesCenters-volumeRadius)<detectorRadius...
),...
2);applicableFrames=find(isCrossing);
end
function [ imageaxis, isOk ] = KAgjA40wxXJWo( detectorType )
imageaxis=[];isOk=false;switch lower(detectorType)
case {''}
fastestDimention='strings';
firstPixelPosition='lower left corner';imageFormat='auto';case {'img', 'esperanto'}
fastestDimention='strings';firstPixelPosition='lower left corner';
imageFormat=detectorType;case {'supernova', 'default', ''}
fastestDimention='strings';firstPixelPosition='lower left corner';
imageFormat='img';case {'cbf', 'snbl', 'pilatus 2m, 24-0111', 'pilatus 2m 24-0111'}
fastestDimention='columns';
firstPixelPosition='upper right corner';imageFormat='cbf';otherwise
warning('unknown detector type')
return
end
imageaxis=struct(...
'imageFastestDimentionOrientation', fastestDimention, ...
'firstPixelPosition', firstPixelPosition,...
'imageFormat', imageFormat);isOk=true;
end
function RarrayCell = wuDvdC1( experimentGeometryInfo, useOnlyThisFrames, pixelFragmentation )
num_of_frames = experimentGeometryInfo.num_Of_frames;
if exist('useOnlyThisFrames', 'var')==1 && ~isempty(useOnlyThisFrames)
useOnlyThisFrames=useOnlyThisFrames(:)';else
useOnlyThisFrames = 1:num_of_frames;
end
binningAngle=1;
if exist('pixelFragmentation', 'var')~=1 || isempty(pixelFragmentation)
binningAngle=1;pixelFragmentation=1;
elseif isstruct(pixelFragmentation)
if pixelFragmentation.usePixelFragmentation==true
if isfield(pixelFragmentation, 'pixelFragmentationAngle')
binningAngle=round(pixelFragmentation.pixelFragmentationAngle);
if binningAngle<1
binningAngle=1;
end
end
end
elseif isnumeric(pixelFragmentation)
binningAngle=round(abs(pixelFragmentation));
if pixelFragmentation==0 pixelFragmentation=1; end
else
warning('Error')
return
end
if strcmp(experimentGeometryInfo.scanned_axe, 'phi')
scanned_axe='phi';
elseif strcmp(experimentGeometryInfo.scanned_axe, 'omega')
scanned_axe='omega';else
warning('cann''t recognize input data')
return
end
RarrayCell = cell(num_of_frames, binningAngle);
goniometerCurrent=struct('alpha', nan, 'beta', nan, 'omega', nan, 'kappa', nan, 'phi', nan);
goniometerCurrent.alpha=experimentGeometryInfo.alpha;
goniometerCurrent.beta=experimentGeometryInfo.beta;
goniometerCurrent.omega=experimentGeometryInfo.omega;
goniometerCurrent.kappa=experimentGeometryInfo.kappa;
goniometerCurrent.phi=experimentGeometryInfo.phi;
coef1=(experimentGeometryInfo.angle_end-experimentGeometryInfo.angle_start)/(num_of_frames*binningAngle);
coef2=experimentGeometryInfo.angle_start-coef1/2;
if ~isfield(pixelFragmentation, 'pixelJoinAngle') || isempty(pixelFragmentation.pixelJoinAngle)
for i=useOnlyThisFrames
for j=1:binningAngle
current_frame_angle=coef2+coef1*((i-1)*binningAngle+j);switch scanned_axe
case 'phi'
goniometerCurrent.phi=current_frame_angle + experimentGeometryInfo.phi;case 'omega'
goniometerCurrent.omega=current_frame_angle + experimentGeometryInfo.omega;
end
R = cQQTjXPqQ8fajmy3(goniometerCurrent);RarrayCell{i,j} = R;
end
end
else
warning('this code is not finished yet')
return
end
end
function [ versionStruct ] = IRTmq__v()
versionStruct=struct(...
'releaseversion'         ,0.71                  ,...
'releasedate'           ,'2020.05.02'           ,...
'matlabversion'         ,version()              ,...
'matlabrelease'         ,version('-release')    ,...
'matlabreleasedate'     ,version('-date')       ,...
'java'                  ,version('-java')         );
end
function [I, header, isOk] = OZNnACLMtb1Y0vkK( fileName )
I=[];header=[];isOk=false;imgFile = fileName;
if exist(imgFile, 'file')~=2
warning(['file ''', imgFile, ''' doesn''t exist'])
return;
end
g = fopen(imgFile, 'r', 'l');masInt8 = fread(g,  '*int8');fclose(g);
if length(masInt8)<5120
warning(['length of the file ''', imgFile, ''' is too small. Cann''t read the header'])
return;
end
headerInt8=(masInt8(1:5120)');[header, isOkHeader]=bXFQVXJTRzV( headerInt8 );
if ~isOkHeader
warning(['Error reading file ''' fileName ''''])
end
if strncmp(header.compression, 'COMPRESSION=TY5', 15)
compressionType=5;
elseif strncmp(header.compression, 'COMPRESSION=TY1', 15)
compressionType=1;else
warning(['Error: Unfedined type of the compression' ])
return
end
fd=header.nx;
sd=header.ny;masBinaryPart=masInt8(5121:end);masBinaryPartLength=length(masBinaryPart);
if masBinaryPartLength<fd*sd
warning(['length of the file ''', imgFile, ''' is too small'])
return;
end
if compressionType==1
mas = double(typecast(masBinaryPart(1:fd*sd),'uint8'));
mas_16_32=masBinaryPart(fd*sd+1:end);mas_16_32_Lng=length(mas_16_32);
if length(mas)<fd*sd
warning(['length of the file ''', imgFile, ''' is too small'])
fclose(g);return;
end
k_int16_i=find(mas==254);k_int32_i=find(mas==255);mas=mas-127;
if ~isempty(k_int16_i)
k_int16_lng = length(k_int16_i);
if mas_16_32_Lng<k_int16_lng*2
warning(['length of the file ''', imgFile, ''' is too small. Cann''t read binary part'])
return;
end
k_int16=double(typecast(mas_16_32(1:k_int16_lng*2),'int16'));
mas(k_int16_i) = k_int16;else
k_int16_lng=0;
end
if ~isempty(k_int32_i)
k_int32_lng = length(k_int32_i);
if mas_16_32_Lng<k_int16_lng*2+k_int32_lng*4
warning(['length of the file ''', imgFile, ''' is too small. Cann''t read binary part'])
return;
end
k_int32=double(typecast(mas_16_32(k_int16_lng*2+1:k_int16_lng*2+k_int32_lng*4),'int32'));
mas(k_int16_i) = k_int16;mas(k_int32_i) = k_int32;
end
I=reshape(cumsum(mas), fd, sd);
elseif compressionType==5
masDouble=double(typecast(masBinaryPart, 'uint8'));lmas=masBinaryPartLength;
if lmas<fd*sd
warning(['length of the file ''', imgFile, ''' is too small'])
return;
end
x=find(masDouble>253);lx=length(x);
if lx==0 || x(1)>sd*fd
masDouble=masDouble-127;I=cumsum(reshape(masDouble(1:sd*fd), fd, sd), 1);isOk=true;return
end
xLogical=true(1,lmas);
xLogical(x)=false;int16_indexs=zeros(1,lx);int16_numbers=zeros(1,2*lx);
int16_ind=0;int32_indexs=zeros(1,lx);int32_numbers=zeros(1,4*lx);int32_ind=0;ee=0;i=0;
while i<lx
i=i+1;curIndex=x(i);
if curIndex>sd*fd+4*int32_ind+2*int16_ind
break
end
if masDouble(x(i))==255
int32_numbers(int32_ind*4+1)=( curIndex+1 );int32_numbers(int32_ind*4+2)=( curIndex+2 );
int32_numbers(int32_ind*4+3)=( curIndex+3 );int32_numbers(int32_ind*4+4)=( curIndex+4 );
int32_indexs(int32_ind+1)=x(i)-4*int32_ind-2*int16_ind;
int32_ind=int32_ind+1;xLogical(curIndex+1) = false;
xLogical(curIndex+2) = false;xLogical(curIndex+3) = false;xLogical(curIndex+4) = true;
for j=1:4
if lx>i && x(i+1)-curIndex<5
i=i+1;else
break
end
end
else
ee=ee+1;
int16_numbers(int16_ind*2+1)=( curIndex+1 );int16_numbers(int16_ind*2+2)=( curIndex+2 );
int16_indexs(int16_ind+1)=x(i)-4*int32_ind-2*int16_ind;
int16_ind=int16_ind+1;xLogical(curIndex+1) = false;xLogical(curIndex+2) = true;
if lx>=i+1 && x(i+1)-curIndex<3
i=i+1;
end
if lx>=i+1 && x(i+1)-curIndex<3
i=i+1;
end
end
end
masDouble=masDouble-127;
xLogical((fd*sd+2*int16_ind+4*int32_ind+1):end)=false;I_delta=masDouble(xLogical);
if int16_ind>0
int16_numbers=int16_numbers(1:int16_ind*2);int16indexs=int16_indexs(1:int16_ind);
I_2=double(typecast(masBinaryPart(int16_numbers), 'int16'));I_delta(int16indexs)=I_2;
end
if int32_ind>0
int32_numbers=int32_numbers(1:int32_ind*4);int32indexs=int32_indexs(1:int32_ind);
I_4=double(typecast(masBinaryPart(int32_numbers), 'int32'));I_delta(int32indexs)=I_4;
end
I=cumsum(reshape(I_delta, fd, sd), 1);else
fclose(g);return
end
isOk = true;
end
function [headerStruct, isOk]=bXFQVXJTRzV( headerInt8 )
headerStruct=[];isOk=[];lineBreakersFind=find(headerInt8==13 | headerInt8==10);
lineBreakers=[0 lineBreakersFind (length(headerInt8)+1)];
numOfLineBreakers=length(lineBreakers);
headerChar=char(headerInt8);cellStr=cell(numOfLineBreakers,1);e=0;
for i=1:numOfLineBreakers-1
if lineBreakers(i+1)-lineBreakers(i)>1
e=e+1;cellStr{e} = (headerChar( lineBreakers(i)+1:lineBreakers(i+1)-1 ));
end
end
numOfLinesInHeader=e;cellStr(numOfLinesInHeader+1:end)=[];
if numOfLinesInHeader<6
warning(['header of the file ''', imgFile, ''' is corrupt'])
return;
end
header_format = cellStr{1};compressionTypeString=deblank(cellStr{2});
header_compression = compressionTypeString;line3=cellStr{3};
col = strfind(line3, 'NX=');header_nx = sscanf(line3(col+3:end), '%d', inf);
col = strfind(line3, 'NY=');header_ny = sscanf(line3(col+3:end), '%d', inf);
col = strfind(line3, 'OI=');header_oi = sscanf(line3(col+3:end), '%d', inf);
col = strfind(line3, 'OL=');header_ol = sscanf(line3(col+3:end), '%d', inf);
line4=cellStr{4};A=sscanf(line4, '%*s %d %*s %d %*s %d %*s %d %*s %d %*s %d', inf);
header_nheader = A(1);header_ng = A(2);header_ns = A(3);header_nk = A(4);
header_ns2 = A(5);header_nh = A(6);line6=cellStr{6};header_time = deblank(line6(6:40));
headerStruct = struct('format', header_format, 'compression', header_compression,...
'nx', header_nx, 'ny', header_ny, 'oi', header_oi, 'ol', header_ol,...
'nheader', header_nheader, 'ng', header_ng, 'ns', header_ns, 'nk', header_nk,...
'ns2', header_ns2, 'nh', header_nh, 'time', header_time);isOk=true;
end
function [ChangeOfBasisNewToOld, newAxe1, newAxe2, newAxe3]=twmBsAr3b(lineDirection)
ChangeOfBasisNewToOld=[];
if ~isnumeric(lineDirection)
disp('ERROR in (fZmakeChangeOfBasisMatrix): input argument should be numeric');
end
if size(lineDirection)~=[1,3];
disp('ERROR in (fZmakeChangeOfBasisMatrix): size of the input argument should be 1x3');
end
if norm(lineDirection)==0;
disp('ERROR in (fZmakeChangeOfBasisMatrix): input argument is a Null vector');
end
lineDirectionNormed=lineDirection/norm(lineDirection);
[lineDirectionsNormedSorted]=sort(lineDirectionNormed);
if lineDirectionsNormedSorted(1)==0 && lineDirectionsNormedSorted(2)==0
if lineDirectionNormed==[1 0 0]
axe1=[1 0 0];axe2=[0 1 0];axe3=[0 0 1];
elseif lineDirectionNormed==[0 1 0]
axe1=[0 1 0];axe2=[0 0 1];axe3=[1 0 0];
elseif lineDirectionNormed==[0 0 1]
axe1=[0 0 1];axe2=[1 0 0];axe3=[0 1 0];else
disp('error')
end
elseif lineDirectionsNormedSorted(1)==0 && lineDirectionsNormedSorted(2)~=0
if lineDirectionNormed(1)==0
axe1=lineDirectionNormed;axe2=[1 0 0];axe3=cross(axe1, axe2);axe3=axe3/norm(axe3);
elseif lineDirectionNormed(2)==0
axe1=lineDirectionNormed;axe2=[0 1 0];axe3=cross(axe1, axe2);axe3=axe3/norm(axe3);
elseif lineDirectionNormed(3)==0
axe1=lineDirectionNormed;axe3=[0 0 1];axe2=cross(axe3, axe1);axe2=axe2/norm(axe2);else
disp('error')
end
else
axe1=lineDirectionNormed;tmpAxe2=cross([0;0;1], axe1);tmpAxe3=cross(axe1, [0;1;0]);
if norm(tmpAxe2)>norm(tmpAxe3)
axe2=tmpAxe2/norm(tmpAxe2);axe3=cross(axe1, axe2);axe3=axe3/norm(axe3);else
axe3=tmpAxe3/norm(tmpAxe3);axe2=cross(axe3, axe1);axe2=axe2/norm(axe2);
end
end
T=[axe1;axe2;axe3]';ChangeOfBasisNewToOld=T;newAxe1=axe1;newAxe2=axe2;newAxe3=axe3;
end
function [ experimentGeometryInfo, isOk ] = gOxS6ShVOiGI( crackerData, runFileData )
experimentGeometryInfo=[];isOk=false;
numOfRuns=runFileData.runFileHead.number_of_scans;detector=crackerData.detector;
experimentGeometryInfo=struct('alpha', 0, 'beta', 0, 'scanned_axe', 0, 'angle_start', 0, 'angle_end', 0, 'num_Of_frames', 0, 'omega', 0, 'detector_thetaArm', 0, 'kappa', 0, 'phi', 0,...
'detector', detector, 'beam', crackerData.beam, ...
'lambda', crackerData.lambda, 'ub_matrix', struct('ub', crackerData.ub, 'iub', crackerData.iub) );
experimentGeometryInfo(2:numOfRuns)=experimentGeometryInfo(1);
for i=1:numOfRuns
currentScan=runFileData.scansStruct(i);scan_to_do=double(currentScan.to_do);
scan_done=double(currentScan.done);scan_angle_start=currentScan.start;
scan_angle_end=currentScan.end;scan_angle_width=currentScan.width;
if scan_to_do~=scan_done
warning(['In run number ' num2str(i) ' ''to_do'' & ''done'' numbers are different. Programm would use ''done'' number'])
end
if scan_angle_start<scan_angle_end && scan_angle_width<0 ||...
scan_angle_start>scan_angle_end && scan_angle_width>0
warning(['Bad data in scan with index ' num2str(i) ': ''start/width/end'' angles'' entries have wrong signs. Sign of the angle''s entry ''width'' would be changed'])
scan_angle_width=-scan_angle_width;
end
if abs(scan_angle_width-(scan_angle_end-scan_angle_start)/scan_done)>1e-4
warning(['Bad data in scan with index ' num2str(i) ': ''to_do'' entry & ''start/width/end'' angles'' entries don''t match to each other. Angle''s entry ''end'' would be recalculated according to the entry ''done'''])
scan_angle_end=scan_angle_start+scan_angle_width*scan_done;
end
experimentGeometryInfo(i).alpha=crackerData.goniometer.alpha;
experimentGeometryInfo(i).beta=crackerData.goniometer.beta;
experimentGeometryInfo(i).scanned_axe=currentScan.scanned_axe;
experimentGeometryInfo(i).num_Of_frames=double(currentScan.done);
experimentGeometryInfo(i).detector_thetaArm=...
crackerData.goniometer.theta+currentScan.detector_thetaArm;
experimentGeometryInfo(i).kappa=crackerData.goniometer.kappa+currentScan.kappa;
experimentGeometryInfo(i).angle_start=scan_angle_start;
experimentGeometryInfo(i).angle_end=scan_angle_end;switch currentScan.scanned_axe
case 'phi'
experimentGeometryInfo(i).omega=crackerData.goniometer.omega+currentScan.omega;
experimentGeometryInfo(i).phi=nan;
experimentGeometryInfo(i).phi=crackerData.goniometer.phi;case 'omega'
experimentGeometryInfo(i).phi=crackerData.goniometer.phi+currentScan.phi;
experimentGeometryInfo(i).omega=nan;
experimentGeometryInfo(i).omega=crackerData.goniometer.omega;otherwise
warning('bad input data, scan axe is not properly defined')
return
end
end
isOk=true;
end
function [isOk, experimentInfo, experimentNames, valueFromMultiDataset, normalization, backgroundWindow, chosenVolume, pixelFragmentation, make3DArray, additionalTools] =...
Ezg29UNgW2(...
experimentInfoInput, experimentNamesInput, valueFromMultiDatasetInput, normalizationInput, backgroundWindowInput, ...
chosenVolumeInput, pixelFragmentationInput, make3DArrayInput, additionalToolsInput)
isOk=false;experimentInfo=experimentInfoInput;
[ isOkExperimentNames, experimentNames] = NktVAMfBiRr3BVa(experimentNamesInput);
[ isOkValueFromMultiDataset, valueFromMultiDataset] = EE49YTi(valueFromMultiDatasetInput);
[ isOkNormalization, normalization] = p2yA5BBshhfEOH(normalizationInput);
[ isOkBackgroundWindow, backgroundWindow] = zDpVIov(backgroundWindowInput);
[ isOkChosenVolume, chosenVolume] = pbZmw7IQX(chosenVolumeInput);
numOfVolumes=length(chosenVolume);volumeshape = chosenVolume.shape;
[ isOkPixelFragmentation, pixelFragmentation] = z4ON_7ofWw(pixelFragmentationInput);
[ isOkMake3DArray, make3DArray] = oDUyn9J(make3DArrayInput, numOfVolumes, volumeshape);
[ isOkAdditionalTools, additionalTools] = gF3lTKVEpP(additionalToolsInput);
if ~all([isOkExperimentNames isOkValueFromMultiDataset isOkNormalization isOkBackgroundWindow isOkChosenVolume isOkPixelFragmentation isOkMake3DArray isOkAdditionalTools])
return
end
isOk=true;
end
function [ isOk, experimentNamesOutput] = NktVAMfBiRr3BVa(experimentNamesInput)
isOk=true;experimentNamesOutput=[];
experimentNames=experimentNamesInput;imageFormat=experimentNames.imageFormat;
if strcmp(imageFormat, 'cbf') || strcmp(imageFormat, 'img') || strcmp(imageFormat, 'edf') || strcmp(imageFormat, 'esperanto')
else
experimentNames.imageFormat='unknown';
end
experimentNamesOutput=experimentNames;isOk=true;
end
function [ isOk, valueFromMultiDatasetOutput] = EE49YTi(valueFromMultiDatasetInput)
valueFromMultiDatasetOutput=[];
isOk=false;valueFromMultiDatasetOutput=valueFromMultiDatasetInput;isOk=true;
end
function [ isOk, normalizationOutput] = p2yA5BBshhfEOH(normalizationInput)
normalizationOutput=[];isOk=false;normalizationOutput.normalizeOnFirstFrame=false;
normalizationOutput.useNormalizationOn = 0;normalizationOutput.returnFrameNumbers=false;
normalizationOutput.rememberFlux=false;normalizationOutput.rememberBg=false;
if isstruct(normalizationInput)
if isfield(normalizationInput, 'normalizeOnFirstFrame') && islogical(normalizationInput.normalizeOnFirstFrame)
normalizationOutput.normalizeOnFirstFrame=normalizationInput.normalizeOnFirstFrame;
end
if isfield(normalizationInput, 'useNormalizationOn') && isnumeric(normalizationInput.useNormalizationOn) &&...
numel(normalizationInput.useNormalizationOn)==1
useNorm=normalizationInput.useNormalizationOn;
if useNorm==1 || useNorm==2 || useNorm==3
normalizationOutput.useNormalizationOn=useNorm;
end
end
if isfield(normalizationInput, 'returnFrameNumbers')
if isnumeric(normalizationInput.returnFrameNumbers)
normalizationInput.returnFrameNumbers=logical(normalizationInput.returnFrameNumbers);
end
if islogical(normalizationInput.returnFrameNumbers) && numel(normalizationInput.returnFrameNumbers)==1
normalizationOutput.returnFrameNumbers=normalizationInput.returnFrameNumbers;
end
end
if isfield(normalizationInput, 'rememberFlux')
if isnumeric(normalizationInput.rememberFlux)
normalizationInput.rememberFlux=logical(normalizationInput.rememberFlux);
end
if islogical(normalizationInput.rememberFlux) && numel(normalizationInput.rememberFlux)==1
normalizationOutput.rememberFlux=normalizationInput.rememberFlux;
end
end
if isfield(normalizationInput, 'rememberBg')
if isnumeric(normalizationInput.rememberBg)
normalizationInput.rememberBg=logical(normalizationInput.rememberBg);
end
if islogical(normalizationInput.rememberBg) && numel(normalizationInput.rememberBg)==1
normalizationOutput.rememberBg=normalizationInput.rememberBg;
end
end
else
disp(' ')
disp(['ERROR: (' mfilename '): input parameter ''normalization'' is broken. Using default value...'])
end
isOk=true;
end
function [ isOk, backgroundWindow] = zDpVIov(backgroundWindowInput)
backgroundWindow=[];isOk=false;backgroundWindow=backgroundWindowInput;isOk=true;
end
function [isOk, chosenVolumeNew] = pbZmw7IQX(chosenVolume)
isOk=false;chosenVolumeNew=[];chosenVolumeNew.shape=[];
if ~isstruct(chosenVolume)
disp(['ERROR in (' mfilename '): parameter ''chosenVolume'' supposed to be struct'])
return
end
numOfVolumes=length(chosenVolume);
if isfield(chosenVolume, 'useNewAxes') &&...
(isnumeric(chosenVolume(1).useNewAxes) || islogical(chosenVolume(1).useNewAxes)) &&...
numel(chosenVolume(1).useNewAxes)==1
chosenVolumeNew(1).useNewAxes=logical(chosenVolume(1).useNewAxes);else
chosenVolumeNew(1).useNewAxes=false;
end
if numOfVolumes>1
chosenVolumeNew(2:numOfVolumes)=chosenVolumeNew(1);
end
volumeShapeIsIdentified=false;isOkUseSphere=false;
isOkUseCylinder=false;isOkUseParallelepiped=false;isOkUseSimpleParallelepiped=false;
if ~volumeShapeIsIdentified
if isfield(chosenVolume, 'sphere') && isstruct(chosenVolume(1).sphere)
sphere=chosenVolume.sphere;
if isfield(sphere, 'useSphere') &&...
(isnumeric(sphere.useSphere) || islogical(sphere.useSphere)) &&...
numel(sphere.useSphere)==1 && logical(sphere.useSphere)
if isfield(sphere, {'radius', 'center'})
radius=sphere.radius;center=sphere.center;
if isnumeric(radius) && isnumeric(center) &&...
numel(radius)==1 && numel(center)==3 &&...
radius>0 && ~any(isnan(center))
sphereNew=[];
sphereNew.useSphere=logical(sphere.useSphere);sphereNew.radius=double(sphere.radius);
sphereNew.center=double(sphere.center(:));isOkUseSphere=true;
end
end
end
end
end
if ~volumeShapeIsIdentified
if isfield(chosenVolume, 'cylinder') && isstruct(chosenVolume(1).cylinder)
cylinder=chosenVolume.cylinder;cylinderNew=struct;
if isfield(cylinder, 'useNewAxes') && (isnumeric(cylinder.useNewAxes) || islogical(cylinder.useNewAxes))
cylinderNew.useNewAxes=logical(cylinder.useNewAxes);
end
if isfield(cylinder, 'useCylinder') &&...
(isnumeric(cylinder.useCylinder) || islogical(cylinder.useCylinder)) &&...
numel(cylinder.useCylinder)==1 && logical(cylinder.useCylinder)
cylinderNew.useCylinder=true;
if isfield(cylinder, 'radius')
radius=cylinder.radius;
if isnumeric(radius) && numel(radius)==1 && radius>0 && isfinite(radius)
cylinderNew.radius=double(radius);
cylinderNewVersion1=cylinderNew;isOkCenterAndDirection=false;
if isfield(cylinder, {'center', 'direction', 'length'})
if isnumeric(cylinder.center) && isnumeric(cylinder.direction) &&...
numel(cylinder.center)==3 && numel(cylinder.direction)==3 &&...
norm(cylinder.direction)>0 && ~any(isnan(cylinder.center)) &&...
isnumeric(cylinder.length) && numel(cylinder.length)==1
isOkCenterAndDirection=true;cylinderNewVersion1.center=double(cylinder.center);
cylinderNewVersion1.direction=double(cylinder.direction);
cylinderNewVersion1.length=double(cylinder.length);
end
end
cylinderNewVersion2=cylinderNew;isOkTopBottomPoints=false;
if isfield(cylinder, {'useTopBottomPoints'})
if (isnumeric(cylinder.useTopBottomPoints) || islogical(cylinder.useTopBottomPoints)) &&...
numel(cylinder.useTopBottomPoints)==1 && logical(cylinder.useTopBottomPoints)
if isfield(cylinder, {'bottomPoint', 'topPoint'})
topPoint=cylinder.topPoint;bottomPoint=cylinder.bottomPoint;
if isnumeric(bottomPoint) && isnumeric(topPoint) &&...
numel(bottomPoint)==3 && numel(topPoint)==3 &&...
norm(bottomPoint(:)-topPoint(:))>0
cylinderNewVersion2.topPoint=double(cylinder.topPoint);
cylinderNewVersion2.bottomPoint=double(cylinder.bottomPoint);isOkTopBottomPoints=true;
end
end
end
end
if any([isOkCenterAndDirection isOkTopBottomPoints])
isOkUseCylinder=true;volumeShapeIsIdentified=true;
if isOkCenterAndDirection
cylinderNew=cylinderNewVersion1;cylinderNew.useTopBottomPoints=false;else
cylinderNew=cylinderNewVersion2;cylinderNew.useTopBottomPoints=true;
end
end
end
end
end
end
end
if ~volumeShapeIsIdentified
if isfield(chosenVolume, 'parallelepiped') && isstruct(chosenVolume(1).parallelepiped)
parallelepipedNew=struct;isAllVolumesParallelepipedOk=true;
for i=1:numOfVolumes
isCurrentVolumesParallelepipedOk=false;chosenVolumeCurrent=chosenVolume(i);
if isstruct(chosenVolumeCurrent.parallelepiped)
parallelepiped=chosenVolumeCurrent.parallelepiped;
if isfield(parallelepiped, 'useParallelepiped') &&...
(isnumeric(parallelepiped.useParallelepiped) || islogical(parallelepiped.useParallelepiped)) &&...
numel(parallelepiped.useParallelepiped)==1 && logical(parallelepiped.useParallelepiped)
if isfield(parallelepiped, {'zeroVertex', 'axe1', 'axe2', 'axe3'})
zeroVertex=parallelepiped.zeroVertex;
if isnumeric(zeroVertex) && numel(zeroVertex)==3 && ~any(isnan(zeroVertex))
axe1=parallelepiped.axe1;axe2=parallelepiped.axe2;axe3=parallelepiped.axe3;
if isnumeric(axe1) && isnumeric(axe2) && isnumeric(axe3) &&...
numel(axe1)==3 && numel(axe2)==3 && numel(axe3)==3 &&...
abs(det([axe1(:) axe2(:) axe3(:)]'))>0
zeroVertex=zeroVertex(:);axe1=axe1(:);axe2=axe2(:);axe3=axe3(:);
axe123Absolute=repmat(zeroVertex,1,8)+[[0;0;0] axe1 axe2 axe3 axe1+axe2 axe1+axe3 axe2+axe3 axe1+axe2+axe3];
axe123Relative=[[0;0;0] axe1 axe2 axe3 axe1+axe2 axe1+axe3 axe2+axe3 axe1+axe2+axe3];
hklBoundsAbsoluteMin=min(axe123Absolute,[],2);
hklBoundsAbsoluteMax=max(axe123Absolute,[],2);
hklBoundsRelative=(max(axe123Relative,[],2)-min(axe123Relative,[],2))/2;
isCurrentVolumesParallelepipedOk=true;parallelepipedNew(i).zeroVertex=double(zeroVertex);
parallelepipedNew(i).axe1=double(axe1);
parallelepipedNew(i).axe2=double(axe2);parallelepipedNew(i).axe3=double(axe3);
parallelepipedNew(i).hklBounds.relative=hklBoundsRelative;
parallelepipedNew(i).hklBounds.absoluteMin=hklBoundsAbsoluteMin;
parallelepipedNew(i).hklBounds.absoluteMax=hklBoundsAbsoluteMax;
end
end
end
end
end
if ~isCurrentVolumesParallelepipedOk
clear('parallelepipedNew')
isAllVolumesParallelepipedOk=false;break
end
end
if isAllVolumesParallelepipedOk
volumeShapeIsIdentified=true;isOkUseParallelepiped=true;
end
end
end
if ~volumeShapeIsIdentified
if isfield(chosenVolume, {'peak', 'delta'})
isAllVolumesSimpleParallelepipedOk=true;chosenVolumeSimpleNewCurrent=struct;
for i=1:numOfVolumes
peak=chosenVolume(i).peak;delta=chosenVolume(i).delta;
if isnumeric(peak) && isnumeric(delta) && numel(peak)==3 && numel(delta)==3 &&...
~any(isnan(peak)) && all(delta)>0
chosenVolumeSimpleNewCurrent(i).peak=peak(:)';
chosenVolumeSimpleNewCurrent(i).delta=double(delta(:)');
chosenVolumeSimpleNewCurrent(i).shape='simple parallelepiped';
chosenVolumeSimpleNewCurrent(i).useNewAxes=false;else
isAllVolumesSimpleParallelepipedOk=false;break
end
end
if isAllVolumesSimpleParallelepipedOk
volumeShapeIsIdentified=true;
isOkUseSimpleParallelepiped=true;chosenVolumeSimpleNew=chosenVolumeSimpleNewCurrent;
end
end
end
isOk=true;
if isOkUseSphere
chosenVolumeNew.sphere=sphereNew;chosenVolumeNew.shape='sphere';
elseif isOkUseCylinder
chosenVolumeNew.cylinder=cylinderNew;chosenVolumeNew.shape='cylinder';
elseif isOkUseParallelepiped
chosenVolumeNew.parallelepiped=parallelepipedNew;chosenVolumeNew.shape='parallelepiped';
elseif isOkUseSimpleParallelepiped
chosenVolumeNew=chosenVolumeSimpleNew;else
isOk=false;
end
if ~isOk
disp('ERROR: input parameter ''chosenVolume'' is incorrect')
return;
end
end
function [ isOk, pixelFragmentation] = z4ON_7ofWw(pixelFragmentation)
isOk=false;isOkPixelFragmentation=false;isOkUseLessMemoryCoefficient=false;
if ~isempty(pixelFragmentation) && isstruct(pixelFragmentation)
if isfield(pixelFragmentation, {'usePixelFragmentation', 'pixelFragmentationX', 'pixelFragmentationY', 'pixelFragmentationAngle'})
if pixelFragmentation.usePixelFragmentation==true
if isnumeric([pixelFragmentation.pixelFragmentationX pixelFragmentation.pixelFragmentationY pixelFragmentation.pixelFragmentationAngle])
pixelFragmentation.pixelFragmentationX=round(double(pixelFragmentation.pixelFragmentationX));
pixelFragmentation.pixelFragmentationY=round(double(pixelFragmentation.pixelFragmentationY));
pixelFragmentation.pixelFragmentationAngle=round(double(pixelFragmentation.pixelFragmentationAngle));
if all([pixelFragmentation.pixelFragmentationX pixelFragmentation.pixelFragmentationY pixelFragmentation.pixelFragmentationAngle]>0)
if pixelFragmentation.pixelFragmentationX~=1 || pixelFragmentation.pixelFragmentationY~=1 || pixelFragmentation.pixelFragmentationAngle~=1
isOkPixelFragmentation=true;
end
end
end
end
end
end
if isOkPixelFragmentation==false
pixelFragmentation=[];
pixelFragmentation.usePixelFragmentation=false;pixelFragmentation.pixelFragmentationX=1;
pixelFragmentation.pixelFragmentationY=1;pixelFragmentation.pixelFragmentationAngle=1;
end
if isfield(pixelFragmentation, 'useLessMemoryCoefficient') &&...
(isnumeric(pixelFragmentation.useLessMemoryCoefficient) || islogical(pixelFragmentation.useLessMemoryCoefficient))
pixelFragmentation.useLessMemoryCoefficient=round(double(pixelFragmentation.useLessMemoryCoefficient));
if pixelFragmentation.useLessMemoryCoefficient>0
isOkUseLessMemoryCoefficient=true;
end
end
if isOkUseLessMemoryCoefficient==false
pixelFragmentation.useLessMemoryCoefficient=100;
end
isOk=true;
end
function [ isOk, make3DArrayNew] = oDUyn9J(make3DArray, numOfVolumes, volumeshape)
isOk=false;make3DArrayNew=struct;makeClear3DArray=true;
if ~isempty(make3DArray) && isstruct(make3DArray) && isfield(make3DArray, {'makeReconstructionImmediately'})
makeReconstructionImmediately=make3DArray(1).makeReconstructionImmediately;
if numel(makeReconstructionImmediately)==1 && (isnumeric(makeReconstructionImmediately) || islogical(makeReconstructionImmediately))
makeReconstructionImmediately=logical(makeReconstructionImmediately);
if makeReconstructionImmediately
makeClear3DArray=false;
end
end
end
if ~makeClear3DArray
if length(make3DArray)~=numOfVolumes
warning(['size of the input parameter ''make3DArray'' is wrong'])
return
end
makeClear3DArray=false;
if ~isempty(make3DArray) && isstruct(make3DArray) && all(isfield(make3DArray, {'makeReconstructionImmediately', 'nh', 'nk', 'nl'}))
for i=1:numOfVolumes
make3DArrayCurrent=make3DArray(i);
nh=make3DArrayCurrent.nh;nk=make3DArrayCurrent.nk;nl=make3DArrayCurrent.nl;
if numel(nh)==1 && numel(nk)==1 && numel(nl)==1 && isnumeric([nh nk nl])
make3DArrayNew(i).nh=round(double(nh));
make3DArrayNew(i).nk=round(double(nk));make3DArrayNew(i).nl=round(double(nl));
make3DArrayNew(i).makeReconstructionImmediately=true;makeAxes=false;
if isfield(make3DArray, 'makeAxes')
makeAxesInput=make3DArray(i).makeAxes;
if numel(makeAxesInput) && ( islogical(makeAxesInput) || isnumeric(makeAxesInput) )
makeAxes=logical(makeAxesInput);
end
end
make3DArrayNew(i).makeAxes=makeAxes;makeCounts=false;
if isfield(make3DArray, 'makeCounts')
makeCountsInput=make3DArray(i).makeCounts;
if numel(makeCountsInput)==1 && ( islogical(makeCountsInput) || isnumeric(makeCountsInput) )
makeCounts=logical(makeCountsInput);
end
end
make3DArrayNew(i).makeCounts=makeCounts;else
makeClear3DArray=true;break
end
end
else
makeClear3DArray=true;
end
end
if (~strcmp(volumeshape, 'simple parallelepiped') && ~strcmp(volumeshape, 'parallelepiped') )...
&& ~makeClear3DArray
warning('Volume shape doesn''t allow to build 3D matrix')
makeClear3DArray = true;
end
if makeClear3DArray
make3DArrayNew=struct;make3DArrayNew.makeReconstructionImmediately=false;isOk=true;
end
isOk=true;
end
function [ isOk, additionalTools] = gF3lTKVEpP(additionalTools)
isOk=true;return
isOk=false;
formatIsOk=false;isOkUseThirdPartyCracker=false;
isOkUseCrackerChanges=false;crackerDataIsOk=false;anglesIsOk=false;
if ~isempty(additionalTools) && isstruct(additionalTools)
if isfield(additionalTools, 'format')
if ischar(additionalTools.format)
if strcmp(additionalTools.format, 'cbf') || strcmp(additionalTools.format, 'img') || strcmp(additionalTools.format, 'edf')
formatIsOk=true;
end
end
end
if isfield(additionalTools, 'thirdPartyCracker') && isstruct(additionalTools.thirdPartyCracker)
if isfield(additionalTools.thirdPartyCracker, 'useThirdPartyCracker')
if (isnumeric(additionalTools.thirdPartyCracker.useThirdPartyCracker) || islogical(additionalTools.thirdPartyCracker.useThirdPartyCracker))
additionalTools.thirdPartyCracker.useThirdPartyCracker=logical(additionalTools.thirdPartyCracker.useThirdPartyCracker);
end
if islogical(additionalTools.thirdPartyCracker.useThirdPartyCracker)
if isfield(additionalTools.thirdPartyCracker, 'thirdPartyCrackerFullName') &&...
ischar(additionalTools.thirdPartyCracker.thirdPartyCrackerFullName)
isOkUseThirdPartyCracker=true;
end
end
end
end
if isfield(additionalTools, 'crackerChanges') && isstruct(additionalTools.crackerChanges)
if isfield(additionalTools.crackerChanges, 'useChanges')
useChanges=additionalTools.crackerChanges.useChanges;
if isnumeric(useChanges)
useChanges=logical(useChanges);additionalTools.crackerChanges.useChanges=useChanges;
end
if islogical(useChanges) && useChanges
isOkUseCrackerChanges=true;
detectorOut=[];detectorOut.d=0;detectorOut.x0=0;detectorOut.y0=0;
if isfield(additionalTools.crackerChanges, 'detector') &&...
isstruct(additionalTools.crackerChanges.detector)
detectorIn=additionalTools.crackerChanges.detector;
if isfield(detectorIn, 'd') && isstruct(detectorIn.d)
d=detectorIn.d;
if isnumeric(d) && numel(d)==1 && isfinite(d)
detectorOut.d=d;
end
end
if isfield(detectorIn, 'x0') && isstruct(detectorIn.x0)
x0=detectorIn.x0;
if isnumeric(x0) && numel(x0)==1 && isfinite(x0)
detectorOut.x0=x0;
end
end
if isfield(detectorIn, 'y0') && isstruct(detectorIn.y0)
y0=detectorIn.y0;
if isnumeric(y0) && numel(y0)==1 && isfinite(y0)
detectorOut.y0=y0;
end
end
end
additionalTools.crackerChanges.detector=detectorOut;
end
end
end
if isfield(additionalTools, 'useThisCracker')
if (isnumeric(additionalTools.useThisCracker) || islogical(additionalTools.useThisCracker))
additionalTools.useThisCracker=logical(additionalTools.useThisCracker);
end
if islogical(additionalTools.useThisCracker) && additionalTools.useThisCracker==true
if isfield(additionalTools, 'crackerData') && isstruct(additionalTools.crackerData)
crackerDataIsOk=true;
end
end
end
if isfield(additionalTools, 'angles') && isstruct(additionalTools.angles)
if isfield(additionalTools.angles, 'useThisAngles')
if (isnumeric(additionalTools.angles.useThisAngles) || islogical(additionalTools.angles.useThisAngles))
additionalTools.angles.useThisAngles=logical(additionalTools.angles.useThisAngles);
end
if islogical(additionalTools.angles.useThisAngles) && additionalTools.angles.useThisAngles==true
if isfield(additionalTools.angles, 'angleStart') && isfield(additionalTools.angles, 'angleStep') && isfield(additionalTools.angles, 'angleAxe')
if isnumeric(additionalTools.angles.angleStart) && isnumeric(additionalTools.angles.angleStep) && ischar(additionalTools.angles.angleAxe)
if length(additionalTools.angles.angleStart)==1 && length(additionalTools.angles.angleStep)==1
if strcmp(additionalTools.angles.angleAxe, 'phi')
anglesIsOk=true;
end
end
end
end
end
end
end
end
if formatIsOk==false
additionalTools.format='unknown';
end
if isOkUseThirdPartyCracker==false
additionalTools.thirdPartyCracker.useThirdPartyCracker=false;
additionalTools.thirdPartyCracker.thirdPartyCrackerFullName='';
end
if isOkUseCrackerChanges==false
additionalTools.crackerChanges.useChanges=false;
additionalTools.crackerChanges.detector=[];
end
if  crackerDataIsOk==false
additionalTools.useThisCracker=false;
end
if anglesIsOk==false
angles=struct('useThisAngles', false);
additionalTools.angles=angles;additionalTools.format='unknown';
end
isOk=true;
end
function [ imageData, imageFormat, isOk ] =...
naDBLREVD( fileNameOrFolderName )
scanStructArray=[];imageFormat=[];isOk=false;
if ~ischar(fileNameOrFolderName)
warning(['input parameter ''fileNameOrFolderName'' should be of class ''char'''])
return
end
if ~(isvector(fileNameOrFolderName) && size(fileNameOrFolderName, 1)==1)
warning(['input parameter ''fileNameOrFolderName'' should be a 1D string'])
return
end
if exist(fileNameOrFolderName, 'dir')==7
workingFolderPath=fileNameOrFolderName;
dectrisaliasesFileFullName=fullfile(workingFolderPath, 'dectrisaliases.ini');
if exist(dectrisaliasesFileFullName, 'file')~=2
warning(['there is no ''dectrisaliases.ini'' file in the folder ''' workingFolderPath ''''])
return
end
elseif exist(fileNameOrFolderName, 'file')==2
dectrisaliasesFileFullName=fileNameOrFolderName;else
warning(['file or folder ' fileNameOrFolderName ' doesn''t exist'])
return
end
g = fopen(dectrisaliasesFileFullName, 'r', 'l');mas = fread(g,  '*int8');
fclose(g);cellStr=regexp(char(mas'), ['(?:', sprintf('\r\n'), ')+'], 'split');
numOfStrings=length(cellStr);
if ~strcmp(cellStr{1},'[Run list aliases]')
warning(['(' mfilename '): first string in the file '...
'''dectrisaliases.ini'' should be ''[Run list aliases]'''])
return
end
cellStr(1)=[];
if isempty(cellStr{end})
cellStr(end)=[];
end
if isempty(cellStr)
warning(['file ''dectrisaliases.ini'' is broken'])
return
end
stringpartsCell=regexp(cellStr, [ '='], 'split');
indexesInEachString=cell(length(stringpartsCell),1);
filenameInEachString=cell(length(stringpartsCell),1);
for i=1:length(stringpartsCell)
if numel(stringpartsCell{i})==2
indexesInEachString{i}=stringpartsCell{i}{1};
filenameInEachString{i}=stringpartsCell{i}{2};
elseif numel(stringpartsCell{i})<2
warning(['string N ' num2str(i+1)...
' in the file ''dectrisaliases.ini'' is wrong'])
return
else
filenameInEachString{i}=stringpartsCell{i}{end};
numOfPostColumns=numel(filenameInEachString{i})+1;
indexesInEachString{i}=cellStr{i}(1:end-numOfPostColumns);
end
end
indexParts=regexp(indexesInEachString, [ '_'], 'split');
txtIndexesCell=cell(numel(indexParts), 1);
scanIndexesCell=cell(numel(indexParts), 1);imageIndexesCell=cell(numel(indexParts), 1);
for j=1:numel(indexParts)
if numel(indexParts{j})==3
txtIndexesCell{j}=indexParts{j}{1};
scanIndexesCell{j}=indexParts{j}{2};imageIndexesCell{j}=indexParts{j}{3};
elseif numel(indexParts{j})<3
warning(['string N ' num2str(j+1)...
' in the file ''dectrisaliases.ini'' is wrong'])
return
else
scanIndexesCell{j}=indexParts{j}{end-1};imageIndexesCell{j}=indexParts{j}{end};
numOfPostColumns=numel(scanIndexesCell{j}) +numel(imageIndexesCell{j})+2;
txtIndexesCell{j}=indexesInEachString{j}(1:end-numOfPostColumns);
end
end
txtIndexesChar=char(txtIndexesCell);scanIndexesChar=char(scanIndexesCell);
imageIndexesChar=char(imageIndexesCell);scanIndexesNum=str2num(scanIndexesChar);
if any(isnan(scanIndexesNum))
badIndex=find(isnan(scanIndexesNum), 1);warning(['string N ' num2str(badIndex+1)...
' in the file ''dectrisaliases.ini'' is wrong'])
return
end
imageIndexesNum=str2num(imageIndexesChar);
if any(isnan(imageIndexesNum))
badIndex=find(isnan(imageIndexesNum), 1);warning(['string N ' num2str(badIndex+1)...
' in the file ''dectrisaliases.ini'' is wrong'])
return
end
if ~all(all(txtIndexesChar==repmat(...
txtIndexesChar(1,:),[size(txtIndexesChar,1) 1])));warning(['bad data in the file '''...
dectrisaliasesFileFullName ''''])
return
end
scanAndFrameIndexes=[scanIndexesNum imageIndexesNum];
[useless, sortIndexes]=sortrows(scanAndFrameIndexes);
txtIndexesChar=txtIndexesChar(sortIndexes,:);
scanIndexesNum=scanIndexesNum(sortIndexes);imageIndexesNum=imageIndexesNum(sortIndexes);
filenameInEachString=filenameInEachString(sortIndexes);
uniqueScans=unique(scanIndexesNum);numOfUniqueScans=numel(uniqueScans);
if uniqueScans(1)~=1 || uniqueScans(end)-uniqueScans(1)~=numOfUniqueScans-1
warning(['some runs are missing in the file '''...
dectrisaliasesFileFullName ''''])
return
end
[useless,useless,ext]=fileparts(filenameInEachString{1});
imageFormat=ext(2:end);imageNamePrefix=txtIndexesChar(1,:);
commentString=['requiredNumberOfImagesInScan: number of the frames'...
' that should be in scan (according to *.run file data or to'...
'the maximum index found in the images'' filenames). '...
'actualNumberOfImagesInScan: number of the images that'...
' have been found in the images'' folder. '...
'scanIsFull: is true when required and actual numbers'...
'of the images are equal'];imageData=struct('imageNamePrefix', imageNamePrefix,...
'scanIsEmpty', true, 'scanIsFull', false, ...
'requiredNumberOfImagesInScan', nan, 'actualNumberOfImagesInScan',...
nan, 'indexesOfTheExistingFrames', [], 'imagesNames', {{}},...
'imagesFormat', imageFormat,...
'comment', commentString);imageData(2:max(uniqueScans))=imageData;
scanStructArray=struct('name', '', 'imageTitle', '',...
'indexOfTheScan', nan, 'numOfTheImages', nan,...
'imagesIndexes', {''}, 'imagesNames', {''});
scanStructArray(numOfUniqueScans)=scanStructArray(1);
for sc=uniqueScans
curScanLogicalIndexes=scanIndexesNum==sc;
curImageIndexes=imageIndexesNum(curScanLogicalIndexes);
curNumOfImages=numel(curImageIndexes);
if curImageIndexes(1)~=1 ||...
curImageIndexes(end)-curImageIndexes(1)~=curNumOfImages-1
warning(['some frames are missing in the file '''...
dectrisaliasesFileFullName ''''])
return
end
imageData(sc).scanIsEmpty=false;
imageData(sc).scanIsFull=true;imageData(sc).requiredNumberOfImagesInScan=curNumOfImages;
imageData(sc).actualNumberOfImagesInScan=curNumOfImages;
imageData(sc).indexesOfTheExistingFrames=curImageIndexes;
imageData(sc).imagesNames=filenameInEachString(curScanLogicalIndexes);
end
isOk=true;
end
function [ crackerDataOutput, isOk ] = F4TlX7u( varargin )
crackerDataOutput=[];isOk=false;
if isempty(varargin) || (length(varargin)==1 && isempty(varargin{1}))
crackerDataOutput=zWfQgEDomFmi1o();isOk=true;return
end
crackerFullName=varargin{1};crackerData = [];
if ~exist(crackerFullName, 'file')
disp(['WARNING: file ''', crackerFullName, ''' doesn''t exist'])
return;
end
g = fopen(crackerFullName, 'rt');dataIsOk = 0;
while ~feof(g)
mas=fgetl(g);
if strncmp(mas, '§ D A T E', 9)
crackerData.date = mas(10:end);dataIsOk = 1;break
end
end
if ~dataIsOk
disp(['WARNING: there is no ''§ D A T E'' string in file ''', crackerFullName, ''''])
fclose(g);return
end
unitCellIsOk = 0;
while ~feof(g)
mas=fgetl(g);
if strncmp(mas, '§CELL INFORMATION', 17)
if ~feof(g)
[A,countA]=fscanf(g, '%*s%f%*s%f%*s%f%*s%f%*s%f%*s%f%*s', [2,6]);
if countA==12
crackerData.latticeParameters       = A(1,1:3);
crackerData.latticeParametersErrors = A(2,1:3);
crackerData.angles                  = A(1,4:6);
crackerData.anglesErrors            = A(2,4:6);unitCellIsOk = 1;break;
end
end
end
end
if ~unitCellIsOk
disp(['WARNING: there is no ''CELL INFORMATION'' string in file ''', crackerFullName, ''''])
fclose(g);return
end
volumeIsOk = 0;
while ~feof(g)
mas=fgetl(g);
if strncmp(mas, '§  V =', 6)
crackerData.volume = str2double(mas(7:end));
if ~isnan(crackerData.volume)
volumeIsOk = 1;break
end
end
end
if ~volumeIsOk
disp(['WARNING: there is no ''V = ''(volume of the unit cell) string in file ''', crackerFullName, ''''])
fclose(g);return
end
[A, isOkGetValue]=urYOIK7C3gyc(g, 'ALPHA');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''ALPHA'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.alpha = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'BETA');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''BETA'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.beta = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'A1');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''A1'' (lambda) in file ''', crackerFullName, ''''])
fclose(g);return
end
lambda.A1 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'A2');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''A2'' (lambda) in file ''', crackerFullName, ''''])
fclose(g);return
end
lambda.A2 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'B1');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''B1'' (lambda) in file ''', crackerFullName, ''''])
fclose(g);return
end
lambda.B1 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X2');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''X2'' (beam: b2) in file ''', crackerFullName, ''''])
fclose(g);return
end
beam.b2 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X3');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''X3'' (beam: b3) in file ''', crackerFullName, ''''])
fclose(g);return
end
beam.b3 = A;beam.comment='b2, b3: X-RAY BEAM ORIENTATION (DEG)';
[A, isOkGetValue]=urYOIK7C3gyc(g, 'OMEGA');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''OMEGA'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.omega = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'THETA');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''THETA'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.theta = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'KAPPA');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''KAPPA'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.kappa = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'PHI');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''PHI'' in file ''', crackerFullName, ''''])
fclose(g);return
end
goniometer.phi = A;
goniometer.comment='omega, theta, kappa, phi: SOFTWARE ZEROCORRECTION (DEG)';
[A, isOkGetValue]=urYOIK7C3gyc(g, 'PIXELSIZE');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''PIXELSIZE IN 1X1 BINNING (MM)'' in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.pixelSize_in_1x1_binning = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X1');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR ROTATION (DEG): X1'' (detector: d1) in file ''', crackerFullName, ''''])
fclose(g);return
end
detector.d1 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X2');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR ROTATION (DEG): X2'' (detector: d1) in file ''', crackerFullName, ''''])
fclose(g);return
end
detector.d2 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X3');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR ROTATION (DEG): X3'' (detector: d1) in file ''', crackerFullName, ''''])
fclose(g);return
end
detector.d3 = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'DISTANCE');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR DISTANCE (MM)'' (detector: d) in file ''', crackerFullName, ''''])
fclose(g);return
end
detector.d = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR ZERO (PIX, 1X1 BINNING): X'' in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.x0_in_1x1_binning = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'Y');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR ZERO (PIX, 1X1 BINNING): Y'' in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.y0_in_1x1_binning = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'X:');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR BINNING (PIX): X'' in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.pixelBinningX = A;[A, isOkGetValue]=urYOIK7C3gyc(g, 'Y:');
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''DETECTOR BINNING (PIX): Y'' in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.pixelBinningY = A;[A, isOkGetValue]=SWj30Pb0Jv3fj1(g, 'PARAMETERS', 6,1);
if ~isOkGetValue
disp(['WARNING: there is no Instrumental Parameter ''CCD PARAMETERS'' (detector: fastestDimention, secondDimention) in file ''', crackerFullName, ''''])
fclose(g);return
end
pixelBinning.dimentionX_in_1x1_binning = A(5);
pixelBinning.dimentionY_in_1x1_binning = A(6);
detector.x0=pixelBinning.x0_in_1x1_binning/pixelBinning.pixelBinningX;
detector.y0=pixelBinning.y0_in_1x1_binning/pixelBinning.pixelBinningY;
detector.pixelSizeX=[...
pixelBinning.pixelSize_in_1x1_binning*pixelBinning.pixelBinningX];
detector.pixelSizeY=[...
pixelBinning.pixelSize_in_1x1_binning*pixelBinning.pixelBinningY];
detector.dimentionX=pixelBinning.dimentionX_in_1x1_binning/pixelBinning.pixelBinningX;
detector.dimentionY=pixelBinning.dimentionY_in_1x1_binning/pixelBinning.pixelBinningY;
detector.pixelBinning=pixelBinning;
detector.comment='d1, d2, d3: DETECTOR ROTATION; d: detector distance; x0, y0: DETECTOR ZERO; : binningX, binningY: DETECTOR BINNING (PIX)';
while ~feof(g)
S=fscanf(g, '%s ', 1);
if strcmp(S, 'UB')
strWasFoundIsOk = 1;break
end
end
if ~strWasFoundIsOk
disp(['WARNING: there is no ''CRYSTALLOGRAPHY UB'' (matrix) string in file ''', crackerFullName, ''''])
fclose(g);return
end
fgetl(g);[A, countA] = fscanf(g, '%*s %*s %f %f %f %f %f %f %f %f %f ', [3,3]);
if countA~=9
disp(['WARNING: ''CRYSTALLOGRAPHY UB'' matrix size mismatch in file ''', crackerFullName, ''''])
fclose(g);return
end
UB = A';fclose(g);crackerData.beam = beam;crackerData.goniometer = goniometer;
crackerData.detector = detector;crackerData.lambda = lambda;
crackerData.ub = UB;crackerData.iub = inv(UB);crackerDataOutput=crackerData;isOk=true;
end
function [ number, isOk ] = urYOIK7C3gyc( fileID, str )
strWasFoundIsOk = 0;number = nan;isOk=false;
while ~feof(fileID)
S=fscanf(fileID, '%s ', 1);
if strcmp(S, str)
strWasFoundIsOk = 1;break
end
end
if strWasFoundIsOk==0
return
end
while ~feof(fileID)
Snum=fscanf(fileID, '%s ', 1);number = str2double(Snum);
if ~isnan(number)
isOk = true;break
end
end
end
function [ numbers, isOk ] = SWj30Pb0Jv3fj1( fileID, str, numOfValues, numOfSkippedStrings )
strWasFoundIsOk = false;numbers = [];isOk=false;skipedStrings=0;
while ~feof(fileID)
S=fscanf(fileID, '%s ', 1);
if strcmp(S, str)
if skipedStrings<numOfSkippedStrings
skipedStrings=skipedStrings+1;else
strWasFoundIsOk = true;break
end
end
end
if ~strWasFoundIsOk
return
end
numbers=zeros(numOfValues,1)*nan;
while ~feof(fileID)
Snum=fscanf(fileID, '%s ', 1);number = str2double(Snum);
if ~isnan(number)
numbers(1)=number;
for i=2:numOfValues
Snum=fscanf(fileID, '%s ', 1);number = str2double(Snum);
if isnan(number)
return
end
numbers(i)=number;
end
isOk = true;break
end
end
end
function parFileDataTemplate=zWfQgEDomFmi1o()
parFileDataTemplate=struct;
parFileDataTemplate.date='';parFileDataTemplate.latticeParameters=[0 0 0];
parFileDataTemplate.latticeParametersErrors=[0 0 0];
parFileDataTemplate.angles=[0 0 0];parFileDataTemplate.anglesErrors=[0 0 0];
parFileDataTemplate.volume=0;detector=struct;detector.d1=0;
detector.d2=0;detector.d3=0;detector.x0=0;detector.y0=0;detector.pixelSizeX=0;
detector.pixelSizeY=0;detector.dimentionX=0;detector.dimentionY=0;pixelBinning=struct;
pixelBinning.pixelSize_in_1x1_binning=0;pixelBinning.x0_in_1x1_binning=0;
pixelBinning.y0_in_1x1_binning=0;pixelBinning.pixelBinningX=0;
pixelBinning.pixelBinningY=0;pixelBinning.dimentionX_in_1x1_binning=0;
pixelBinning.dimentionY_in_1x1_binning=0;detector.pixelBinning=pixelBinning;
detector.comment='d1, d2, d3: DETECTOR ROTATION; d: detector distance; x0, y0: DETECTOR ZERO; : binningX, binningY: DETECTOR BINNING (PIX)';
beam=struct;beam.b2=0;beam.b3=0;beam.comment= 'b2, b3: X-RAY BEAM ORIENTATION (DEG)';
goniometer=struct;goniometer.alpha=0;goniometer.beta=0;
goniometer.omega=0;goniometer.theta=0;goniometer.kappa=0;goniometer.phi=0;
goniometer.comment= 'omega, theta, kappa, phi: SOFTWARE ZEROCORRECTION (DEG)';
lambda=struct;lambda.A1=0;lambda.A2=0;
lambda.B1=0;parFileDataTemplate.beam=beam;parFileDataTemplate.goniometer=goniometer;
parFileDataTemplate.detector=detector;parFileDataTemplate.lambda=lambda;
parFileDataTemplate.ub=zeros(3,3);parFileDataTemplate.iub=zeros(3,3);
end
function [ runFileData, isOk ] = ODnsRaXF( varargin             )
runFileData=[];runFileHead=[];scansStruct=[];
referenceScansStruct=[];experimentName=[];numOfRuns=[];numOfReferenceRuns=[];isOk=false;
if isempty(varargin) || (length(varargin)==1 && isempty(varargin{1}))
runFileData=DqNOAtji();isOk=true;return
elseif length(varargin)==1
fileNameOrFolderName=varargin{1};else
warning('wrong input parameters')
return
end
if ~ischar(fileNameOrFolderName)
warning(['(' mfilename '): input parameter ''fileNameOrFolderName'' should be of class ''char'''])
return
end
if ~(isvector(fileNameOrFolderName) || size(fileNameOrFolderName,1)==1)
warning(['(' mfilename '): input parameter ''fileNameOrFolderName'' should be a 1D string'])
return
end
if exist(fileNameOrFolderName, 'dir')==7
workingFolderPaths=fileNameOrFolderName;
[useless,workingFolderName]=fZfileParts(workingFolderPaths);
fileNameVarient1=fullfile(workingFolderPaths, [workingFolderName '.run']);
fileNameVarient2=fullfile(workingFolderPaths, ['.run']);
if exist(fileNameVarient1, 'file')==2
runFileFullName=fileNameVarient1;
elseif exist(fileNameVarient2, 'file')==2
runFileFullName=fileNameVarient2;else
warning(['(' mfilename '): there are no files ''.run'' or ''' [workingFolderName '.run'] ''' in the folder ''' workingFolderPaths ''''])
return
end
elseif exist(fileNameOrFolderName, 'file')==2
runFileFullName=fileNameOrFolderName;else
warning(['(' mfilename '): file or folder ' fileNameOrFolderName ' doesn''t exist'])
return
end
g = fopen(runFileFullName, 'r', 'l');mas = fread(g,  '*int8');fclose(g);
if length(mas)<616
warning(['(' mfilename '): file ''' runFileFullName ''' is too short'])
return
end
runFileData=DqNOAtji();
runFileHead=runFileData.runFileHead;scansStructTemplate=runFileData.scansStruct;
part_01=mas(1:512);part_01_name_full=part_01(1:256);part_01_path_full=part_01(257:end);
part_01_name_nonZeroLastIndex=find(part_01_name_full~=0, 1, 'last');
part_01_path_nonZeroLastIndex=find(part_01_path_full~=0, 1, 'last');
if ~isempty(part_01_name_nonZeroLastIndex)
part_01_name=part_01_name_full(1:part_01_name_nonZeroLastIndex);else
part_01_name=[];
end
if ~isempty(part_01_path_nonZeroLastIndex)
part_01_path=part_01_path_full(1:part_01_path_nonZeroLastIndex);else
part_01_path=[];
end
nameInsideRunFile=char(part_01_name)';
pathInsideRunFile=char(part_01_path)';part_02=mas(513:514);
part_02_uint16=typecast(part_02, 'uint16');numOfScansInsideRunFile=part_02_uint16;
part_03=mas(515:516);part_03_uint16=typecast(part_03, 'uint16');
numOfReferenceScansInsideRunFile=part_03_uint16;
part_03=mas(517:520);part_03_unknown=typecast(part_03, 'int32');part_04=mas(521:524);
part_04_int32=typecast(part_04, 'int32');totalNumberOfFrames=part_04_int32;
part_05=mas(525:526);part_05_uint16=typecast(part_05, 'uint16');
if part_05_uint16==1
activateReferenceFrames=part_05_uint16;
elseif part_05_uint16==0
activateReferenceFrames=part_05_uint16;else
activateReferenceFrames='unknown';
end
part_06=mas(527:528);part_06_uint16=typecast(part_06, 'uint16');
referenceRunsFrequency_1_per_X_dc_frames__X=part_06_uint16;
runFileHead.name=nameInsideRunFile;
runFileHead.path=pathInsideRunFile;runFileHead.number_of_scans=numOfScansInsideRunFile;
runFileHead.number_of_reference_scans=numOfReferenceScansInsideRunFile;
runFileHead.unknown_part_1=part_03_unknown;
runFileHead.total_number_of_frames=totalNumberOfFrames;
runFileHead.activate_reference_frames=activateReferenceFrames;
runFileHead.reference_runs_frequency_1_per_X_dc_frames__X=referenceRunsFrequency_1_per_X_dc_frames__X;
allSscansStructArray=scansStructTemplate;
allSscansStructArray(numOfScansInsideRunFile+numOfReferenceScansInsideRunFile)=scansStructTemplate;
numOfPreviousBytes=528;
for i=1:double(numOfScansInsideRunFile)+double(numOfReferenceScansInsideRunFile)
j=i-1;n=numOfPreviousBytes+j*88;
part_04=mas(n+1:n+2);part_04_uint16=typecast(part_04, 'uint16');
allSscansStructArray(i).scan_index=part_04_uint16;
part_05=mas(n+3:n+4);part_05_uint16=typecast(part_05, 'uint16');
if part_05_uint16==0
allSscansStructArray(i).scanned_axe='omega';
elseif part_05_uint16==4
allSscansStructArray(i).scanned_axe='phi';else
warning(['(' mfilename '): bad data in the file ''' runFileFullName ''' - parameter ''scanned_axe'', bytes ' num2str(n+3) ':' num2str(n+4)])
return
end
part_06=typecast(mas(n+5:n+8), 'int32');
allSscansStructArray(i).unknown_part_2=part_06;part_07=mas(n+9:n+16);
part_07_double=typecast(part_07, 'double');allSscansStructArray(i).omega=part_07_double;
part_08=mas(n+17:n+24);part_08_double=typecast(part_08, 'double');
allSscansStructArray(i).detector_thetaArm=part_08_double;part_09=mas(n+25:n+32);
part_09_double=typecast(part_09, 'double');allSscansStructArray(i).kappa=part_09_double;
part_10=mas(n+33:n+40);part_10_double=typecast(part_10, 'double');
allSscansStructArray(i).phi=part_10_double;part_11=mas(n+41:n+48);
part_11_double=typecast(part_11, 'double');allSscansStructArray(i).start=part_11_double;
part_12=mas(n+49:n+56);part_12_double=typecast(part_12, 'double');
allSscansStructArray(i).end=part_12_double;part_13=mas(n+57:n+64);
part_13_double=typecast(part_13, 'double');allSscansStructArray(i).width=part_13_double;
part_14=mas(n+65:n+72);part_14_double=typecast(part_14, 'double');
allSscansStructArray(i).unknown_part_3=part_14_double;
part_15=mas(n+73:n+76);part_15_uint32=typecast(part_15, 'uint32');
allSscansStructArray(i).to_do=part_15_uint32;part_16=mas(n+77:n+80);
part_16_uint32=typecast(part_16, 'uint32');allSscansStructArray(i).done=part_16_uint32;
part_17=mas(n+81:n+88);part_17_double=typecast(part_17, 'double');
allSscansStructArray(i).exposure=part_17_double;
end
experimentName=nameInsideRunFile;
numOfRuns=numOfScansInsideRunFile;numOfReferenceRuns=numOfReferenceScansInsideRunFile;
scansStruct=allSscansStructArray(1:numOfScansInsideRunFile);
referenceScansStruct=allSscansStructArray(1:numOfReferenceScansInsideRunFile);
runFileData=[];runFileData.runFileHead=runFileHead;runFileData.scansStruct=scansStruct;
runFileData.referenceScansStruct=referenceScansStruct;
runFileData.experimentName=experimentName;runFileData.numOfRuns=numOfRuns;
runFileData.numOfReferenceRuns=numOfReferenceRuns;isOk=true;
end
function runFileDataTemplate=DqNOAtji()
runFileHeadTemplate=struct(...
'name', '', 'path', '', 'number_of_scans',uint16(0),...
'number_of_reference_scans', uint16(0), 'unknown_part_1', int32(2),...
'total_number_of_frames', int32(0), 'activate_reference_frames', uint16(false),...
'reference_runs_frequency_1_per_X_dc_frames__X', uint16(false)...
);scansStructTemplate=struct(...
'scan_index', uint16(0), 'scanned_axe', 'omega', 'unknown_part_2', int32(0),...
'omega', double(0), 'detector_thetaArm', double(0), 'kappa', double(0),...
'phi', double(0), 'start', double(0), 'end', double(0),...
'width', double(0), 'unknown_part_3', double(0), 'to_do', uint32(0),...
'done', uint32(0), 'exposure',double(0)...
);
runFileDataTemplate=struct('runFileHead', runFileHeadTemplate, 'scansStruct', scansStructTemplate, ...
'referenceScansStruct', scansStructTemplate, 'experimentName', '',...
'numOfRuns', uint16(0), 'numOfReferenceRuns', uint16(0));
end
function [  ...
crackerData ,runFileData ,imageFormatOut ,...
imagesDataOut ,experimentFilesName ,isOk...
] =...
mYIMtqDYewJeExY( ...
currentDatasetFolderFullName ,currentDatasetName ,experimentNames ,...
additionalTools ,reduceSomeWarnings)
if exist('currentDatasetName', 'var')~=1
currentDatasetName = [];
end
if exist('experimentNames', 'var')~=1
experimentNames = [];
end
if exist('additionalTools', 'var')~=1
additionalTools = [];
end
if exist('reduceSomeWarnings', 'var')~=1
reduceSomeWarnings = false;
end
crackerData=[];
runFileData=[];imageFormatOut=[];imagesDataOut=[];experimentFilesName = [];isOk=false;
if isfield(additionalTools, 'cracker') && isfield(additionalTools.cracker, 'data2use') &&...
isfield(additionalTools, 'run') && isfield(additionalTools.run, 'data2use')
crackerData2use=additionalTools.cracker.data2use;
runData2use=additionalTools.run.data2use;
elseif isfield(additionalTools, 'run') && isfield(additionalTools.run, 'data2use')
crackerData2use=false;runData2use=additionalTools.run.data2use;else
crackerData2use=false;runData2use=false;
if reduceSomeWarnings && isempty(additionalTools)
else
disp([ 'Warning: parameter ''additionalTools'' is wrong. Using auto mode...' ]);
end
end
addChanges2Cracker=false;addChanges2Run=false;
if crackerData2use==3
addChanges2Cracker=true;crackerData2use=0;
end
if crackerData2use==3
addChanges2Run=true;crackerData2use=0;
end
if crackerData2use==0 && runData2use==0
[crackerData, runFileData, experimentFilesName, isOkCrackerAndRun]=ME6M8V8(currentDatasetFolderFullName, currentDatasetName ,reduceSomeWarnings);
elseif crackerData2use==0 && (runData2use==1 || runData2use==2)
[crackerData, runFileData, experimentFilesName, isOkCrackerAndRun]=Rmj5MxV(currentDatasetFolderFullName, currentDatasetName, additionalTools);
elseif (crackerData2use==1 || crackerData2use==2) && runData2use==0
[crackerData, runFileData, experimentFilesName, isOkCrackerAndRun]=ysNANhG15VdATeQ(currentDatasetFolderFullName, currentDatasetName, additionalTools);
elseif (crackerData2use==1 || crackerData2use==2) && (runData2use==1 || runData2use==2)
[crackerData, runFileData, isOkCrackerAndRun]=jjQ6En7g8Rh5N0Ib(additionalTools);
end
if ~isOkCrackerAndRun
return
end
if addChanges2Cracker
[crackerData, isOkCrackerAdd]=tfmEPB1AiHlnvT(crackerData, additionalTools);
if ~isOkCrackerAdd
return
end
end
if ~isstruct(experimentNames)
experimentNames=struct('imageFormat', 'unknown', 'imagesFolderType', {'auto'});
end
if isfield(experimentNames, 'imageFormat') && ~strcmpi(experimentNames.imageFormat, 'unknown')
imageFormatIn=experimentNames.imageFormat;else
imageFormatIn={'img', 'cbf', 'edf', 'esperanto'};
end
if ~isfield(experimentNames, 'imagesFolderType')
experimentNames.imagesFolderType='auto';
end
switch experimentNames.imagesFolderType
case 'experimentFolder'
[ imageFormat, imagesData, isOkImages]=...
YFHK9Sl1(...
currentDatasetFolderFullName,imageFormatIn, runFileData);case 'subfolder'
[ imageFormat, imagesData, isOkImages]=...
YFHK9Sl1(fullfile(...
currentDatasetFolderFullName,...
experimentNames.imagesFolderName),imageFormatIn, runFileData);case 'specialFolder'
[ imageFormat, imagesData, isOkImages]=...
YFHK9Sl1(...
experimentNames.imagesFolderName,imageFormatIn, runFileData);case 'auto'
[imageFormat, imagesData, isOkImages]=...
vGR5GxGwpw6LB0(...
currentDatasetFolderFullName, runFileData, imageFormatIn);otherwise
return
end
if isOkImages
for sc=1:length(imagesData)
imageFolder_inScan=imagesData(sc).imagesFolder;
imagesNames_inScan=imagesData(sc).imagesNames;
imagesData(sc).imagesFullNames=iOR5OPq(imageFolder_inScan, imagesNames_inScan);
end
imagesDataOut=imagesData;imageFormatOut=imageFormat;
if isfield(additionalTools, 'run')
if isfield(additionalTools.run, 'useImagesNamesFromTopToDown')
if additionalTools.run.useImagesNamesFromTopToDown
imagesDataOut.imagesNames=imagesDataOut.imagesNames(end:-1:1);
imagesDataOut.imagesFullNames=imagesDataOut.imagesFullNames(end:-1:1);
end
end
end
else
return
end
isOk=true;
end
function [crackerData, runFileData, experimentFilesName, ...
isOkCrackerAndRun]=ME6M8V8(...
currentDatasetFolderFullName, currentDatasetName ,reduceSomeWarnings)
crackerData=[];runFileData=[];warningsStruct=[];experimentFilesName=[];
isOkCrackerAndRun=false;crackerFileName1=[currentDatasetName '_cracker.par'] ;
crackerFileName2=[ '_cracker.par'] ;
runFileName1=[currentDatasetName '.run'] ;runFileName2=[ '.run'] ;
crackerFileFullName1=fullfile(currentDatasetFolderFullName, crackerFileName1);
crackerFileFullName2=fullfile(currentDatasetFolderFullName, crackerFileName2);
runFileFullName1=fullfile(currentDatasetFolderFullName, runFileName1);
runFileFullName2=fullfile(currentDatasetFolderFullName, runFileName2);
crackerAndRunExist=false;
if exist(crackerFileFullName1, 'file')==2 && exist(runFileFullName1, 'file')==2
crackerFileFullName=crackerFileFullName1;runFileFullName=runFileFullName1;
crackerAndRunExist=true;experimentFilesName=crackerFileName1(1:end-12);
elseif exist(crackerFileFullName2, 'file')==2 && exist(runFileFullName2, 'file')==2
crackerFileFullName=crackerFileFullName2;runFileFullName=runFileFullName2;
crackerAndRunExist=true;experimentFilesName=crackerFileName2(1:end-12);else
[cellArrayOfParFilesNames, numOfParFiles]=a9SljmjRJu(currentDatasetFolderFullName, 'par');
for i=1:numOfParFiles
crackerFileName3=cellArrayOfParFilesNames{i};
if length(crackerFileName3)>=12 && strcmpi(crackerFileName3(end-11:end), '_cracker.par')
fileNameShort=crackerFileName3(1:end-12);runFileName3=[fileNameShort '.run'];
crackerFileFullName3=fullfile(currentDatasetFolderFullName, crackerFileName3);
runFileFullName3=fullfile(currentDatasetFolderFullName, runFileName3);
if exist(runFileFullName3, 'file')==2
crackerFileFullName=crackerFileFullName3;
runFileFullName=runFileFullName3;crackerAndRunExist=true;
warningString=['Can''t find *_cracker.par file and *.run files with proper names'];
if ~reduceSomeWarnings
disp(warningString)
end
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString,0 );
warningString=['Using ''' crackerFileFullName3 ''''];
if ~reduceSomeWarnings
disp(warningString)
end
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString,0 );
warningString=['Using ''' runFileFullName3 ''''];
if ~reduceSomeWarnings
disp(warningString)
end
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString,0 );
experimentFilesName=crackerFileName3(1:end-12);break
end
end
end
end
if crackerAndRunExist
[crackerData, isOkCracker ]= F4TlX7u(crackerFileFullName);
[ runFileData, isOkRun ]= ODnsRaXF(runFileFullName);
isOkCrackerAndRun= all([isOkCracker isOkRun]);else
warningString=['cann''t find appropriate *_cracker.par and *.run files '];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );
end
end
function [crackerData, runFileData, experimentFilesName, isOkCrackerAndRun]=...
Rmj5MxV(...
currentDatasetFolderFullName, currentDatasetName, additionalTools)
crackerFileName1=[currentDatasetName '_cracker.par'] ;
crackerFileName2=[ '_cracker.par'] ;
crackerFileFullName1=fullfile(currentDatasetFolderFullName, crackerFileName1);
crackerFileFullName2=fullfile(currentDatasetFolderFullName, crackerFileName2);
isOkCrackerAndRun=false;warningsStruct=[];crackerExists=false;
if exist(crackerFileFullName1, 'file')==2
crackerFileFullName=crackerFileFullName1;
crackerExists=true;experimentFilesName=currentDatasetName;
elseif exist(crackerFileFullName2, 'file')==2
crackerFileFullName=crackerFileFullName2;crackerExists=true;experimentFilesName='';else
[cellArrayOfParFilesNames, numOfParFiles]=a9SljmjRJu(currentDatasetFolderFullName, 'par');
for i=1:numOfParFiles
crackerFileName3=cellArrayOfParFilesNames{i};
if length(crackerFileName3)>=12 && strcmpi(crackerFileName3(end-11:end), '_cracker.par')
crackerFileFullName3=fullfile(currentDatasetFolderFullName, crackerFileName3);
crackerFileFullName=crackerFileFullName3;
crackerExists=true;experimentFilesName=crackerFileName3(1:end-12);
warningString=['cann''t find *_cracker.par file with proper name'];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );
warningString=['using ''' crackerFileFullName3 ''''];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );break
end
end
end
if crackerExists
[crackerData, isOkCracker ]= F4TlX7u(crackerFileFullName);else
warningString=['cann''t find appropriate *_cracker.par file '];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );
end
if additionalTools.run.data2use==1
[ runFileData, isOkRun ]= ODnsRaXF(additionalTools.run.thirdPartyRunFileFullName);
elseif additionalTools.run.data2use==2
runFileData=[];runFileData.runFileHead=[];runFileData.runFileHead.number_of_scans=1;
runFileData.scansStruct=additionalTools.run;runFileData.numOfRuns=1;isOkRun=true;
end
isOkCrackerAndRun= all([isOkCracker isOkRun]);
end
function [crackerData, runFileData, experimentFilesName, isOkCrackerAndRun]=...
ysNANhG15VdATeQ(...
currentDatasetFolderFullName, currentDatasetName, additionalTools)
runFileName1=[currentDatasetName '.run'] ;runFileName2=[ '.run'] ;
runFileFullName1=fullfile(currentDatasetFolderFullName, runFileName1);
runFileFullName2=fullfile(currentDatasetFolderFullName, runFileName2);
isOkRun=false;warningsStruct=[];runExists=false;
if exist(runFileFullName1, 'file')==2
runFileFullName=runFileFullName1;runExists=true;experimentFilesName=currentDatasetName;
elseif exist(runFileFullName2, 'file')==2
runFileFullName=runFileFullName2;runExists=true;experimentFilesName='';else
[cellArrayOfRunFilesNames, numOfRunFiles]=a9SljmjRJu(currentDatasetFolderFullName, 'run');
for i=1:numOfRunFiles
runFileName3=cellArrayOfRunFilesNames{i};
if length(runFileName3)>=4 && strcmpi(runFileName3(end-3:end), '.run')
runFileFullName3=fullfile(currentDatasetFolderFullName, runFileName3);
runFileFullName=runFileFullName3;
runExists=true;experimentFilesName=runFileName3(1:end-4);
warningString=['cann''t find *.run file with proper name'];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );
warningString=['using ''' runFileFullName3 ''''];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );break
end
end
end
if runExists
[ runFileData, isOkRun ]= ODnsRaXF(runFileFullName);else
warningString=['cann''t find appropriate *_run.par file '];
warningsStruct=iJBkBsNQ8LyCT( warningsStruct, warningString );
end
if additionalTools.cracker.data2use==1
[crackerData, isOkCracker ]= F4TlX7u(additionalTools.cracker.thirdPartyCrackerFileFullName);
elseif additionalTools.run.data2use==2
crackerData=additionalTools.cracker;isOkCracker=true;
end
isOkCrackerAndRun= all([isOkCracker isOkRun]);
end
function [crackerData, runFileData, isOkCrackerAndRun]=...
jjQ6En7g8Rh5N0Ib(additionalTools)
if additionalTools.cracker.data2use==1
[crackerData, isOkCracker ]= F4TlX7u(additionalTools.cracker.thirdPartyCrackerFileFullName);
elseif additionalTools.cracker.data2use==2
crackerData=additionalTools.cracker;isOkCracker=true;
end
if additionalTools.run.data2use==1
[ runFileData, isOkRun ]= ODnsRaXF(additionalTools.run.thirdPartyRunFileFullName);
elseif additionalTools.run.data2use==2
runFileData=[];runFileData.runFileHead=[];runFileData.runFileHead.number_of_scans=1;
runFileData.scansStruct=additionalTools.run;runFileData.numOfRuns=1;isOkRun=true;
end
isOkCrackerAndRun= all([isOkCracker isOkRun]);
end
function [crackerData, isOkCracker]=...
tfmEPB1AiHlnvT(...
crackerDataFromFile, additionalTools)
crackerDef=crackerDataFromFile;crackerAdd=additionalTools.cracker;crackerData=crackerDef;
if isfield(crackerAdd, 'detector')
detector=crackerAdd.detector;
if isfield(detector, 'x0')
crackerData.detector.x0=crackerData.detector.x0+detector.x0;
end
if isfield(detector, 'y0')
crackerData.detector.y0=crackerData.detector.y0+detector.y0;
end
end
isOkCracker=true;
end
function [imageFormat, imagesData, isOkImages]=...
vGR5GxGwpw6LB0(...
folder, runFileData, imageFormats)
imageFormat=[];
imagesData=[];isOkImages=false;dectrisaliasesFileName='dectrisaliases.ini';
dectrisaliasesFileFullName=fullfile(folder, dectrisaliasesFileName);
if exist(dectrisaliasesFileFullName, 'file')==2
[imagesData, imageFormat, isOkDectrisaliases ]=...
naDBLREVD(dectrisaliasesFileFullName);
if isOkDectrisaliases
imagesData.imagesFolder=folder;isOkImages=true;return
end
end
imagesFolder=fullfile(folder,'frames');
if exist(imagesFolder, 'dir')==7
[imageFormat, imagesData, isOkImages]=...
k2uG9uAqeCHE(imagesFolder, imageFormats, runFileData);
if isOkImages
return
end
end
[imageFormat, imagesData, isOkImages]=...
k2uG9uAqeCHE(folder, imageFormats, runFileData);
end
function [ imageFormatOut, imagesData, isOkImages]=...
YFHK9Sl1(...
folder,imageFormats, runFileData)
imageFormatOut=[];imagesData=[];isOkImages=false;
[imageFormatOut, imagesData, isOkImages]=k2uG9uAqeCHE(folder, imageFormats, runFileData);
end
function [ R, isOk ] = MZTpYH9gfUM2Qh(rotationAxe, angleInDegrees)
R=[];isOk = 0;cs = cosd(angleInDegrees);sn = sind(angleInDegrees);
if rotationAxe==1
R=[   1   0   0;0  cs  sn;0 -sn  cs];
elseif rotationAxe==2
R=[  cs   0 -sn;0   1   0;sn   0  cs];
elseif rotationAxe==3
R=[  cs  sn   0;-sn  cs   0;0   0   1];else
warning( 'first input argument ''rotationAxe'' should be = 1, 2 or 3')
return
end
isOk = 1;
end
function [ R, isOk ] = cQQTjXPqQ8fajmy3( goniometerDataStruct )
g=goniometerDataStruct;
R=MZTpYH9gfUM2Qh(3,g.omega)*MZTpYH9gfUM2Qh(2,g.alpha)*MZTpYH9gfUM2Qh(3,g.kappa)*MZTpYH9gfUM2Qh(2,-g.alpha)*MZTpYH9gfUM2Qh(2,g.beta)*MZTpYH9gfUM2Qh(3,g.phi)*MZTpYH9gfUM2Qh(2,-g.beta);
isOk = 1;
end
function [ub, isok] = ITmX6D_( filefullname )
ub = [];isok = false;
if isempty(filefullname)
warning('input variable is empty')
return;
end
if ~ischar(filefullname)
warning('input type is incorrect')
return;
end
if exist(filefullname, 'file')~=2
warning(['file ''', filefullname, ''' doesn''t exist'])
return;
end
g = fopen(filefullname, 'r', 'l');file_stream_uint8 = fread(g,  1e3, '*uint8')';
fclose(g);file_length=length(file_stream_uint8);
if file_length<100
warning(['file size of the ''', filefullname, ''' is too small'])
return;
end
file_stream_char = char(file_stream_uint8);
cellStr=regexp(file_stream_char, ['(?:', sprintf('\r\n'), ')+'], 'split');
if ~strcmp(cellStr{1}, 'UB matrix:')
warning(['unknown record in the file ''' filefullname ''''])
return
end
string_=cellStr{2};[A, count, errmsg] = sscanf(string_, '%E%E%E%E%E%E%E%E%E', 9);
if ~isempty(errmsg) || count~=9
warning(['unknown header record in the string ' int2str(ind) ' of the file ''' filename ''''])
return
end
ub = reshape(A, [3, 3]);isok = true;
end
function [ warningsStruct ] = iJBkBsNQ8LyCT( varargin )
if nargin>=2
if isempty(varargin{1})
[ warningsStruct ]=yHmo34Dv2();else
warningsStruct=varargin{1};
end
warningString=varargin{2};
numOfWarnings=warningsStruct.numOfWarnings;numOfWarnings=numOfWarnings+1;
warningsStruct.warningsCellArray{numOfWarnings}=warningString;
warningsStruct.numOfWarnings=numOfWarnings;
if nargin==2
warning(warningString)
end
elseif nargin==0 || (nargin==1 && isempty(varargin{1}))
[ warningsStruct ]=yHmo34Dv2();
end
end
function [ warningsStruct ]=yHmo34Dv2()
warningsStruct=struct('warningsCellArray', {[]}, 'numOfWarnings', 0);
warningsStruct.warningsCellArray=cell(1,100);
end