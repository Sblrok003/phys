%%%experiments info file v0.7


%--------------------------------------------------------------------------





% PZT2.4% 2018.12.05 cooling 0kV:

experimentNames.experimentIdentifierString='PZT2.4 N03b (ramp_cool_0kV) sum by temperature';
experimentNames.mainFolder = 'D:\Политех учёба\LAB\for student\PZT2.4 N03b (ramp_cool_0kV) sum by temperature\';
experimentNames.folderNameTemplate = '*';
experimentNames.imageFormat='cbf';

experimentInfo.date.year=2018;
experimentInfo.date.month=12;
experimentInfo.date.day=05;
experimentInfo.sampleName='PZT2.4';
experimentInfo.sampleNameInLatex='PbZr_{97.6}Ti_{2.4}O_{3}';
experimentInfo.comments='ramp, cool, 0kV';

backgroundWindow.bgWindowUpperLeftCornerCoordinates=[761 1601];
backgroundWindow.bgWindowBottomRightCornerCoordinates=[769 1610];
backgroundWindow.bgWindowCoordinateSystem='snblAlbula';
backgroundWindow.defaultFlux=1e5;

multiDataset.multiDatasetValues=[331:-5:256];
%multiDataset.typeIndexInName = true;
multiDataset.indexDigits = 3;

detectorImageStoreGeometry.auto = false;
detectorImageStoreGeometry.imageFastestDimentionOrientation='columns';
detectorImageStoreGeometry.firstPixelPosition='upper right corner';

