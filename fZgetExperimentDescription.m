function [experimentMetadata, isOk]= fZgetExperimentDescription(inputArgument, varargin)
experimentMetadata=[];isOk=false;
if numel(varargin)==1
fileOrFolderName=varargin{1};else
fileOrFolderName='';
end
if ischar(inputArgument) && ~isempty(inputArgument) &&...
CxyQ4mAVhnczJ(inputArgument)
experimentIdentifierString=inputArgument;
[experimentMetadata, isOk]=BXJCxUz(experimentIdentifierString, fileOrFolderName);
if ~isOk
warning(['cann''t find the ''experimentIdentifierString'' in the experiment description catalogue'])
return
end
elseif isstruct(inputArgument) && numel(inputArgument)==1
quickReconstruction=inputArgument;
[experimentMetadata, isOk]=J3vOWWQVFccBo(quickReconstruction);else
warning(['wrong first input argument'])
return
end
end
function [experimentMetadata, isOkFound]=BXJCxUz(experimentIdentifierString, fileOrFolderName)
experimentMetadata=[];isOkFound=false;isFolder=false;
if isempty(fileOrFolderName) || ~ischar(fileOrFolderName) ||...
~isvector(fileOrFolderName) ||...
size(fileOrFolderName,1)~=1
folderName=pwd;isFolder=true;else
if exist(fileOrFolderName, 'dir')==7
isFolder=true;folderName=fileOrFolderName;
elseif exist(fileOrFolderName, 'file')==2
fileName=fileOrFolderName;else
warning(['file or folder ''' fileOrFolderName ''' doesn''t exist'])
return
end
end
if ~isFolder
[experimentMetadata, isOkFound, fileIs_properInfoFile ] =...
yVmTFgzw7VYJVii(fileName, experimentIdentifierString );
if ~isOkFound
if ~fileIs_properInfoFile
warning(['file ''' fileName ''' is corrupted'])
else
warning(['can''t find entry ''' experimentIdentifierString ''' in the file ''' fileName ''''])
end
end
else
[filesList,numOfFiles]=XzhzdAAKXt(folderName,{'*.m','*.txt'});
filesFullNamesList=IoSUJY43Y(folderName, filesList);
for f=1:numOfFiles
fileName=filesFullNamesList{f};[experimentMetadata, isOkFound ] =...
yVmTFgzw7VYJVii(fileName, experimentIdentifierString );
if isOkFound
break
end
end
if ~isOkFound
disp(['cann''t find entry ''' experimentIdentifierString ''' in any file within the folder ''' folderName ''''])
end
end
end
function [experimentMetadata, isOk]=...
J3vOWWQVFccBo(quickReconstruction)
isOk=false;qR=quickReconstruction;
if ~isstruct(qR)
disp(['ERROR (' mfilename '): input argument is wrong'])
return
end
if ~isfield(qR, {'experimentIdentifierString', 'mainFolder'})
disp(['ERROR (' mfilename '): input argument is wrong'])
return
end
if ~ischar(qR.experimentIdentifierString) || ~ischar(qR.mainFolder) ||...
size(qR.experimentIdentifierString, 1)~=1 || size(qR.mainFolder, 1)~=1
disp(['ERROR (' mfilename '): input argument is wrong'])
return
end
defaultStructs=DriASXJE();experimentMetadata=defaultStructs;
experimentMetadata.experimentNames.experimentIdentifierString=qR.experimentIdentifierString;
experimentMetadata.experimentNames.mainFolder=qR.mainFolder;
experimentMetadata.multiDataset.thereIsOnlyOneFolderInDataset=true;
if isfield(qR, 'multiDatasetValue')
experimentMetadata.multiDataset.multiDatasetValues=qR.multiDatasetValue;
end
isOk=true;
end
function [defaultStructs]=DriASXJE()
experimentInfoStr=struct(...
'date', struct('year', 0, 'month', 0, 'day', 0, 'hour', 0, 'minute', 0), ...
'sampleName', 'Unknown', ...
'sampleNameInLatex', 'Unknown', ...
'comments', '');experimentNamesStr=struct(...
'experimentIdentifierString', '', ...
'mainFolder', '', ...
'additionalExperimentSubPath', '', ...
'folderNameTemplate', '', ...
'imageFormat', 'auto', ...
'imagesFolderType', 'auto', ...
'imagesFolderName', '', ...
'comments', '');backgroundWindowStr=struct(...
'bgWindowUpperLeftCornerCoordinates', [1 1], ...
'bgWindowBottomRightCornerCoordinates', [2 2], ...
'bgWindowCoordinateSystem', 'snblAlbula',...
'defaultFlux', 1);multiDatasetStr=struct(...
'thereIsOnlyOneFolderInDataset', false, ...
'multiDatasetValues', nan, ...
'digitsAreDynamic', false, ...
'digits', 3, ...
'typeIndexInName', false, ...
'indexSuffix', '', ...
'indexDigitsAreDynamic', false, ...
'indexDigits', 2, ...
'multiDatasetType', 'temperature', ...
'multiDatasetTypeShort', 'T', ...
'units', 'C');detectorImageStoreGeometry=struct(...
'auto', true,...
'imageFastestDimentionOrientation', '', ...
'firstPixelPosition', '');defaultStructs=struct(...
'experimentInfo', experimentInfoStr, ...
'experimentNames', experimentNamesStr, ...
'backgroundWindow', backgroundWindowStr, ...
'multiDataset', multiDatasetStr, ...
'detectorImageStoreGeometry', detectorImageStoreGeometry);
end
function [cellArrayOfNames, listLength, list, isOk ] = XzhzdAAKXt( folderPath, varargin )
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
function f = IoSUJY43Y(varargin)
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
function [ versionStruct ] = j7nNGjREJ()
versionStruct=struct(...
'releaseversion'         ,0.71                  ,...
'releasedate'           ,'2020.05.02'           ,...
'matlabversion'         ,version()              ,...
'matlabrelease'         ,version('-release')    ,...
'matlabreleasedate'     ,version('-date')       ,...
'java'                  ,version('-java')         );
end
function [ isrow_ok ] = CxyQ4mAVhnczJ( array )
isrow_ok=false;
if isvector(array) &&...
size(array,1)==1
isrow_ok=true;
end
end
function [ experimentMetadata, isOkFound, fileIs_properInfoFile ] =...
yVmTFgzw7VYJVii(filename, identifierString )
experimentMetadata=[];isOkFound=false;[mas, fileIs_properInfoFile]=IZ0LfH2w(filename);
if ~fileIs_properInfoFile
return
end
fileIs_properInfoFile=true;[cellStrClean, stringsIndexesInOriginalFile]=YlLM2Hobm(mas);
[experimentsArray, stringsIndexesCellArray]=...
mC0Xp4R_0x(...
cellStrClean, stringsIndexesInOriginalFile);
[experimentIndex, experimentIdentifierWasFound]=...
Z0Qqyqo(...
experimentsArray, stringsIndexesCellArray, identifierString, filename);
if ~experimentIdentifierWasFound
return
end
if experimentIdentifierWasFound
strings=experimentsArray{experimentIndex};
stringsIndexes=stringsIndexesCellArray{experimentIndex};
[experimentMetadata, isOkFound]=...
iYzlx_x1e3VM(strings, stringsIndexes, filename);
end
end
function [mas, fileIs_properInfoFile]=IZ0LfH2w(filename)
fileIs_properInfoFile=false;
if exist(filename, 'file')~=2
return
end
g = fopen(filename, 'r', 'l');mas = fread(g,  '*uint8');fclose(g);
if numel(mas)<100
return
end
versionIsOk=de35fh5kwp(mas(1:100), filename);
if ~versionIsOk
return
end
fileIs_properInfoFile=true;
end
function versionIsOk=de35fh5kwp(masFirstSymbols, filename)
versionIsOk=false;currentProgramVersion=j7nNGjREJ();
if all(masFirstSymbols(4:26)'=='experiments info file v')
masFirstSymbols=masFirstSymbols(27:end);
newlineIndex=find(masFirstSymbols==13 | masFirstSymbols==10, 1, 'first');
version_char=masFirstSymbols(1:(newlineIndex-1));
version_double=str2double(char(version_char'));
if ~isempty(version_double) && isnumeric(version_double) &&...
isscalar(version_double) && isfinite(version_double) &&...
isreal(version_double) && version_double>0
if currentProgramVersion.releaseversion>=version_double
versionIsOk=true;else
disp(['current program version is ' num2str(currentProgramVersion.releaseversion)])
disp(['you should use version ' num2str(version_double) ' or later to read file'])
disp(['''' filename ''''])
end
end
end
end
function [cellStrClean, stringsIndexesInOriginalFile]=...
YlLM2Hobm(mas)
cellStr=regexp(char(mas'), ['(?:', sprintf('\r\n|\n'), ')'], 'split');
numOfStrings=length(cellStr);goodStrings=false(numOfStrings, 1);
for s=1:numOfStrings
cellStr{s}=strtrim(cellStr{s});
end
for s=1:numOfStrings
str=cellStr{s};
if ~isempty(str) && ...
(...
(str(1)>96 && str(1)<123) ||...
(str(1)>64 && str(1)<91)...
)
goodStrings(s)=true;
end
end
stringsIndexesInOriginalFile=(1:numOfStrings)';
stringsIndexesInOriginalFile=stringsIndexesInOriginalFile(goodStrings);
cellStrClean=cellStr(goodStrings);
end
function [experimentsArray, stringsIndexesCellArray]=...
mC0Xp4R_0x(...
cellStrClean, stringsIndexesInOriginalFile)
numOfCleanStrings=numel(cellStrClean);experimentEntries=false(numOfCleanStrings, 1);
for s=1:numOfCleanStrings
if strncmp(cellStrClean{s}, 'experimentNames.experimentIdentifierString', 42) ||...
strncmp(cellStrClean{s}, 'experimentIdentifierString', 26)
experimentEntries(s)=true;
end
end
numOfExperiments=sum(experimentEntries);experimentsArray=cell(numOfExperiments, 1);
stringsIndexesCellArray=cell(numOfExperiments, 1);
experimentIndexes=find(experimentEntries);
for s=1:numOfExperiments
expfirstString=experimentIndexes(s);
if s~=numOfExperiments
expLastString=experimentIndexes(s+1)-1;else
expLastString=numOfCleanStrings;
end
experimentsArray{s}=cellStrClean(expfirstString:expLastString);
stringsIndexesCellArray{s}=stringsIndexesInOriginalFile(expfirstString:expLastString);
end
end
function [experimentMetadata, isOkFound]=...
iYzlx_x1e3VM(strings, stringsIndexes, filename)
experimentMetadata=[];isOkFound=false;numOfStringsInBlock=numel(strings);
defaultStructs=DriASXJE();experimentInfo=defaultStructs.experimentInfo;
experimentNames=defaultStructs.experimentNames;
backgroundWindow=defaultStructs.backgroundWindow;
multiDataset=defaultStructs.multiDataset;
detectorImageStoreGeometry=defaultStructs.detectorImageStoreGeometry;
for s=1:numOfStringsInBlock
try
str=strings{s};evalc(str);catch
strIndex=stringsIndexes(s);disp(['Error in file ''' filename ''''])
disp(['in string N ' int2str(strIndex) ':'])
disp(str)
return
end
end
experimentMetadata=struct(...
'experimentInfo', experimentInfo,...
'experimentNames', experimentNames,...
'backgroundWindow', backgroundWindow,...
'multiDataset', multiDataset,...
'detectorImageStoreGeometry', detectorImageStoreGeometry...
);isOkFound=true;
end
function [experimentIndex, experimentIdentifierWasFound]=...
Z0Qqyqo(...
experimentsArray, stringsIndexesCellArray, identifierString, filename)
experimentIndex=[];
experimentIdentifierWasFound=false;numOfExperiments=numel(experimentsArray);
for ex=1:numOfExperiments
try
str=experimentsArray{ex}{1};evalc(str);catch
strIndex=stringsIndexesCellArray{ex}(1);disp(['Error in file ''' filename ''''])
disp(['in string N ' int2str(strIndex) ':'])
disp(str)
return
end
if exist('experimentNames', 'var')==1 &&...
isstruct(experimentNames) &&...
isfield(experimentNames, 'experimentIdentifierString')...
experimentIdentifierString=experimentNames.experimentIdentifierString;
end
if exist('experimentIdentifierString', 'var')==1 &&...
ischar(experimentIdentifierString) &&...
isvector(experimentIdentifierString) &&...
size(experimentIdentifierString,1)==1
experimentIdentifierString=strtrim(experimentIdentifierString);else
continue
end
if strcmp(experimentIdentifierString, identifierString)
experimentIdentifierWasFound=true;experimentIndex=ex;break
end
end
end