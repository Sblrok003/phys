function [I, cbfHead, isOk] = fZcbfRead( cbfFileName, varargin )
% reads *.cbf file and returns 2D array of the intensities

use_java_decompressor = 1; % matlab reads file & analizes the header, java decompresses the data



I=[];
cbfHead=[];
isOk=false;


%  check java class >>>
javaclassexists = false;
if use_java_decompressor
    if exist('com.company.Cbf', 'class')==8
        javaclassexists = true;
    else
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
%  check java class <<<


if nargin>1 && ischar(varargin{1}) && ...
        (strcmpi(varargin{1}, 'minimal') || strcmpi(varargin{1}, 'min') || strcmpi(varargin{1}, 'm') )
    useminimalheader = true;
else
    useminimalheader = false;
end


% checks the existanse of the file
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
masInt8 = fread(g,  '*int8');
mas=double(masInt8);
fclose(g);

lmas=length(mas);
% detector
% fd=1475;
% sd=1679;
% first Element of the first string

% identifier of the binary part
% idenB1=(12);
% idenB2=([26;4;-43]);

% looking for the identifier
isBinExist=false;
isTextExist=false;
k=0;

binIdentifierTale=[26;4;-43];

while k<(lmas-3)
    k=k+1;
    if mas(k)==12
        if all(mas(k+1:k+3)==binIdentifierTale)
            isBinExist=true;
            if k>1
                isTextExist=true;
                lastTextSymbolPosition=k-1;
            end
            k=k+3;
            break;
        end
    end
end

% 'k' was put at the end of the bin Identifier

% checks: does identifier exist
if ~isTextExist
    warning('cann''''t find text header');
    return;
end
if ~isBinExist
    warning('cann''''t find any binary identifier');
    return;
end

if useminimalheader
    cbfHead=fZcbfHeadAnalyse_minimal(mas(1:lastTextSymbolPosition));
else
    cbfHead=fZcbfHeadAnalyse(mas(1:lastTextSymbolPosition));
end

fd=cbfHead.X_Binary_Size_Fastest_Dimension;
sd=cbfHead.X_Binary_Size_Second_Dimension;

if isnan(fd) || isnan (sd) || isnan(cbfHead.X_Binary_Size) ||...
        isempty(fd) || isempty (sd) || isempty(cbfHead.X_Binary_Size)
    warning([filename ': corrupted file header'])
    return
end

% java >>>
if use_java_decompressor
    K = com.company.Cbf;
    K.putcompressedstream(masInt8(k+1:end),fd, sd, cbfHead.X_Binary_Size);
    isOkDecode = K.getstatus();
    if ~isOkDecode
        warning(K.getmessage()');
        return
    end
    I=reshape(K.getvalues,[fd, sd]);
    isOk = true;
    return
end
% java <<<

if double(fd*sd)*1.03<double(cbfHead.X_Binary_Size)
    %I  = fZcbfRead_decompressor_old_ver(mas, k, fd, sd);
    %isOk_decomp=true;
    [I, isOk_decomp]  = fZcbfRead_decompressor_third_party(mas(k+1:end), cbfHead.X_Binary_Size, fd, sd);
    isOk=isOk_decomp;
    return
end


% Build arrays

x=find(mas==-128);
lx=length(x);
if lx==0
%     I_delta=mas(k+1:k+sd*fd);
    I=reshape(cumsum((mas(k+1:k+sd*fd))), fd, sd);
    isOk=true;
    return
end

i=find(x>k, 1, 'first');
if isempty(i)
%     I_delta=mas(k+1:k+sd*fd);
    I=reshape(cumsum((mas(k+1:k+sd*fd))), fd, sd);
    isOk=true;
    return
end
% i=0
i=i-1;

xLogical=true(1,lmas);
xLogical(x)=false;
xLogical(1:k)=false;

int16_indexs=zeros(1,lx);
int16_numbers=zeros(1,2*lx);
int16_ind=0;
int32_indexs=zeros(1,lx);
int32_numbers=zeros(1,4*lx);
int32_ind=0;


% fff=masInt8(k+1:k+cbfHead.X_Binary_Size);
% md5str=fZmd5(fff, 'base64');
% disp(md5str)
% disp(cbfHead.MD5)



while i<=(lx-1)
    i=i+1;
    curIndex=x(i);
    if curIndex-6*int32_ind-2*int16_ind-k>sd*fd
        disp('Warning: Last Element read (fZcbfRead.m)')
        break
    else
    end
    if lx>=i+1 && (x(i+1)==curIndex+2 && mas(curIndex+1)==0)
        %int 32
        int32_numbers(int32_ind*4+1)=( curIndex+3 );
        int32_numbers(int32_ind*4+2)=( curIndex+4 );
        int32_numbers(int32_ind*4+3)=( curIndex+5 );
        int32_numbers(int32_ind*4+4)=( curIndex+6 );
        int32_indexs(int32_ind+1)=x(i)-6*int32_ind-2*int16_ind-k;
        int32_ind=int32_ind+1;
        xLogical(curIndex+1) = false;
        xLogical(curIndex+2) = false;
        xLogical(curIndex+3) = false;
        xLogical(curIndex+4) = false;
        xLogical(curIndex+5) = false;
        xLogical(curIndex+6) = true;
        i=i+1;
        curIndex=x(i);
        for j=1:4
            if lx>=i+1 && x(i+1)-curIndex<5
                i=i+1;
            else
                break
            end
        end
    else
        % int 16
        int16_numbers(int16_ind*2+1)=( curIndex+1 );
        int16_numbers(int16_ind*2+2)=( curIndex+2 );
        int16_indexs(int16_ind+1)=x(i)-6*int32_ind-2*int16_ind-k;
        int16_ind=int16_ind+1;
        xLogical(curIndex+1) = false;
        xLogical(curIndex+2) = true;
        if lx>=i+1 && x(i+1)-curIndex<3
            i=i+1;
        end
        if lx>=i+1 && x(i+1)-curIndex<3
            i=i+1;
        end
    end
end


xLogical((k+fd*sd+2*int16_ind+6*int32_ind+1):end)=false;

% disp(['bytes: ' num2str(fd*sd+2*int16_ind+6*int32_ind)])

I_delta=mas(xLogical);




% typecast int16
if int16_ind>0
    int16_numbers=int16_numbers(1:int16_ind*2);
    int16indexs=int16_indexs(1:int16_ind);
    I_2=double(typecast(int8(mas(int16_numbers)), 'int16'));
    % building 2D array
    I_delta(int16indexs)=I_2;
end

% typecast int32
if int32_ind>0
    int32_numbers=int32_numbers(1:int32_ind*4);
    int32indexs=int32_indexs(1:int32_ind);
    I_4=double(typecast(int8(mas(int32_numbers)), 'int32'));
    % building 2D array
    I_delta(int32indexs)=I_4;
end


% disp(num2str(size(I_delta)))
% fd*sd
I=reshape(cumsum((I_delta)), fd, sd);


isOk = true;


end





function  cbfHead  = fZcbfHeadAnalyse( mas )
    % Analyse cbf head
    
    %cellStr=strsplit(char(mas'), '\r\n', 'CollapseDelimiters',true);
    %cellStr=regexp(char(mas'), 'r\n\', 'split');
    cellStr=regexp(char(mas'), ['(?:', sprintf('\r\n'), ')+'], 'split');
    %cellStr=regexp(char(mas'), '(?:r\n\)+', 'split');
    %cellStr
    
    k=length(cellStr);
    
    % fndA cell=>>five columns:
    % 1: 'identifier string'
    % 2: 'length of the iden. str.' 
    % 3: 'value'
    % 4: 'post value symbols (can be units)'
    % 5: 'format'
    % 6: 'field in the struct'
    
    % sets default format as 'value (numeric) & units (char)'
    fndA=cell(100,6);
    for i=1:size(fndA, 1)
        fndA{i,5}='%f %s';
    end
    
    
    n=1;
    fndA{n,1}='# Detector: ';
    fndA{n,6}='Detector';
    fndA{n,5}='%s';
    n=n+2;                    % << n+2 - reserved for the string 'Data & time'
    fndA{n,1}='# Pixel_size';
    fndA{n,6}='Pixel_size';
    fndA{n,5}='%f m x %f %s';
    n=n+1;
    fndA{n,1}='# Exposure_time';
    fndA{n,6}='Exposure_time';
    n=n+1;
    fndA{n,1}='# Exposure_period';
    fndA{n,6}='Exposure_period';
    n=n+1;
    fndA{n,1}='# Wavelength';
    fndA{n,6}='Wavelength';
    n=n+1;
    fndA{n,1}='# Flux';
    fndA{n,6}='Flux';
    n=n+1;
    fndA{n,1}='X-Binary-Size-Fastest-Dimension:';
    fndA{n,6}='X_Binary_Size_Fastest_Dimension';
    n=n+1;
    fndA{n,1}='X-Binary-Size-Second-Dimension:';
    fndA{n,6}='X_Binary_Size_Second_Dimension';
    n=n+1;
    fndA{n,1}='# Start_angle';
    fndA{n,6}='Start_angle';
    n=n+1;
    fndA{n,1}='# Angle_increment';
    fndA{n,6}='Angle_increment';
    n=n+1;
    fndA{n,1}='# Omega ';
    fndA{n,6}='Omega';
    n=n+1;
    fndA{n,1}='# Omega_increment';
    fndA{n,6}='Omega_increment';
    n=n+1;
    fndA{n,1}='# Phi ';
    fndA{n,6}='Phi';
    n=n+1;
    fndA{n,1}='# Phi_increment';
    fndA{n,6}='Phi_increment';
    n=n+1;
    fndA{n,1}='# Kappa';
    fndA{n,6}='Kappa';
    n=n+1;
    fndA{n,1}='# Oscillation_axis';
    fndA{n,6}='Oscillation_axis';
    fndA{n,5}='%s';
    n=n+1;
    fndA{n,1}='# Temperature';
    fndA{n,6}='Temperature';
    n=n+1;
    fndA{n,1}='# Blower';
    fndA{n,6}='Blower';
    n=n+1;
    fndA{n,1}='# Detector_distance';
    fndA{n,6}='Detector_distance';
    n=n+1;
    fndA{n,1}='# Detector_Voffset';
    fndA{n,6}='Detector_Voffset';
    n=n+1;
    fndA{n,1}='X-Binary-Size:';
    fndA{n,6}='X_Binary_Size';
    n=n+1;
    fndA{n,1}='# Beam_xy';
    fndA{n,6}='Beam_xy';
    fndA{n,5}='(%f, %f) %s';
    n=n+1;
    fndA{n,1}='Content-MD5:';
    fndA{n,6}='MD5';
    fndA{n,5}='%s';
    fndA(n+1:end,:)=[];
    
    for i=1:n
        fndA{i, 2}=length(fndA{i, 1});
        if strncmp(fndA{i,5},'%s',2)
            fndA{i, 3}='';
        else
            fndA{i, 3}=nan;
        end
    end
    
    
    % parsing >>>
    for i=1:k
        m=cellStr{i};
        fndA{1, 3}='';
        fndA{2, 3}='';
        %looking for the 'Detector' string
        if strncmp(fndA{1, 1}, m, fndA{1, 2})
            c=m(fndA{1, 2}+1:end);
            fndA{1, 3}=c;
            if k>i
                % checks whether the string after the 'Detector' string contains
                % the date and time
                m=cellStr{i+1};
                if length(m)==28 && strcmp(m(7),'-') && strcmp(m(10),'-')...
                        && strcmp(m(13),'T') && strcmp(m(16),':') &&...
                         strcmp(m(19),':') && strcmp(m(22),'.')
                     c=m(3:end);
                     fndA{2, 3}=c;
                else
                    strL=length(m);
                    if strL>14 && strcmp(m(end-3), '.')  &&...
                            strcmp(m(end-6), ':') && strcmp(m(end-9), ':') ...
                            && strcmp(m(1:2),'# ')
                        c=m(3:end);
                        fndA{2, 3}=c;
                    end
                end
            end
            break
        end
    end
%   
    % II:
    for i=1:k
        m=cellStr{i};
        for j=1:n
            if strncmp(fndA{j, 1}, m, fndA{j, 2})
                c=textscan(m(fndA{j, 2}+1:end), fndA{j, 5},'delimiter','','CollectOutput',1);
                if ~isempty(c)
                    if ~iscell(c{1})
                        fndA{j, 3}=c{1};
                    else
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
    fndA{2,6}='Data';
    cbfHead=cell2struct(fndA(:,3)',fndA(:,6)',2);
end

function  cbfHead  = fZcbfHeadAnalyse_minimal( mas )
% Analyse cbf head



k = 0;
for k=1:numel(mas)
    if mas(k)==88
        if mas(k+1)==45 && mas(k+2)==66
            break
        end
    end
end

M = mas(k:end)';

l = numel(M);

u=find(M==88);

d = 10.^(9:-1:0);

b = [88 45 66 105 110 97 114 121 45 83 105 122 101 58 32];
n = 15;
for i=1:numel(u)
    if all(M(u(i):u(i)+n-1)==b)
        for j=u(i)+n:l
            if M(j)<48 || M(j)>57
                break
            end
        end
        X_Binary_Size = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));
        break
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
        X_Binary_Size_Fastest_Dimension = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));
        break
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
        X_Binary_Size_Second_Dimension = sum(((M(u(i)+n:j-1))-48).*d(10-j+n+1+u(i):end));
        break
    end
end

cbfHead = struct(...
    'X_Binary_Size_Fastest_Dimension'   ,X_Binary_Size_Fastest_Dimension  ,...
    'X_Binary_Size_Second_Dimension'    ,X_Binary_Size_Second_Dimension   ,...
    'X_Binary_Size'                     ,X_Binary_Size);
   
end



function  [I]  = fZcbfRead_decompressor_old_ver(mas_Double, k, fd, sd)
    % Build arrays
%     I=[];
%     isOk=false;
    
    I_delta=zeros(1,fd*sd);
    % I_2=zeros(sd*fd*2, 1, 'int32');
    % I_4=zeros(sd*fd*4, 1, 'int32');
    % I_2_i=zeros(sd*fd, 1);
    % I_4_i=zeros(sd*fd, 1);
    % I_2_j=zeros(sd*fd, 1);
    % I_4_j=zeros(sd*fd, 1);
    
    I_2=zeros(2e3, 1);
    I_4=zeros(4e3, 1);
    I_2_i=zeros(1e3, 1);
    I_4_i=zeros(1e3, 1);
    
    
    k_int16=0;
    k_int32=0;
    k_int16_2=0;
    k_int32_4=0;
    
    % analyses bytes in the array 'mas_Double'
    for i=1:sd*fd
        k=k+1;
        if mas_Double(k)==-128;
            k=k+1;
            a1=mas_Double(k);
            k=k+1;
            a2=mas_Double(k);
            if a1==0 && a2==-128
                k_int32=k_int32+1;
                k_int32_4=k_int32_4+4;
                k=k+1;
                a1=mas_Double(k);
                k=k+1;
                a2=mas_Double(k);
                k=k+1;
                a3=mas_Double(k);
                k=k+1;
                a4=mas_Double(k);
                I_4(k_int32_4-3)=a1;
                I_4(k_int32_4-2)=a2;
                I_4(k_int32_4-1)=a3;
                I_4(k_int32_4  )=a4;
                I_4_i(k_int32)=i;
            else
                k_int16=k_int16+1;
                k_int16_2=k_int16_2+2;
                I_2(k_int16_2-1)=a1;
                I_2(k_int16_2  )=a2;
                I_2_i(k_int16)=i;
            end
        else
            % building 2D array
            I_delta(i)=mas_Double(k);
        end
    end
    
    % typecast int16
    if k_int16>0
        I_2=I_2(1:k_int16_2);
        I_2_i=I_2_i(1:k_int16);
        %     I_2_j=I_2_j(1:k_int16);
        I_2=double(typecast(int8(I_2), 'int16'));
        % building 2D array
        for i=1:k_int16
            I_delta(I_2_i(i))=I_2(i);
        end
    end
    
    
    % typecast int32
    if k_int32>0
        I_4=I_4(1:k_int32_4);
        I_4_i=I_4_i(1:k_int32);
        %     I_4_j=I_4_j(1:k_int32);
        I_4=double(typecast(int8(I_4), 'int32'));
        % building 2D array
        for i=1:k_int32
            I_delta(I_4_i(i))=I_4(i);
        end
    end
    
    
    I=reshape(cumsum((I_delta)), fd, sd);
    
    isOk = 1;
end

function  [I, isok]  = fZcbfRead_decompressor_third_party(mas_Double, X_Binary_Size, fd, sd)    
	I = zeros(fd,sd);
    isok = false;
    
    dat_in=typecast(int8(mas_Double), 'uint8');

    ind_out = 1;
    ind_in = 1;
    val_curr = 0;
    %val_diff = 0;
    while (ind_in <= X_Binary_Size)
        val_diff = double(dat_in(ind_in));
        ind_in = ind_in +1;
        if (val_diff ~= 128)
            % if not escaped as -128 (0x80=128) use the current byte as
            % difference, with manual complement to emulate the sign
            if (val_diff >= 128)
                val_diff = val_diff - 256;
            end
        else
            % otherwise check for 16-bit integer value
            if ((dat_in(ind_in) ~= 0) || (dat_in(ind_in+1) ~= 128))
                % if not escaped as -32768 (0x8000) use the current 16-bit integer
                % as difference
                val_diff = double(dat_in(ind_in)) + ...
                    256 * double(dat_in(ind_in+1));
                % manual complement to emulate the sign
                if (val_diff >= 32768)
                    val_diff = val_diff - 65536;
                end
                ind_in = ind_in +2;
            else
                ind_in = ind_in +2;
                % if everything else failed use the current 32-bit value as
                % difference
                val_diff = double(dat_in(ind_in)) + ...
                    256 * double(dat_in(ind_in+1)) + ...
                    65536 * double(dat_in(ind_in+2)) + ...
                    16777216 * double(dat_in(ind_in+3));
                % manual complement to emulate the sign
                if (val_diff >= 2147483648)
                    val_diff = val_diff - 4294967296;
                end
                ind_in = ind_in +4;
            end
        end
        val_curr = val_curr + val_diff;
        I(ind_out) = val_curr;
        ind_out = ind_out +1;
    end
    
    if (ind_out-1 ~= fd*sd)
        warning(filename,[ 'mismatch between ' num2str(ind_out-1) ...
            ' bytes after decompression with ' num2str(fd*sd) ...
            ' expected' ]);
        return
    end
    
    isok=true;
end

