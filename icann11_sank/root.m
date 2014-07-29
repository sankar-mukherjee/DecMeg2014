close all
clear all
clc

%% Settings (modify these)

% Location of the ICANN'11 competition data
dataPath = '..\..\python\data';
% subject=[01;02;03;04;05;06;07;08;09;10;11;12;13;14;15;16];
subject=[17;18;19;20;21;22;23];

% Select the set of initial features. Available are:
%     1 - Mean (= detrended mean)
%     2 - Slope of fitted line
%     3 - Detrended variance
%     4 - Detrended standard deviation
%     5 - Detrended skewness
%     6 - Detrended kurtosis
%     7 - Variance
%     8 - Standard deviation
%     9 - Skewness
%    10 - Kurtosis
%    11 - Fluctuation around mean
generatedFeatures = [4,10]; % Mean and detrended standard deviation
% generatedFeatures = 7;

% Choose the frequency band(s) you want to use. Available are:
%    0 - Don't use frequency bands but the unfiltered signal
%    1 - Use signal bandpass filtered to 1 - 4 Hz
%    2 - Use signal bandpass filtered to 4 - 7 Hz
%    3 - Use signal bandpass filtered to 7 - 13 Hz
%    4 - Use signal bandpass filtered to 17 - 23 Hz
%    5 - Use signal bandpass filtered to 27 - 43 Hz
freqBands = 0;
% freqBands = 1:5;

% Elastic net mixing parameter
alpha = 0.8;

% Random number generator seed
seed = 1;

% Number of CV folds in validating regularization parameter lambda
nfolds = 5;

%% Initialize
rng(seed)

%% Load training data and generate features
disp('Loading training data ...')
% 
% tmpFileName = sprintf(...
%     ['TmpTrainData_Features_%i', repmat('-%i', 1, length(generatedFeatures)-1)], ...
%      generatedFeatures);
 
 tmpFileName = sprintf(...
    ['TmpTestData_Features_%i', repmat('-%i', 1, length(generatedFeatures)-1)], ...
     generatedFeatures);

usePreComputedData = exist(tmpFileName, 'file') && ...
    strcmpi(questdlg(sprintf(...
    'Do you want to use previously prepared data in file ''%s''?', ...
    tmpFileName), 'Using same data as in previous run', 'Yes', 'No', 'Yes'), 'Yes');
drawnow

if usePreComputedData
    load(tmpFileName, 'XTrain', 'yTrain')
else
    XTrain=[];yTrain=[];
    for s=1:length(subject)
%         sub_file = ['train_subject' num2str(subject(s)) '.mat'];
        sub_file = ['test_subject' num2str(subject(s)) '.mat'];
%         load(fullfile(dataPath, sub_file), 'X', 'y', 'tmin','tmax','sfreq')
        load(fullfile(dataPath, sub_file), 'X', 'Id', 'tmin','tmax','sfreq')
        % Generate features
        disp('Generating features ...')
        disp(sub_file)
        td1 = [];
        for k = 1 : length(freqBands)
            tmp = computeFeatures(X, generatedFeatures,tmin,tmax,sfreq);
            tmp = reshape(tmp, size(tmp,1), []);
            td1 = [td1, tmp];
        end
        
        XTrain1 = td1;
        clear td1 tmp
        
%         yTrain1 = y;
        yTrain1 = Id;
        
        XTrain = [XTrain;XTrain1];
        yTrain=[yTrain;yTrain1];
        
        clear XTrain1 yTrain1
    end
    % Save for future use
    XTest=XTrain;
    Id=yTrain;
    save(tmpFileName, 'XTest', 'Id')
%     save(tmpFileName, 'XTrain', 'yTrain')
end

