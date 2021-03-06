clear all;close all;clc;
% Max channel performace (counting from last)[44 39 73 76 88 91]
channels = [ 262   267   233   230   218   215];
Time = 126:250;
trainPath = '../../python/data/train/';
subject=1:16;

X_train = [];
y_train = [];
X_test = [];
ids_test = [];
for s=1:length(subject)
    filename = sprintf(strcat(trainPath,'train_subject%02d.mat'),subject(s));
    disp(filename);
    load(filename);    
    XX=X(:,channels,Time);
    features = createFeatures(XX);
    X_train = [X_train;features];
    y_train = [y_train;y];
end

% Crating the testset. (Please specify the absolute path for the test data)
disp('Creating the testset.');
subjects_test = 17:23;
testPath = '../../python/data/test/';
for s = 1 : length(subjects_test)
    filename = sprintf(strcat(testPath,'test_subject%02d.mat'),subjects_test(s));
    load(filename);    
    disp(filename);
    XX=X(:,channels,Time);
    features = createFeatures(XX);
    X_test = [X_test;features];
    ids_test = [ids_test;Id];
end


% Your training code should be here:
disp('Training the classifier ...')
[BFinal,FitInfoFinal] = lasso(X_train,single(y_train),'Lambda',0.005,'Alpha',0.9);

% Testing the trained classifier on the test data
y_pred = [ones(size(X_test,1),1) X_test] * [FitInfoFinal.Intercept;BFinal];
y_pred_thresholded = zeros(size(y_pred));
y_pred_thresholded(y_pred>=median(y_pred))= 1;

% Saving the results in the submission file:
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(y_pred_thresholded)
    fprintf(f,'%d,%d\n',ids_test(i),y_pred_thresholded(i));
end
fclose(f);
disp('Done.');
