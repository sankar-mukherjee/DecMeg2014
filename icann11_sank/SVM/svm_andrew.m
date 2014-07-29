%% Initialization
clear ; close all; clc

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

% linear kernel
load('../TmpTrainData_Features_1-2-3-4-5-6-7-8-9-10-11.mat'); 
X=XTrain;y=yTrain;
y=double(y);
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% gaussian kernel

%x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
%sim = gaussianKernel(x1, x2, sigma);

%load('../TmpTrainData_Features_1-4.mat'); 
%C = 1; sigma = 0.1;

%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%p = svmPredict(model, X);

%fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

