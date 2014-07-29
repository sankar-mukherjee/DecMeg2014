

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 1836;  % 20x20 Input Images of Digits
num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../TmpTrainData_Features_1-2-3-4-8-10.mat'); % training data stored in arrays X, y
% XTrain=XTrain/norm(XTrain);
m = size(XTrain, 1);
yTrain=yTrain+1;
% Randomly select 100 data points to display
rand_indices = randperm(m);

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(XTrain, yTrain, num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, XTrain);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == yTrain)) * 100);
%predicted_nums = reshape(mod(pred(rand_indices(1:100)),10),10,10)';
