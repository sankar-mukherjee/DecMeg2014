%% logistic regression with matlab default
clear;clc;
load('Ociipital_train.mat');

y=yTrain+1;
X= XTrain;
%randomized samples
[XTrain i]=datasample(X,1200,'Replace',false);    X(i,:) = [];
YTrain=y(i);
[Xcv i]=datasample(X,528,'Replace',false);    X(i,:) = [];
Ycv=y(i);

%training
B=mnrfit(XTrain,YTrain);

%Test with Cross validation
P = mnrval(B,Xcv);
[val, pred] = max(P, [], 2);
fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == Ycv)) * 100);

%% Saving the results in the submission file:
X=zeros(size(XTest));
for i=1:size(XTest,1)
    X(i,:) = XTest(i,:)-mean(XTest(i,:));
    X(i,:) = X(i,:)./std(X(i,:));    
end
XTest=X;
P = mnrval(B,XTest);
[val, pred] = max(P, [], 2);
pred=pred-1;
filename_submission = 'OccipitalLobe_matlab_Logistic.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(Id)
    fprintf(f,'%d,%d\n',Id(i),pred(i));
end
fclose(f);
