%% logistic regression with matlab default
clear;clc;
load('TmpTrainData_Features_1-4.mat');
load('TmpTestData_Features_1-4.mat');

% feature normalize
X=zeros(size(XTrain));
for i=1:size(XTrain,1)
    X(i,:) = XTrain(i,:)-mean(XTrain(i,:));
    X(i,:) = X(i,:)./std(X(i,:));    
end
y=double(yTrain)+1;

%randomized samples
[XTrain i]=datasample(X,2000,'Replace',false);    X(i,:) = [];
YTrain=y(i);
[Xcv i]=datasample(X,1000,'Replace',false);    X(i,:) = [];
Ycv=y(i);

%training
B=mnrfit(XTrain,YTrain);

%Test with Cross validation
P = mnrval(B,Xcv);
[val, pred] = max(P, [], 2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Ycv)) * 100);

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
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(Id)
    fprintf(f,'%d,%d\n',Id(i),pred(i));
end
fclose(f);