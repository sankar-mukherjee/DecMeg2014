clear;clc;

load('../TmpTrainData_Features_1-4.mat');

train_x=(((XTrain-min(XTrain(:)))/(max(XTrain(:)) -min(XTrain(:)))));
% test_x  =(((X_test-min(X_test(:)))/(max(X_test(:)) -min(X_test(:)))));
train_y = zeros(size(yTrain,1),2);
for i=1:size(train_y,1)
    if(yTrain(i)==1)
        train_y(i,2)=1;
    elseif (yTrain(i)==0)
        train_y(i,1)=1;
    end
end
% test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0);
dbn.sizes = [400 400];
opts.numepochs =   10;
opts.batchsize = 6;
opts.momentum  =   0.95;
opts.alpha     =   0.002;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%%
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 2);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  5;
opts.batchsize = 1569;
nn = nntrain(nn, train_x, train_y, opts);
labels = nnpredict(nn, test_x);