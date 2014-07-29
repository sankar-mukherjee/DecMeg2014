X=X_train;y=y_train;
y=double(y);
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

p = svmPredict(model, X_test);
filename_submission = 'andrewSVM.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(ids_test)
    fprintf(f,'%d,%d\n',ids_test(i),p(i));
end
fclose(f);