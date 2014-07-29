[U, S] = pca(X_train);
Z = projectData(X_train, U, 1000);

y_train=y_train+1;
%training
B=mnrfit(z,y_train);

%Test with Cross validation
P = mnrval(B,X_train);
[val, pred] = max(P, [], 2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

%% Saving the results in the submission file:
P = mnrval(B,X_test);
[val, pred] = max(P, [], 2);
pred=pred-1;
filename_submission = 'pca_MatlabLogistic.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(ids_test)
    fprintf(f,'%d,%d\n',ids_test(i),P(i));
end
fclose(f);