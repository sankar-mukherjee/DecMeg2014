clear all;clc;
subjects_train = 1:16;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));
% We throw away all the MEG data outside the first 0.5sec from when
% the visual stimulus start:
tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
X_train = [];
y_train = [];
X_test = [];
ids_test = [];
% Crating the trainset. (Please specify the absolute path for the train data)
disp('Creating the trainset.');
for i = 1 : length(subjects_train)
    path = '../python/data/train/';  % Specify absolute path
    filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_train(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    yy = data.y;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('yy: %d trials',size(yy,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_train = [X_train;features];
    y_train = [y_train;yy];
end

% Crating the testset. (Please specify the absolute path for the test data)
disp('Creating the testset.');
subjects_test = 17:23;
for i = 1 : length(subjects_test)
    path = '../python/data/test/';  % Specify absolute path
    filename = sprintf(strcat(path,'test_subject%02d.mat'),subjects_test(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    ids = data.Id;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('Ids: %d trials',size(ids,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_test = [X_test;features];
    ids_test = [ids_test;ids];
end