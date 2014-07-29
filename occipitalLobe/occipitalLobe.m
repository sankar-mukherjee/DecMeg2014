% Selecting the channels that matter
close all;clear all;clc

%%
load('../../additional_files/NeuroMagSensorsDeviceSpace.mat'); 
scatter(getcolumn(pos(:,1:2),1),getcolumn(pos(:,1:2),2));figure(gcf);
disp('select points related to occipetal lobe and create a variable (default)');
pause;
%calculation for occipital lobe. Extracted by first ploting the xy postion
%from sensor position then seleting few points (z) from the back of the
%head. ans has 3 values for each sensor name the variable as ans so apply zz =
%unique(ans,'rows')

zz = unique(ans,'rows');
p=pos(:,1,1);
a=zz(:,1);
Occi_index=[];
for i=1:length(a)
   Occi_index =[Occi_index; find(abs(p-a(i))<1e-3)];
end
%%

load('../../python/data/train/train_subject01.mat'); 

%load time series data after the stimulas i.e. .5*250=125 to 375
x=X(:,:,.5*sfreq+1:end);
Occi=x(:,Occi_index,:); %Occipital lobe

%% normalize by mean for each channel (not sure if its right approach)
Occi_norm=zeros(size(Occi));
for i=1:size(Occi,1)
    for j=1:size(Occi,2)
        Occi_norm(i,j,:) = squeeze(Occi(i,j,:))-norm(squeeze(Occi(i,j,:))); 
    end   
end

%% band pass filter butterworth filter (!!) 2-30 Hz with 0.01 stop band
BS = 2; BE = 100; 
Wp = [BS BE]/(sfreq/2); Ws = [BS-0.01 BE+0.01]/(sfreq/2); Rp = 3; Rs = 5;
[n,Wn] = buttord(Wp,Ws,Rp,Rs);
[b,a] = butter(5, Wn, 'bandpass');

Occi_filt=zeros(size(Occi));
for i=1:size(Occi,1)
    for j=1:size(Occi,2)
        Occi_filt(i,j,:) = filtfilt(b, a,double(squeeze(Occi(i,j,:)))); 
    end   
end
Occi=Occi_filt;

%% differntiate two class
index_C1 = find(y==1);index_C2 = find(y==0);
Occi_norm_C1 = Occi(index_C1,:,:);
Occi_norm_C2 = Occi(index_C2,:,:);

%averaging accross trail for two classes
meanAcrosTrail_C1 =zeros(size(Occi_norm_C1,2),size(Occi_norm_C1,3));
meanAcrosTrail_C1_std =zeros(size(Occi_norm_C1,2),size(Occi_norm_C1,3));
for i=1:size(Occi_norm_C1,2)
   meanAcrosTrail_C1(i,:) = mean(squeeze(Occi_norm_C1(:,i,:))); 
   meanAcrosTrail_C1_std(i,:) = std(squeeze(Occi_norm_C1(:,i,:))); 
end

meanAcrosTrail_C2 =zeros(size(Occi_norm_C2,2),size(Occi_norm_C2,3));
meanAcrosTrail_C2_std =zeros(size(Occi_norm_C2,2),size(Occi_norm_C2,3));
for i=1:size(Occi_norm_C2,2)
   meanAcrosTrail_C2(i,:) = mean(squeeze(Occi_norm_C2(:,i,:))); 
   meanAcrosTrail_C2_std(i,:) = std(squeeze(Occi_norm_C2(:,i,:))); 
end

%% plotting

plot(meanAcrosTrail_C2');title('Class 0 ');figure(gcf);
figure,plot(meanAcrosTrail_C1');title('Class 1 ');figure(gcf);


