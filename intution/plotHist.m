
trial=100;
z=squeeze(channel_index(trial,:,:));
for i=1:9
   subplot(3,3,i); hist(z(:,i));
end




 