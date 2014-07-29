load('../../python/data/train_subject1.mat')
Trial=size(X,1);
channel_index=zeros(Trial,306,10);
for t=1:Trial
    z=X(t,:,:);
    z=squeeze(z);    
    for i=1:306
        %     x(i,:)=a(i,:)/max(a(i,:));
        for s=1:10
            [x index]=max(abs(z(i,:)));
            z(i,index)=0;
            channel_index(t,i,s)=index;
        end
    end
end