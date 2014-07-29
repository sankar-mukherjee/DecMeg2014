function Fe = computeFeatures(data, generatedFeatures,tmin,tmax,sfreq)

% Function computes frequency band -wise features from ICANN 2011 MEG data.
%
% Inputs:
%
% data -- A 3-D data matrix of the size nrTimeFrames*nrChannels*nrTimePoints
% extracted from a specific frequency band.
%
% generatedFeatures -- IDs of generated features. Available are:
%     1 - Mean (= detrended mean)
%     2 - Slope of fitted line
%     3 - Detrended variance
%     4 - Detrended standard deviation
%     5 - Detrended skewness
%     6 - Detrended kurtosis
%     7 - Variance
%     8 - Standard deviation
%     9 - Skewness
%    10 - Kurtosis
%    11 - Fluctuation around mean
%
% Outputs:
%
% Fe -- 3-D feature matrix of the size nrSamples*nrChannels*nrFeatures.
%
% See also: COMPUTEFEATURES
%
%
% Jukka-Pekka Kauppi
% Tampere University of Technology
% Department of Signal Processing
% 16.8.2011
%
tmin_original=tmin;
beginning = (tmin - tmin_original) * sfreq+1;
e = (tmax - tmin_original) * sfreq;
data = data(:, :, beginning:e);

% generate features:
nrFeatures = length(generatedFeatures);
nrChannels = size(data,2);
Fe = zeros(size(data,1),nrChannels,nrFeatures);

h = waitbar(0, 'Generating features ...');

for sampleNr = 1:size(data,1) % for each time-window
    waitbar(sampleNr / size(data,1), h)
    %     disp(['          Time-window ' num2str(sampleNr) '/' num2str(size(data,1))])
    for channelNr = 1:size(data,2)
        ts = squeeze(data(sampleNr,channelNr,:)); % get time-series data
        
        % Fit a line
        if any(ismember(2:6, generatedFeatures))
            X = [ones(length(ts),1) (1:length(ts))'];
            B = X \ ts(:);
            res = ts(:) - X*B;
        end            
        
        for feat = 1 : nrFeatures
            switch generatedFeatures(feat)
                
                case 1 % Mean (= detrended mean)
                    x = mean(ts);
                    
                case 2 % Slope of fitted line
                    x = B(2);
                    
                case 3 % Detrended variance
                    x = var(res);
                    
                case 4 % Detrended standard deviation
                    x = std(res);
                    
                case 5 % Detrended skewness
                    x = skewness(res);
                    
                case 6 % Detrended kurtosis
                    x = kurtosis(res);
                    
                case 7 % Variance
                    x = var(ts);
                    
                case 8 % Standard deviation
                    x = std(ts);
                    
                case 9 % Skewness
                    x = skewness(ts);
                
                case 10 % Kurtosis
                    x = kurtosis(ts);
                    
                case 11 % Increasing/decreasing function or fluctuation
                        % around mean, range between [0 1]
                    F = double(diff(ts) >= 0);
                    F(F == 0) = -1;
                    x = abs(sum(F))/length(F);

                otherwise
                    error('Unknown feature index: %i', generateFeatures(feat))
            end
            
            Fe(sampleNr,channelNr,feat) = x;
        end 
    end %  channels
end % time-windows

close(h)
drawnow

