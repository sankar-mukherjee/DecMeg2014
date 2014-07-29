DecMeg2014
==========

Kaggle DecMeg2014 - Decoding the Human Brain

https://www.kaggle.com/c/decoding-the-human-brain

Final place 78/288
=======================================================================================
Channel_Performance	=	Select channels with high performance and time [0.5 - 1]. concat all the time series into a single vector.

icann11_sank	=	Different combination of features created using ICANN'11 Mind Reading challenge.

intution	=	Select 10 sample points (according to amplitude max - min) from each time series of each channel.

occipitalLobe	=	Experiment with selecting only the occipitalLobe sensors. selecting from 0.5 sec. Bandpass filter. Average trails for each subject with each class.

PCA_andrew	=	Create train and test set by concatenting 306 channles into a one long vector [Kaggle provided]. pca with [andrew ng] 1000 components. Classify with Matlab Logistic regression [mnrfit].

Final.csv = predicted file [68%]

=============================================================================================
python_scripts_from_net	=	good scripts (https://github.com/djpetti/dec-meg-2014)
