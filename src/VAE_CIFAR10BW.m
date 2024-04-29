clc;
clear;
%% Settings

numLatentChannels = 128;
numEpochs = 100;
learnRate = 1e-3;
%% Load the dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(true);

figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,1:1000);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train VAE

mVAE = VAE.trainVAE(numLatentChannels, trainingImages, numEpochs, learnRate);

%% Test set

mse = mVAE.test(testImages)

%% Generate new images
mVAE.generateNew();

%% Save
save("models/VAE_CIFAR10BW.mat", "mVAE");

