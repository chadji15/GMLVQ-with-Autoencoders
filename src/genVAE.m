clc;
clear;
addpath(genpath(pwd));
%% Settings

settings.hiddenSize = 32;
settings.numEpochs = 20;
settings.learnRate = 1e-3;
settings.dataset = "FashionMNIST";
settings.savePath = "models/VAE_"+ settings.dataset + ".mat";
%% Load the dataset
[trainingImages, trainingLabels, testImages, testLabels] = loadDataset(settings.dataset);


figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,1:1000);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train VAE

autoenc = VAE.trainVAE(settings.hiddenSize, trainingImages, settings.numEpochs, settings.learnRate);

%% Test set

mse = autoenc.test(testImages)

%% Generate new images
autoenc.generateNew();

%% Save
save(settings.savePath, "autoenc", "settings");
