clc;
clear;
addpath(genpath(pwd));
%% Settings

settings.hiddenSize = 256;
settings.numEpochs = 100;
settings.dataset = "CIFAR10";
settings.savePath = "models/CAE_"+ settings.dataset + ".mat";
settings.activation = "sigmoid";
settings.learnRate = 1e-3;
%% Load the dataset
[trainingImages, trainingLabels, testImages, testLabels] = loadDataset(settings.dataset);


figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,1:1000);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train autoencoder

autoenc = CustomAutoencoder(trainingImages, ...
    settings.hiddenSize, ...
    settings.numEpochs, ...
    "activation",settings.activation, ...
    "learnRate", settings.learnRate);

%% Test set

mse = autoenc.test(testImages)

%% Generate new images
autoenc.generateNew();

%% Save
save(settings.savePath, "autoenc", "settings");
