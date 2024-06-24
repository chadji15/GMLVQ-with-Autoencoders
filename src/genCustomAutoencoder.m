clc;
clear;
addpath(genpath(pwd));
%% Settings

settings.hiddenSize = 128;
settings.numEpochs = 30;
settings.dataset = "CIFAR10";
settings.activation = "sigmoid";
settings.savePath = "models/VAE_sigmoid_CIFAR10.mat";
settings.learnRate = 1e-3;  
settings.rescaleInput = false;
settings.classes = ["airplane", "automobile", "bird", "cat", ...
            "deer", "dog", "frog", "horse", "ship", "truck"];
% settings.classes = 0:1;
%% Load the dataset
[trainingImages, trainingLabels, testImages, testLabels] = ...
    loadDataset(settings.dataset, settings.classes);


figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,1:1000);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train autoencoder

autoenc = VAE(trainingImages, ...
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
