clc;
clear;
addpath(genpath(pwd));
%% Settings

settings.hiddenSize = 10;
settings.numEpochs = 20;
settings.dataset = "MNIST";
settings.activation = "sigmoid";
settings.savePath = "models/FCAE_tanh_MNIST.mat";
settings.learnRate = 1e-3;  
settings.rescaleInput = false;
settings.classes = 0:2;
%% Load the dataset
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);


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

mse = autoenc.test(testImages(:,:,:,1:8))

%% Generate new images
autoenc.generateNew();

%% Save
save(settings.savePath, "autoenc", "settings");
