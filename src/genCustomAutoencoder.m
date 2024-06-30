clc;
clear;
addpath(genpath(pwd));
%% Settings

settings.hiddenSize = 32;
settings.numEpochs = 30;
settings.dataset = "FashionMNIST";
settings.activation = "tanh";
settings.savePath = "models/FCAE_tanh_FashionMNIST10.mat";
settings.learnRate = 1e-3;  
settings.rescaleInput = false;
settings.classes = ["T-shirt/top", "Trouser", "Pullover", ...
            "Dress", "Coat","Sandal", "Shirt","Sneaker", "Bag", "Ankle boot"];
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

autoenc = FCAE(trainingImages, ...
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
