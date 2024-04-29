clc;
clear;
addpath(genpath(pwd));
%% Load the dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();

figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,1:1000);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train autoencoder

hiddenSize = 128;

autoenc = CustomAutoencoder(hiddenSize, trainingImages,100);

%% Test set
mse = autoenc.test(testImages)

%% Generate new images

YNew = autoenc.generateNew();

%% Save
save("models/CAE_CIFAR10.mat", "autoenc");