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

%% Split images

img = trainingImages(:,:,:,1);
showRGBchannels(img);

%% Train autoencoder

hiddenSize = 32;
maxEpochs = 30;
images = trainingImages;

autoenc = StackedAutoencoder(hiddenSize, images, maxEpochs);

%% Test set
mse = autoenc.test(testImages)

%% Generate new images

YNew = autoenc.generateNew();

%% Save
save("models/StackedCAE_CIFAR10.mat", "autoenc");