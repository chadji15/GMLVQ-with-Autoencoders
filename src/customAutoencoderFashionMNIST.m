clc;
clear;
%% Load the dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();

figure;
set(gcf,'color',[1 1 1])
thumbnails = trainingImages(:,:,:,[1:1000]);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails,'size',[20 50])

%% Train autoencoder

hiddenSize = 32;

autoenc = CustomAutoencoder22(hiddenSize, trainingImages);

%% Test set
mse = autoenc.test(testImages)

%% Generate new images

YNew = autoenc.generateNew();

%% Save
save("models/CAE22_FashionMNIST.mat", "autoenc");