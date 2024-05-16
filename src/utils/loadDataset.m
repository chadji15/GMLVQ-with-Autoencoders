function [trainingImages, trainingLabels,testImages,testLabels] = loadDataset(dataset)
%LOADDATASET Summary of this function goes here
%   Detailed explanation goes here
    if dataset == "FashionMNIST"
    [trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();
    elseif dataset == "MNIST"
    [trainingImages, trainingLabels, testImages, testLabels] = loadMNIST();
    elseif dataset == "CIFAR10"
    [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();
    elseif dataset == "CIFAR10BW"
    [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(true);
    else
    return
    end
end

