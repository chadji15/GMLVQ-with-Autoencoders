function [trainingImages, trainingLabels,testImages,testLabels] = loadDataset(dataset)
%LOADDATASET Auxiliary function for making the scripts more modular.
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

