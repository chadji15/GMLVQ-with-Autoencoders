function [trainingImages, trainingLabels,testImages,testLabels] = loadDataset(dataset, classes)
%LOADDATASET Auxiliary function for making the scripts more modular.
    if ~exist("classes", "var") 
        classes = "default";
    end 
    if dataset == "FashionMNIST"
    [trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST(classes);
    elseif dataset == "MNIST"
    [trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(classes);
    elseif dataset == "CIFAR10"
    [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(classes);
    elseif dataset == "CIFAR10BW"
    [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(classes,true);
    else
    return
    end
end

