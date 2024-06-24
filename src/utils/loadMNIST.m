function [trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(classes)
%loadMNIST Reads the MNIST dataset from the .mat file and splits it into
% training and test set. We keep the digits 0 and 1 at the time,
% but this can become a parameter if need be.
    % Load the mnist dataset
    load('mnist/mnist.mat');
    
    % Keep only the labels that interest us
    if ~exist("classes", "var") || all(classes == "default")
        classes = [0 1];
    end
    
    idx = ismember(training.labels, classes);
    trainingImages = training.images(:,:,idx);
    sz = size(trainingImages);
    trainingImages = reshape(trainingImages, sz(1), sz(2), 1, sz(3));
    trainingLabels = training.labels(idx);
    
    idx = ismember(test.labels, classes);
    testImages = test.images(:,:,idx);
    sz = size(testImages);
    testImages = reshape(testImages, sz(1), sz(2), 1, sz(3));
    testLabels = test.labels(idx);
end