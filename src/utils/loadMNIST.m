function [trainingImages, trainingLabels, testImages, testLabels] = loadMNIST()
    % Load the mnist dataset
    load('mnist/mnist.mat');
    
    % Keep only the labels that interest us
    classes = [0 1];
    
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