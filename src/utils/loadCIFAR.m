function [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(classes,bw)
%loadCIFAR Load the CIFAR10 dataset from the .mat files and splits it into
% training and test sets. For now we keep the classes "horse" and "ship". 
% If the parameter "bw" is true, the images are turned into grayscale
% before returning them.
% 
    if ~exist("bw", "var")
        bw = false;
    end
    [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load('.\data');
    % Select two classes
    
    if ~exist("classes", "var") || (isstring(classes) && all(classes == "default"))
        classes = ["horse", "ship"];
    end

    classes = categorical(classes);
    
    trainIdx = find(ismember(trainingLabels, classes));
    trainingImages = trainingImages(:,:,:,trainIdx);
    trainingLabels = trainingLabels(trainIdx);
    
    testIdx = find(ismember(testLabels, classes));
    testImages = testImages(:,:,:,testIdx);
    testLabels = testLabels(testIdx);

    % If bw is true, turn training and test images to grayscale
    if bw
        for c=1:size(trainingImages, 4)
            trainingImagesBW(:,:,:,c) = rgb2gray(trainingImages(:,:,:,c));
        end
        for c=1:size(testImages, 4)
            testImagesBW(:,:,:,c) = rgb2gray(testImages(:,:,:,c));
        end
        trainingImages = trainingImagesBW;
        testImages = testImagesBW;
    end
end