function [trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(bw)
    if ~exist("bw", "var")
        bw = false;
    end
    [trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load('.\data');
    % Select two classes
    
    classes = categorical(["frog", "truck"]);
    
    trainIdx = find(ismember(trainingLabels, classes));
    trainingImages = trainingImages(:,:,:,trainIdx);
    trainingLabels = trainingLabels(trainIdx);
    
    testIdx = find(ismember(testLabels, classes));
    testImages = testImages(:,:,:,testIdx);
    testLabels = testLabels(testIdx);

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