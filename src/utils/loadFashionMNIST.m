function [trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST(classes)
%loadFashionMNIST Loads the FashionMNIST dataset from the binary files and
% splits it into training and test set. For now we keep the classes "Bag"
% and "Trouser" but this can become a parameter if need be.
    TrainimageFileName = 'fashionMNIST/train-images-idx3-ubyte';
    TrainlabelFileName = 'fashionMNIST/train-labels-idx1-ubyte';
    TestimageFileName = 'fashionMNIST/t10k-images-idx3-ubyte';
    TestlabelFileName = 'fashionMNIST/t10k-labels-idx1-ubyte';
    
    [XTrain,YTrain] = processFashionMNISTdata(TrainimageFileName,TrainlabelFileName);
    [XTest,YTest] = processFashionMNISTdata(TestimageFileName,TestlabelFileName);
    
    YTrain = string(YTrain);
    YTest = string(YTest);
    
    % Keep only the labels that interest us
    if ~exist("classes", "var") || (isstring(classes) && all(classes == "default"))
        classes = ["Bag", "Trouser"];
    end
    
    idx = ismember(YTrain, classes);
    trainingImages = XTrain(:,:,:,idx);
    trainingLabels = YTrain(idx);
    
    idx = ismember(YTest, classes);
    testImages = XTest(:,:,:,idx);
    testLabels = YTest(idx);
end