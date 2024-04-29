function [trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST()
    TrainimageFileName = 'fashionMNIST/train-images-idx3-ubyte';
    TrainlabelFileName = 'fashionMNIST/train-labels-idx1-ubyte';
    TestimageFileName = 'fashionMNIST/t10k-images-idx3-ubyte';
    TestlabelFileName = 'fashionMNIST/t10k-labels-idx1-ubyte';
    
    [XTrain,YTrain] = processFashionMNISTdata(TrainimageFileName,TrainlabelFileName);
    [XTest,YTest] = processFashionMNISTdata(TestimageFileName,TestlabelFileName);
    
    YTrain = string(YTrain);
    YTest = string(YTest);
    
    % Keep only the labels that interest us
    classes = ["Bag", "Trouser"];
    
    idx = ismember(YTrain, classes);
    trainingImages = XTrain(:,:,:,idx);
    trainingLabels = YTrain(idx);
    
    idx = ismember(YTest, classes);
    testImages = XTest(:,:,:,idx);
    testLabels = YTest(idx);
end