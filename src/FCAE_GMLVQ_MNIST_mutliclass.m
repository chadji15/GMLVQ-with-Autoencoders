settings.classes = 0:1;
settings.rescaleInput = false;
settings.hiddenSize = 10;
settings.numEpochs = 20;
settings.activation = "tanh";
settings.learnRate = 1e-3;
settings.doztr = true;
settings.savePath = "models/GMLVQ_FCAE_tanh_MNIST_5prot.mat";
settings.runs = 1;
settings.percentage = 10;
settings.totalSteps = 20;
settings.prototypesPerClass = 5;

[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);

if settings.rescaleInput
    trainingImages = rescale(trainingImages,-1,1);
end


autoenc = FCAE(trainingImages, ...
    settings.hiddenSize, ...
    settings.numEpochs, ...
    "activation",settings.activation, ...
    "learnRate", settings.learnRate);

% encode the training data
xencoded = autoenc.encode(trainingImages);

% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

% train the gmlvq model
gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters("doztr", settings.doztr), ...
    settings.totalSteps, ...
    [repmat(1, 1,settings.prototypesPerClass), repmat(2,1,settings.prototypesPerClass)]);

result = gmlvq.runValidation(settings.runs,settings.percentage);


save(settings.savePath, "autoenc", "result", "gmlvq", "lt", "settings")
