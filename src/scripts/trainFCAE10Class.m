settings.activation = activation;
settings.learnRate = 1e-3;
settings.doztr = true;
settings.totalSteps = 30;
settings.runs = 1;
settings.percentage = 10; %for validation
settings.rescaleInput = false;
settings.classes = 1:10;
settings.dataset = "MNIST";
settings.savePath = "models/GMVLQ_FCAE_tanh_MNIST10.mat";
settings.hiddenSize = 32;
settings.numEpochs = 100;


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
if size(xencoded,1) < size(xencoded, 2)
    xencoded = transpose(xencoded);
end

% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

% train the gmlvq model
gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters("doztr", settings.doztr), settings.totalSteps);

result = gmlvq.runValidation(settings.runs,settings.percentage);

save(settings.savePath, "autoenc", "result", "gmlvq", "lt", "settings");