settings.classes = ["T-shirt/top", "Trouser", "Pullover", ...
            "Dress", "Coat","Sandal", "Shirt","Sneaker", "Bag", "Ankle boot"];
settings.rescaleInput = false;
settings.hiddenSize = 32;
settings.numEpochs = 30;
settings.activation = "tanh";
settings.learnRate = 1e-3;
settings.doztr = true;
settings.savePath = "models/GMLVQ_FCAE_tanh_FashionMNIST10.mat";
settings.runs = 1;
settings.percentage = 10;
settings.totalSteps = 20;
settings.prototypesPerClass = 1;
settings.dataset = "FashionMNIST";

[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST(settings.classes);

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
    settings.totalSteps,1:10);

result = gmlvq.runValidation(settings.runs,settings.percentage);


save(settings.savePath, "autoenc", "result", "gmlvq", "lt", "settings")
