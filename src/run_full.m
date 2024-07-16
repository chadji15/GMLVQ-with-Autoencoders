% This is a custom auxiliary script to run the whole pipeline of experiment
% based on a settings struct. Mostly used to automate experiments.

function savepath = run_full(settings)

[trainingImages, trainingLabels, testImages, testLabels] = loadDataset(settings.dataset, settings.classes);

if settings.rescaleInput
    trainingImages = rescale(trainingImages,-1,1);
end
if settings.arch == "CAE"
    autoenc = CustomAutoencoder(trainingImages, ...
        settings.hiddenSize, ...
        settings.numEpochs, ...
        "activation",settings.activation, ...
        "learnRate", settings.learnRate);
elseif settings.arch == "FCAE"
    autoenc = FCAE(trainingImages, ...
        settings.hiddenSize, ...
        settings.numEpochs, ...
        "activation",settings.activation, ...
        "learnRate", settings.learnRate);
elseif settings.arch == "VAE"
    autoenc = VAE(trainingImages, ...
        settings.hiddenSize, ...
        settings.numEpochs, ...
        "activation",settings.activation, ...
        "learnRate", settings.learnRate);
end
% encode the training data
xencoded = autoenc.encode(trainingImages);

% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

% train the gmlvq model
gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters("doztr", settings.doztr), settings.totalSteps);

result = gmlvq.runValidation(settings.runs,settings.percentage);

save(settings.savePath, "autoenc", "result", "gmlvq", "lt", "settings")
savepath = settings.savePath;
end