function savepath = run_full(settings)

[trainingImages, trainingLabels, testImages, testLabels] = loadDataset(settings.dataset);

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
    autoenc = VAE.trainVAE(trainingImages, ...
        settings.hiddenSize, ...
        settings.numEpochs, ...
        "learnRate", settings.learnRate);
end
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
% decode the prototypes
nPrototypes = size(result.averageRun.prototypes,1);
prototypes = result.averageRun.prototypes;

if settings.doztr
    % revert the zscore transfor mation that takes place in the toolbox
    prototypes = result.averageRun.prototypes .* repmat(result.averageRun.stdFeatures,nPrototypes,1)...
        + repmat(result.averageRun.meanFeatures, nPrototypes, 1);
    
end


save(settings.savePath, "autoenc", "result", "gmlvq", "prototypes", "lt", "settings")
savepath = settings.savePath;
end