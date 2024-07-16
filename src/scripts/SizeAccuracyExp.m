% This script runs the experiment where we compare the performance of the
% classifiers that are based on autoencoders with different hidden sizes.

%% Settings

settings.doztr = true;
settings.totalSteps = 30;
settings.runs = 1;
settings.percentage = 10; %for validation
settings.activation = "tanh";
settings.epochs =  50;
settings.classes =  ["T-shirt/top", "Trouser", "Pullover", ...
            "Dress", "Coat","Sandal", "Shirt","Sneaker", "Bag", "Ankle boot"];


%% Load the dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST(settings.classes);
%% Train autoencoder

origPrototypes = [];
accuracy = [];
auroc = [];
primEigIm = [];
mse = [];

% Loop over all the hidden sizes we want to test.
% For large datasets its best to use a step size larger than 1.
for hiddenSize=5:125
    % Train the autoencoder
    % Can be replaced with any of the other two architectures.
    autoenc = FCAE(trainingImages,hiddenSize,settings.epochs, ...
        'activation',settings.activation, ...
        "plots", "none");

    % Calculate MSE on test set.
    mse(hiddenSize) = autoenc.test(testImages,false);

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
    
    
    classes = keys(lt.labelMap);
    % Save the prototypes
    origPrototypes(:,:,:,:,hiddenSize) = autoenc.decode(prototypes);
    cmatrix = result.averageRun.validationPerf(end).confusionMatrix;
    % Save accuracy on validation set
    accuracy(hiddenSize) = sum(diag(cmatrix)) / sum(cmatrix,"all");
    % Save auroc on validation set
    auroc(hiddenSize) = result.averageRun.validationPerf(end).auroc;
    % Decode the eigenvectors of the relevance matrix and save them
    rel = result.averageRun.lambda;
    Z = diag(result.averageRun.stdFeatures);
    rel_inv = Z;
    if settings.doztr
    rel_inv = Z.' * rel * Z;
    end
    [V, D] = eig(rel_inv,'vector');
    [m,idx] = max(D);
    primEig = V(:,idx);
    primEigIm(:,:,hiddenSize) = autoenc.decode(transpose(primEig));
end

res.primEigIm = primEigIm;
res.auroc = auroc;
res.accuracy = accuracy;
res.mse = mse;
res.origPrototypes = origPrototypes;

%% Save
save("SizeAccuracyExpRes10class.mat", "res", "settings");