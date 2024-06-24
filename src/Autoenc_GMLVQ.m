clc;
clear;

%% Settings

model = "VAE_sigmoid_CIFAR10.mat";
settings.modelPath = "models/" + model;
% settings.classes = 0:1;
load(settings.modelPath);
settings.dataset = "CIFAR10";
settings.doztr = true;
settings.totalSteps = 30;
settings.runs = 10;
settings.percentage = 10; %for validation
settings.savePath = "models/GMLVQ_VAE_sigmoid_CIFAR10.mat";
settings.prototypesPerClass = 1;
settings.modelPath = "models/" + model;


%%

if settings.dataset == "FashionMNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();
elseif settings.dataset == "MNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);
elseif settings.dataset == "CIFAR10"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();
elseif settings.dataset == "CIFAR10BW"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(["horse", "ship"], true);
else
return
end

%%
% encode the training data
xencoded = autoenc.encode(trainingImages);


% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

% train the gmlvq model
gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels, ...
    GMLVQ.Parameters("doztr", settings.doztr), settings.totalSteps, ...
    [repmat(1,1,settings.prototypesPerClass), repmat(2,1,settings.prototypesPerClass)]);

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
origPrototypes = autoenc.decode(prototypes);
for i = 1:length(classes)*settings.prototypesPerClass
    subplot(length(classes),settings.prototypesPerClass,i);
    imshow(squeeze(origPrototypes(:,:,:,i)), []);
end

%% Save 

save(settings.savePath, "result", "gmlvq", "lt", "settings", "autoenc")