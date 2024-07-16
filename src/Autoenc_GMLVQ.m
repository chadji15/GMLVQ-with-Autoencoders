% Train GMLVQ on the latent space using an autoencoder that is already
% trained
clc;
clear;

%% Settings

% This is the path of the trained autoencoder
model = "CAE_sigmoid_FashionMNIST10.mat";
modelPath = "models/" + model;
% settings.classes = 0:1;
load(modelPath);
settings.dataset = "MNIST";
% This is true if z-score transformation should be used
settings.doztr = true;
% Total GMLVQ stepts
settings.totalSteps = 30;
settings.runs = 10;
settings.percentage = 10; %for validation
settings.savePath = "models/GMLVQ_CAE_sigmoid_FashionMNIST10.mat";
settings.prototypesPerClass = 1;
settings.modelPath = "models/" + model;


%%

if settings.dataset == "FashionMNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST(settings.classes);
elseif settings.dataset == "MNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);
elseif settings.dataset == "CIFAR10"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(settings.classes);
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
    GMLVQ.Parameters("doztr", settings.doztr), settings.totalSteps, 1:10);
result = gmlvq.runValidation(settings.runs,settings.percentage);

% The following section decodes and visualizes the prototypes
% Uncomment to execute
% decode the prototypes
% nPrototypes = size(result.averageRun.prototypes,1);
% prototypes = result.averageRun.prototypes;
% 
% if settings.doztr
%     % revert the zscore transfor mation that takes place in the toolbox
%     prototypes = result.averageRun.prototypes .* repmat(result.averageRun.stdFeatures,nPrototypes,1)...
%         + repmat(result.averageRun.meanFeatures, nPrototypes, 1);
% 
% end
% 
% 
% classes = keys(lt.labelMap);
% origPrototypes = autoenc.decode(prototypes);
% for i = 1:length(classes)*settings.prototypesPerClass
%     subplot(length(classes),settings.prototypesPerClass,i);
%     imshow(squeeze(origPrototypes(:,:,:,i)), []);
% end

%% Save 

save(settings.savePath, "result", "gmlvq", "lt", "settings", "autoenc")