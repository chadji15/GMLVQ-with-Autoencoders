clc;
clear;

%% Settings

model = "FCAE_tanh_FashionMNIST.mat";
settings.modelPath = "models/" + model;
settings.dataset = "FashionMNIST";
settings.doztr = true;
settings.totalSteps = 10;
settings.runs = 1;
settings.percentage = 10; %for validation
settings.savePath = "models/GMLVQ_" + model;
%%
load(settings.modelPath);
if ~exist('autoenc','var')
    autoenc = mVAE;
end

if settings.dataset == "FashionMNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();
elseif settings.dataset == "MNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST();
elseif settings.dataset == "CIFAR10"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();
elseif settings.dataset == "CIFAR10BW"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(true);
else
return
end

%%
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
origPrototypes = autoenc.decode(prototypes);
for i = 1:length(classes)
    subplot(1,length(classes),i);
    imshow(squeeze(origPrototypes(:,:,:,i)), []);
end

%% Save 

save(settings.savePath, "result", "gmlvq", "prototypes", "lt", "settings")