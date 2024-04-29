clc;
clear;
load("models/CAE_MNIST.mat");

%% Settings

doztr = false;
totalSteps = 10;
runs = 1;
percentage = 10;
%%
if ~exist('autoenc','var')
    autoenc = mVAE;
end

%%

[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST();

% encode the training data
xencoded = autoenc.encode(trainingImages);

% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

% train the gmlvq model
gmlvq = GMLVQ.GMLVQ(xencoded, transformedLabels,GMLVQ.Parameters("doztr", doztr), totalSteps);

result = gmlvq.runValidation(runs,percentage);

% decode the prototypes
nPrototypes = size(result.averageRun.prototypes,1);
prototypes = result.averageRun.prototypes;

if doztr
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