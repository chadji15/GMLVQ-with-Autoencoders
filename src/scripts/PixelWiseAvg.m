%% After loading GMLVQ + Autoenc model
load(settings.modelPath);


if settings.dataset == "FashionMNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();
elseif settings.dataset == "MNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST();
elseif settings.dataset == "CIFAR10"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();
elseif settings.dataset == "CIFAR10BW"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(["horse", "ship"],true);
else
return
end

%% decode the prototypes
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
    subplot(2,length(classes),i);
    imshow(squeeze(origPrototypes(:,:,:,i)));
    title("Prototype")
end

%% Calculate pixel-wise average

idx = trainingLabels == trainingLabels(1);
subplot(2,length(classes),3);
imgAvg1 = mean(trainingImages(:,:,:,idx),4);
imshow(imgAvg1);
title("Pixel-wise average")
subplot(2,length(classes),4);
imgAvg2 = mean(trainingImages(:,:,:,~idx),4);
imshow(imgAvg2);
title("Pixel-wise average")