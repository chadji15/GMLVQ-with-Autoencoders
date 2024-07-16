% Auxiliary script to compare prototypes with pixel-wise average images.
%% After loading GMLVQ + Autoenc model
%load(settings.modelPath);


if settings.dataset == "FashionMNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadFashionMNIST();
elseif settings.dataset == "MNIST"
[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);
elseif settings.dataset == "CIFAR10"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR();
elseif settings.dataset == "CIFAR10BW"
[trainingImages, trainingLabels, testImages, testLabels] = loadCIFAR(["horse", "ship"],true);
else
return
end

%% decode the prototypes
nPrototypes = size(result.averageRun.prototypes,1);
run = result.results(end).run;
prototypes = run.prototypes;

if settings.doztr
    % revert the zscore transfor mation that takes place in the toolbox
    prototypes = prototypes .* repmat(run.stdFeatures,nPrototypes,1)...
        + repmat(run.meanFeatures, nPrototypes, 1);
    
end


classes = keys(lt.labelMap);
origPrototypes = autoenc.decode(prototypes);
% for i = 1:length(classes)
%     subplot(2,length(classes),i);
%     imshow(squeeze(origPrototypes(:,:,:,i)));
%     title("Prototype")
% end
% 
% %% Calculate pixel-wise average
% 
% idx = trainingLabels == gmlvq.plbl(1);
% subplot(1,length(classes),1);
% imgAvg1 = mean(trainingImages(:,:,:,idx),4);
% imshow(imgAvg1);
% title("Horse pixel-wise average")
% subplot(1,length(classes),2);
% imgAvg2 = mean(trainingImages(:,:,:,~idx),4);
% imshow(imgAvg2);
% title("Ship pixel-wise average")

%%
similarity = [];
for i=1:length(settings.classes)
    idx = trainingLabels == lt.reverse(gmlvq.plbl(i));
    imgAvg = mean(trainingImages(:,:,:,idx),4);
    similarity(i) = ssim(double(origPrototypes(:,:,:,i)), mean(trainingImages(:,:,:,idx),4));
end

disp(similarity)