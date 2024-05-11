clear autoenc;
load(settings.modelPath);
if ~exist('autoenc','var')
    autoenc = mVAE;
end

%% Calculate accuracy

disp("Accuracy")
cmatrix = result.averageRun.validationPerf(end).confusionMatrix;
accuracy = sum(diag(cmatrix)) / sum(cmatrix,"all")

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
    imshow(squeeze(origPrototypes(:,:,:,i)));
end

%% Plot eigenvectors
rel = result.averageRun.lambda;
Z = diag(result.averageRun.stdFeatures);
rel_inv = Z;
if settings.doztr
rel_inv = Z.' * rel * Z;
end
[V, D] = eig(rel_inv,'vector');

% [m,idx] = max(D);
% primEig = V(:,idx);
% primEigIm = autoenc.decode(transpose(primEig));
% subplot(1,2,1);
% imshow(primEigIm);
% 
% primEig = V(:,end-idx);
% primEigIm = autoenc.decode(transpose(primEig));
% subplot(1,2,2);
% imshow(primEigIm);

num = 10;
figure;
for i=1:num
    dec = autoenc.decode(transpose(V(:,i)));
    %im = rescale(dec);
    im=dec;
    subplot(2,5,i);
    imshow(im,[]);
end

