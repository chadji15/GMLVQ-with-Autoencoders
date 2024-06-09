%% Calculate accuracy

disp("Accuracy")
cmatrix = result.averageRun.validationPerf(end).confusionMatrix;
accuracy = sum(diag(cmatrix)) / sum(cmatrix,"all")

mrun = result.results(end).run;
% decode the prototypes
nPrototypes = size(mrun.prototypes,1);
prototypes = mrun.prototypes;

if settings.doztr
    % revert the zscore transfor mation that takes place in the toolbox
    prototypes = mrun.prototypes .* repmat(mrun.stdFeatures,nPrototypes,1)...
        + repmat(mrun.meanFeatures, nPrototypes, 1);
    
end


classes = keys(lt.labelMap);
origPrototypes = autoenc.decode(prototypes);
for i = 1:length(classes) %* settings.prototypesPerClass
    subplot(settings.prototypesPerClass,length(classes),i);
    imshow(squeeze(origPrototypes(:,:,:,i)));
end

%% Plot eigenvectors
rel = mrun.lambda;
Z = diag(mrun.stdFeatures);
rel_inv = Z;
if settings.doztr
rel_inv = Z.' * rel * Z;
end
[V, D] = eig(rel_inv,'vector');

% [m,idx] = max(D);
% primEig = V(:,idx);
% primEigIm = autoenc.decode(transpose(primEig));
% subplot(1,2,1);
% imshow(primEigIm,[]);

% primEig = V(:,end-idx);
% primEigIm = autoenc.decode(transpose(primEig));
% subplot(1,2,2);
% imshow(primEigIm);
% 
num = 10;
figure;
for i=1:num
    dec = autoenc.decode(transpose(V(:,i)));
    %im = rescale(dec);
    im=dec;
    subplot(2,5,i);
    imshow(im,[]);
end

