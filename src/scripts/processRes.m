% Custom auxiliary script for quickly generating a summary of the results 
% of GMLVQ trained using an autoencoder that interest us. The autoencoder
% and the results of GMLVQ should already be loaded before execution.

%% Calculate accuracy

disp("Accuracy")
cmatrix = result.averageRun.validationPerf(end).confusionMatrix;
accuracy = sum(diag(cmatrix)) / sum(cmatrix,"all")

% use one of the validation runs as an example
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

%display the prototypes
tiledlayout(1,length(classes));
for i = 1:length(classes) %* settings.prototypesPerClass
    nexttile(i);
    imshow(squeeze(origPrototypes(:,:,:,i)));
end

%% Plot eigenvectors
rel = mrun.lambda;
Z = diag(mrun.stdFeatures);
rel_inv = Z;
% revert z-score transformation
if settings.doztr
    rel_inv = Z.' * rel * Z;
end
[V, D] = eig(rel_inv,'vector');
% keep only the real parts of the results. The imaginary parts are
% negligible in all of the cases we observed.
D = real(D);
V = real(V);

% display the decoded leading eigenvector
[m,idx] = max(D);
primEig = V(:,idx);
primEigIm = autoenc.decode(transpose(primEig));
imshow(primEigIm, []);


% display the decoded eigenvectors
% uncomment to execute.
% num = 10;
% figure;
% for i=1:num
%     dec = autoenc.decode(transpose(V(:,i)));
%     im=dec;
%     subplot(2,5,i);
%     imshow(im);
% end

