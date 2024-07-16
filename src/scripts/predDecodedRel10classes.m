% This script can be used to decode the relevance matrix from GMLVQ, use it
% to make predictions and calculate this predictor's accuracy and agreement
% with the "teacher" predictor.

clc;
clear;

%% load the model
load('C:\Users\xrist\Desktop\Uni\master\Thesis\code\models\GMLVQ_FCAE_tanh_MNIST_10runs.mat')


%%
%numClasses = length(settings.classes);
% How many classes we need to distinguish between.
numClasses = 10;
% how many eigenvectors to use to reconstruct the relevance matrix
numEig = 9;
% reverse z-score transformation from relevance matrix
run = result.results(end).run;
Z = diag(run.stdFeatures);
rel = run.lambda;
if settings.doztr
    rel = Z.' * rel * Z;
end

% find eigenvectors and sort
[V, D] = eig(rel, 'vector');
% keep only the real parts of the results. The real parts are negligible in
% every observed case.
V = real(V);
D = real(D);
% Sore eigenvectors by their corresponding eigenvalues.
[d,ind] = sort(D, "descend");
Ds =  D(ind);
Vs = V(:,ind);

clear primEigIm;
clear u;
% pass all eigenvectors through the decoder
for i=1:numEig
    primEigIm(:,:,i) = autoenc.decode(Vs(:,i)');
    u(:,i) = reshape(primEigIm(:,:,i),[],1) * sqrt(Ds(i));
end

% multiply the vector with itself and  scale it by the eigenvalue
rel_dec = u*u';

%%
% undo z-score transformation for prototypes
% decode the prototypes
nPrototypes = size(run.prototypes,1);
prototypes = run.prototypes;

if settings.doztr
    % revert the zscore transfor mation that takes place in the toolbox
    prototypes = run.prototypes .* repmat(run.stdFeatures,nPrototypes,1)...
        + repmat(run.meanFeatures, nPrototypes, 1);
    
end

origPrototypes = autoenc.decode(prototypes);
for i=1:size(origPrototypes,4)
    protVectors(:,i) = reshape(origPrototypes(:,:,:,i),[],1);
end

%% load dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadMNIST(settings.classes);

% rescale
if isfield(settings, "rescaleInput") && settings.rescaleInput
    testImages = rescale(testImages,-1,1);
end
% encode the training data
xencoded = autoenc.encode(testImages);
encodedFeatures = xencoded;
% do z-score transformation
nFeatureVectors = length(xencoded);
encodedFeatures = (xencoded - repmat(run.meanFeatures, nFeatureVectors, 1)) ...
                    ./ repmat(run.stdFeatures, nFeatureVectors, 1);

% convert the labels to the range 1-N
lt = LabelTransformer(unique(testLabels));
transformedLabels = lt.transform(testLabels);

% initialize vectors for predictions
predEncoded = zeros(nFeatureVectors,1);
predDecoded = zeros(nFeatureVectors,1);
%%
for i=1:nFeatureVectors
    % classify in encoded space
    dist = [];
    % calculate the distance of the data point to all prototypes
    for j=1:size(run.prototypes,1)
        dist(j) = GMLVQ_distance(run.lambda,  encodedFeatures(i,:), run.prototypes(j,:));
    end
    
    % find the minimum distance 
    [~,idx] = min(dist);
    % assign the class of the prototype with the minimum distance
    predEncoded(i) = run.gmlvq.plbl(idx);

    % classify in decoded space
    % same as above but in original feature space
    distDec = [];
    trainingVector = reshape(testImages(:,:,:,i),1, []);
    for j=1:size(run.prototypes,1)
        distDec(j) = GMLVQ_distance(rel_dec, trainingVector, protVectors(:,j)');
    end
    [~,idx] = min(distDec);
    predDecoded(i) = run.gmlvq.plbl(idx);
    
end
%% Calculate accuracies and agreement.
encodedAcc = sum(predEncoded == transformedLabels) ...
    / nFeatureVectors

decodedAcc = sum(predDecoded == transformedLabels) ...
    / nFeatureVectors

agreement = sum(predEncoded == predDecoded) / nFeatureVectors