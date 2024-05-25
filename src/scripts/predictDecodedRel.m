clc;
clear;

%% load the model
load('C:\Users\xrist\Desktop\Uni\master\Thesis\code\models\GMVLQ_FCAE_tanh_FashionMNIST.mat')

%%
% reverse z-score transformation from relevance matrix

run = result.results(end).run;
Z = diag(run.stdFeatures);
rel = run.lambda;
if settings.doztr
    rel = Z.' * rel * Z;
    rel = round(rel,4);
end

% find leading eigenvalue and eigenvector
[V, D] = eig(rel, 'vector');

[m,idx] = max(D);
primEig = V(:,idx)';
% pass through decoder
primEigIm = autoenc.decode(primEig);

% reshape the matrix into a vector
u = reshape(primEigIm,[],1);
% multiply the vector with itself and  scale it by the eigenvalue
rel_dec = m * (u * u.');

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

% transform to vectors
class1ProtDec = reshape(squeeze(origPrototypes(:,:,:,1)),1,[]);
class2ProtDec = reshape(squeeze(origPrototypes(:,:,:,2)),1,[]);

%% load dataset

[trainingImages, trainingLabels, testImages, testLabels] = loadDataset(settings.dataset);

% encode the training data
xencoded = autoenc.encode(trainingImages);
encodedFeatures = xencoded;
% do z-score transformation
nFeatureVectors = length(xencoded);
encodedFeatures = (xencoded - repmat(run.meanFeatures, nFeatureVectors, 1)) ...
                    ./ repmat(run.stdFeatures, nFeatureVectors, 1);

% convert the labels to the range 1-N
lt = LabelTransformer(unique(trainingLabels));
transformedLabels = lt.transform(trainingLabels);

predEncoded = zeros(nFeatureVectors,1);
predDecoded = zeros(nFeatureVectors,1);
%%
for i=1:nFeatureVectors
    % classify in encoded space
    dist1Enc = GMLVQ_distance(run.lambda, encodedFeatures(i,:), run.prototypes(1,:));
    dist2Enc = GMLVQ_distance(run.lambda, encodedFeatures(i,:), run.prototypes(2,:));
    if dist1Enc <= dist2Enc
        predEncoded(i) = 1;
    else
        predEncoded(i) = 2;
    end

    % classify in decoded space
    trainingVector = reshape(trainingImages(:,:,:,i),1, []);
    dist1Dec = GMLVQ_distance(rel_dec, trainingVector, class1ProtDec);
    dist2Dec = GMLVQ_distance(rel_dec, trainingVector, class2ProtDec);
    if dist1Dec <= dist2Dec
        predDecoded(i) = 1;
    else
        predDecoded(i) = 2;
    end
end
%%
encodedAcc = sum(predEncoded == transformedLabels) ...
    / nFeatureVectors

decodedAcc = sum(predDecoded == transformedLabels) ...
    / nFeatureVectors

agreement = sum(predEncoded == predDecoded) / nFeatureVectors