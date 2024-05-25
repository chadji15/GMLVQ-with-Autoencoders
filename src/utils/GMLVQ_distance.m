function dist = GMLVQ_distance(relevanceMatrix, v1,v2)
%GMLVQ_distance Calculate the distance between the two vectors based on the
%metric defined by the relevance matrix
dist = (v1-v2) * relevanceMatrix * (v1-v2)';
end