function inv = invSigmoid(x)
%invSigmoid Sigmoid inverse.
    inv = ln(x) - ln(1-x);
end