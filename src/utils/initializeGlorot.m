function weights = initializeGlorot(sz,numOut,numIn,className)
%initializeGlorot Glorot weight initializer for deep learning toolbox
arguments
    sz
    numOut
    numIn
    className = 'single'
end

Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end