function parameter = initializeZeros(sz,className)
% Weight initializer for deep learning toolbox. Initializes weights of
% given dimensions to zero.
arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter);

end