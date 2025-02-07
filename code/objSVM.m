function obj = objSVM(A,b,x)
    obj = sum(max(0, 1 - b .* (A * x)));
end

