function g = gradSVM(A, b, x)
    % A: n x d data matrix (augmented with a column of ones if using bias)
    % b: n x 1 vector of labels, assumed to be -1 or +1
    % x: d x 1 parameter vector (including bias)

    margins = b .* (A * x);         % Compute y_i * (a_i' * x) for each sample
    activeIdx = (margins < 1);      % Indicator for which samples contribute (margin < 1)
    
    % Compute the subgradient for active samples:
    if any(activeIdx)
        % Sum subgradients for active samples:
        g = -A(activeIdx, :)' * b(activeIdx);
    else
        g = zeros(size(x));
    end
    
end