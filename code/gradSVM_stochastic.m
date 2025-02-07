function g = gradSVM_stochastic(A, b, x, mini_batch_ratio)
    % A: n x d data matrix (augmented with a column of ones if using bias)
    % b: n x 1 vector of labels, assumed to be -1 or +1
    % x: d x 1 parameter vector (including bias)
    % mini_batch_ratio: portion of data to be used at every iteration
    
    n = size(A,1);
    batchSize = ceil(n*mini_batch_ratio);
    idxBatch = sort(randperm(n,batchSize));
    

    % Extract the minibatch data:
    A_batch = A(idxBatch, :);  % minibatch features (each row is one sample)
    b_batch = b(idxBatch);       % minibatch labels

    margins = b_batch .* (A_batch * x);         % Compute y_i * (a_i' * x) for each sample
    activeIdx = (margins < 1);            % Indicator for which samples contribute (margin < 1)

    % Compute the subgradient for active samples:
    if any(activeIdx)
        % Sum subgradients for active samples:
        g = -A_batch(activeIdx, :)' * b_batch(activeIdx);
        % Optionally, average the subgradient over the minibatch:
        g = g * (n/batchSize);
    else
        g = zeros(size(x));
    end
        
end