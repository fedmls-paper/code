function [xOut, info] = Scaffnew(A, b, nrIters, pComm0, lr0)
% Scaffnew performs prox-skip for minimizing ||Ax - b||_1
%
%   Input:
%     A         - cell array where A{i} is the local matrix for client i
%     b         - cell array where b{i} is the local vector for client i
%     nrIters   - total number of local iterations to be run
%     pComm0    - initial communication probability, updated as pComm0/sqrt(k)
%     lr0       - initial learning rate, updated as lr0/sqrt(k)
%
%   Output:
%     xOut      - the aggregated model after the final round
%     info      - Struct containing performance metrics (e.g., loss values)


% Number of clients
N = numel(A);

% Initialize the performance struct
info.obj = zeros(nrIters*pComm0*5,1); % Objective value at the end of this round
info.numLS = zeros(nrIters*pComm0*5,1); % Total number local steps after this round
numCommRound = 0; % Communication round counter

% Initialize the global model
xAvg = zeros(size(A{1},2), 1);

% Initialize the local models
x = cell(1, N);
h = cell(1, N);
for i = 1:N, h{i} = 0; end

% Broadcast the aggregated model to all clients
for i = 1:N, x{i} = xAvg; end

% Local iterations
for k = 1:nrIters
    
    % Update learning rate
    lr = lr0 / sqrt(k);
    
    for i = 1:N % for all clients

        % Compute the local subgradient
        g = A{i}' * sign( A{i} * x{i} - b{i} ); 

        % Update the parameters
        x{i} = x{i} - lr * g;

    end
    
    % Communicate with probability pComm0/sqrt(k)
    % Note: we always communicate in the last iteration
    if (rand(1) < pComm0/sqrt(k)) || (k == nrIters) 

        % Aggregation
        xAvg = 0;
        for i = 1:N
            xAvg = xAvg + x{i} / N;
        end

        % Update the communication round counter
        numCommRound = numCommRound + 1; 

        % Compute the objective, update the performance struct
        for i = 1:N
            info.obj(numCommRound) = info.obj(numCommRound) + norm(A{i}*xAvg - b{i},1);
        end
        info.numLS(numCommRound) = k;

        % Broadcast the aggregated model to all clients
        for i = 1:N
            x{i} = xAvg;
        end

    end

    % Update the control variables
    for i = 1:N
        h{i} = h{i} + (pComm0/lr) * (xAvg - x{i});
    end

end

% Model to output
xOut = xAvg;

% Delete the unusued part of performance struct
info.obj(numCommRound+1:end) = [];
info.numLS(numCommRound+1:end) = [];

end