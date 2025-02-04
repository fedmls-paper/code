function [xOut, info] = FedAvg(A, b, nrComRnd, nrLcStep0, lr0)
% FedAvg performs federated averaging for minimizing ||Ax - b||_1
%
%   Input:
%     A         - cell array where A{i} is the local matrix for client i
%     b         - cell array where b{i} is the local vector for client i
%     nrComRnd  - number of communication rounds (outer iterations)
%     nrLcStep  - number of local gradient steps per communication round
%     lr0       - initial learning rate, updated as lr0/sqrt(t)
%
%   Output:
%     xOut      - the aggregated model after the final round
%     info      - Struct containing performance metrics (e.g., loss values)

% Number of clients
N = numel(A);

% Initialize the performance struct
info.obj = zeros(nrComRnd,1); % Objective value at the end of this round
info.numLS = zeros(nrComRnd,1); % Total number local steps after this round
numLS = 0; % Total number local steps is initialized to 0

% Initialize the global model  
xAvg = zeros(size(A{1},2), 1);

% Initialize the local models
x = cell(1, N);

% Communication rounds
for t = 1:nrComRnd

    % Broadcast the aggregated model to all clients
    for i = 1:N
        x{i} = xAvg;
    end

    % Update learning rate
    lr = lr0 / sqrt(t);
    
    % Compute number of local steps to run this round
    nrLcStep = ceil(nrLcStep0 * t);

    % Local steps
    for i = 1:N

        for k = 1:nrLcStep

            % Compute the local subgradient
            g = A{i}' * sign( A{i} * x{i} - b{i} );

            % Update the parameters
            x{i} = x{i} - lr * g;

        end
        
    end

    % Update total number of local steps so far
    numLS = numLS + k;

    % Aggregation
    xAvg = 0;
    for i = 1:N
        xAvg = xAvg + x{i} / N;
    end
    
    % Compute the objective, update the performance struct
    for i = 1:N
        info.obj(t) = info.obj(t) + norm(A{i}*xAvg - b{i},1);
    end
    info.numLS(t) = numLS;
    
end

% Model to output
xOut = xAvg;

end

