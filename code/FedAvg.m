function [xOut, info] = FedAvg(A, b, obj, grad, nrComRnd, nrLcStep0, lr0)
% FedAvg performs federated averaging
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
info.nrComRnd = nrComRnd;
info.lr0 = lr0;
info.nrLcStep0 = nrLcStep0;
info.obj = zeros(nrComRnd,1); % Objective value at the end of this round
info.time = zeros(nrComRnd,1); % wall clock time at the end of this round
info.numLS = zeros(nrComRnd,1); % Total number local steps after this round
numLS = 0; % Total number local steps is initialized to 0

% Initialize the global model
xAvg = zeros(size(A{1},2), 1);

% Initialize the local models
x = cell(1, N);

% initiate timer
timestart = tic;

% Communication rounds
for t = 1:nrComRnd

    % Broadcast the aggregated model to all clients
    for i = 1:N
        x{i} = xAvg;
    end

    % Update learning rate
    lr = lr0 / sqrt(t);

    % Compute number of local steps to run this round
    nrLcStep = ceil(nrLcStep0*t);

    % Local steps
    for i = 1:N

        for k = 1:nrLcStep

            % Compute the local subgradient
            g = grad(A{i}, b{i}, x{i});

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
        info.obj(t) = info.obj(t) + obj(A{i}, b{i}, xAvg); 
    end
    info.numLS(t) = numLS;
    info.time(t) = toc(timestart);

    % Print progress every 100 communication rounds
    if mod(t, round(nrComRnd/10)) == 0
        fprintf('Round %d: Objective = %f, Total local steps = %d, Elapsed time = %.2f sec\n', ...
            t, info.obj(t), numLS, info.time(t));
    end
    
end

% Model to output
xOut = xAvg;

end


