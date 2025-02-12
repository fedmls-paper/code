function [xOut, info] = Scaffold(A, b, obj, grad, nrComRnd, nrLcStep0, option, lrg0, lrl0)
% Scaffold: Stochastic controlled averaging for federated learning
%
%   Input:
%     A         - cell array where A{i} is the local matrix for client i
%     b         - cell array where b{i} is the local vector for client i
%     nrComRnd  - number of communication rounds (outer iterations)
%     nrLcStep  - number of local gradient steps per communication round
%     option    - Option 1 or 2 from the paper
%     lrg0      - initial global learning rate
%     lrl0      - initial local learning rate, updated as lrl0 / sqrt(r)
%
%   Output:
%     xOut      - the aggregated model after the final round
%     info      - Struct containing performance metrics (e.g., loss values)

% Number of clients
N = numel(A);

% Initialize the performance struct
info.nrComRnd = nrComRnd;
info.nrLcStep0 = nrLcStep0;
info.option = option;
info.lrg0 = lrg0;
info.lrl0 = lrl0;
info.obj = zeros(nrComRnd, 1); % Objective value at the end of this round
info.time = zeros(nrComRnd, 1); % wall clock time at the end of this round
info.numLS = zeros(nrComRnd, 1); % Total number local steps after this round

numLS = 0; % Total number local steps is initialized to 0
numCommRound = 0; % Communication round counter

% Initialize the global model
xAvg = zeros(size(A{1}, 2), 1);
cAvg = zeros(size(A{1}, 2), 1);
% dxAvg = zeros(size(A{1}, 2), 1);
% dcAvg = zeros(size(A{1}, 2), 1);

% Initialize the local models
x = cell(1, N);
c = cell(1, N);
dy = cell(1, N);
dc = cell(1, N);
for i = 1:N
    x{i} = 0;
    c{i} = 0;
end

% Broadcast the aggregated model to all clients
for i = 1:N
    x{i} = xAvg;
end

lrg = lrg0;

% initiate timer
timestart = tic;

% Communication rounds
for r = 1:nrComRnd

    % Update learning rate
%     lrl = lrl0 / sqrt(r);
    lrl = lrl0 / (r * lrg);  % Alp's variant, similar to paper page 5.

    % Compute number of local steps to run this round
    nrLcStep = ceil(nrLcStep0 * r);

    % For all clients
    for i = 1:N
        y_i = xAvg;

        % Local iterations
        for k = 1:nrLcStep
            % Compute the local subgradient
            g_y = grad(A{i}, b{i}, y_i);

            y_i = y_i - lrl * (g_y - c{i} + cAvg);
        end

        if option == 1
            g_x = grad(A{i}, b{i}, xAvg);
            c_plus = g_x;
        elseif option == 2
            c_plus = c{i} - cAvg + (1 / (nrLcStep * lrl)) * (xAvg - y_i);
        else
            fprintf(['Wrong option ', option, '!!']);
        end

        dy{i} = y_i - xAvg;
        dc{i} = c_plus - c{i};

        c{i} = c_plus;
    end

    % Update total number of local steps so far
    numLS = numLS + nrLcStep;

    % Aggregation
    dxAvg = 0;
    dcAvg = 0;
    for i = 1:N
        dxAvg = dxAvg + dy{i} / N;
        dcAvg = dcAvg + dc{i} / N;
    end

    xAvg = xAvg + lrg * dxAvg;
    cAvg = cAvg + 1 * dcAvg; % The 1 is from partial participation, |S| / N

    % Compute the objective, update the performance struct
    for i = 1:N
        info.obj(r) = info.obj(r) + obj(A{i}, b{i}, xAvg);
    end
    info.numLS(r) = numLS;
    info.time(r) = toc(timestart);

    % Print progress every 100 communication rounds
    if mod(r, 100) == 0
        fprintf('Round %d: Objective = %f, Total local steps = %d, Elapsed time = %.2f sec\n', ...
            r, info.obj(r), nrLcStep, info.time(r));
    end    
end

% Model to output
xOut = xAvg;

% Delete the unusued part of performance struct
info.obj(nrComRnd+1:end) = [];
info.time(nrComRnd+1:end) = [];
info.numLS(nrComRnd+1:end) = [];

end
