function [xOut, info] = SubgradDescent(A, b, obj, grad, maxit, lr0)
% FedAvg performs federated averaging
%
%   Input:
%     A         - cell array where A{i} is the local matrix for client i
%     b         - cell array where b{i} is the local vector for client i
%     nrComRnd  - number of communication rounds (outer iterations)
%     lr0       - initial learning rate, updated as lr0/sqrt(t)
%
%   Output:
%     xOut      - the aggregated model after the final round
%     info      - Struct containing performance metrics (e.g., loss values)


% Initialize the performance struct
info.nrComRnd = maxit;
info.lr0 = lr0;
info.obj = zeros(maxit,1); % Objective value at the end of this round
info.time = zeros(maxit,1); % wall clock time at the end of this round

% Initialize the global model
x = zeros(size(A,2), 1);

% initiate timer
timestart = tic;

% Communication rounds
for t = 1:maxit

    % Update learning rate
    lr = lr0 / sqrt(t);

    % Compute the local subgradient
    g = grad(A, b, x);

    % Update the parameters
    x = x - lr * g;

    % Compute the objective, update the performance struct
    info.obj(t) = obj(A, b, x); 
    info.time(t) = toc(timestart);

end

% Model to output
xOut = x;

end


