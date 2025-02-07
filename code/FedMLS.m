function [xOut, info] = FedMLS(A, b, obj, grad, nrComRnd, nrLcStep0, lambda0)
% FedMLS performs federated multiple local steps
%
%   Input:
%     A         - cell array where A{i} is the local matrix for client i
%     b         - cell array where b{i} is the local vector for client i
%     nrComRnd  - number of communication rounds (outer iterations)
%     nrLcStep0 - initial number of local steps, updated as nrLcStep0*t
%     lambda0   - initial penalty parameter, updated as lambda0/t
%
%   Output:
%     xOut      - the aggregated model after the final round
%     info      - Struct containing performance metrics (e.g., loss values)
    
% Number of clients
n = numel(A);

% Initialize the performance struct
info.nrComRnd = nrComRnd;
info.lambda0 = lambda0;
info.nrLcStep0 = nrLcStep0;
info.obj = zeros(nrComRnd,1); % Objective value at the end of this round
info.time = zeros(nrComRnd,1); % wall clock time at the end of this round
info.numLS = zeros(nrComRnd,1); % Total number local steps after this round
numLS = 0; % Total number local steps is initialized to 0

% Initialize the global models
y = zeros(size(A{1},2), 1);
z = y;
w = y;

% Initialize the local models
yi = cell(1, n);
for i = 1:n, yi{i} = y; end
zi = yi;
wi = yi;

% initiate timer
timestart = tic;

% Communication rounds
for t = 1:nrComRnd
    
    % Update the parameters
    lambda = lambda0/(t);
    gamma = 2/(t+1);
    gamma_next = 2/(t+2);
    beta = 4/(lambda*t);
    beta_next = 4/(lambda*(t+1));

    % Compute number of local steps to run this round
    nrLcStep = ceil(nrLcStep0 * t);

    % Local steps
    for i = 1:n

        vi = zi{i} - 1/(beta*lambda) * (yi{i} - y);
        ui = zi{i};
        ui_tilde = zi{i};

        for k = 1:nrLcStep
            g = grad(A{i}, b{i}, ui);
            u_hat = ui - 2/(2+k) * ( (1/(n*beta))*g + ui - vi );
            ui = u_hat;
            theta = (2*k+2)/(k^2+3*k);
            ui_tilde = (1-theta) * ui_tilde + theta * ui;
        end

        zi{i} = ui;
        z_tilde = ui_tilde;
        wi{i} = (1-gamma)*wi{i} + gamma*z_tilde;
        yi{i} = (1-gamma_next)*wi{i} + gamma_next*zi{i};

    end
    
    % Update total number of local steps so far
    numLS = numLS + k;

    % Aggregation
    yAvg = 0;
    for i = 1:n
        yAvg = yAvg + yi{i} / n;
    end
    
    % Server steps
    w = (1-gamma)*w + gamma*z;
    y = (1-gamma_next)*w + gamma_next*z;
    z = z - 1/(beta_next*lambda) * (y - yAvg);
    
    % Compute the objective, update the performance struct
    for i = 1:n
        info.obj(t) = info.obj(t) + obj(A{i}, b{i}, w);
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
xOut = w;

end